"""
Author: Salva Rühling Cachay
"""

import numpy as np
import pandas as pd
import xarray as xa
import torch
from torch.utils.data import DataLoader

from utilities.utils import get_index_mask


class Xs_Dataset(torch.utils.data.Dataset):
    def __init__(self, X_atm, labels, X_oc=None):
        self.X_atm = torch.tensor(X_atm).float()
        self.X_oc = torch.tensor(X_oc if X_oc is not None else X_atm).float()  # Default X_oc = X_atm
        self.labels = torch.tensor(labels).float()

    def __getitem__(self, i):
        return self.X_atm[i], self.X_oc[i], self.labels[i]

    def __len__(self):
        return self.X_atm.shape[0]


def to_dataloaders(cmip5_oc, soda, godas, cmip5_atm, era5, noaa, 
                   batch_size, valid_split=0, validation=['SODA'], verbose=True,
                   concat_cmip5_and_soda=True, shuffle_training=True, separate_inputs=False):
    """
     n - length of time series (i.e. dataset size)
     m - number of nodes/grid cells (33 if using exactly the ONI region)
    """

    sodaX = np.array(soda[0]) if not isinstance(soda[0], np.ndarray) else soda[0]
    cmip5_ocX = np.array(cmip5_oc[0]) if not isinstance(cmip5_oc[0], np.ndarray) else cmip5_oc[0]
    godasX = np.array(godas[0]) if not isinstance(godas[0], np.ndarray) else godas[0]

    noaaX = np.array(noaa[0]) if not isinstance(noaa[0], np.ndarray) else noaa[0]
    cmip5_atmX = np.array(cmip5_atm[0]) if not isinstance(cmip5_atm[0], np.ndarray) else cmip5_atm[0]
    era5X = np.array(era5[0]) if not isinstance(era5[0], np.ndarray) else era5[0]
    

    # Split into atm & oc if separate_inputs=True, otherwise keep the same
    if not separate_inputs:
        noaaX = sodaX
        cmip5_atmX = cmip5_ocX
        era5X = godasX

    if concat_cmip5_and_soda:  # instead of transfer, concat the cmip5 and soda data
        if validation.lower() == 'cmip5':
            first_val = min(len(godas[1]) * 2, 600)
            cmip5_atm_trainX, cmip5_oc_trainX, cmip5_trainY = cmip5_atmX[:-first_val], cmip5_ocX[:-first_val], cmip5_oc[1][:-first_val]
            validX_atm, validX_oc, validY = cmip5_atmX[-first_val:], cmip5_ocX[-first_val:], cmip5_oc[1][-first_val:]
            noaa_trainX, soda_trainX, soda_trainY = noaaX, sodaX, soda[1]

        elif 'soda' in validation.lower():
            cmip5_atm_trainX, cmip5_oc_trainX, cmip5_trainY = cmip5_atmX, cmip5_ocX, cmip5_oc[1]
            if valid_split > 0:
                first_val = int(valid_split * len(sodaX))
                noaa_trainX, soda_trainX, soda_trainY = noaaX[:-first_val], sodaX[:-first_val], soda[1][:-first_val]
                if validation.lower() == 'soda':
                    validX_atm, validX_oc, validY = noaaX[-first_val:], sodaX[-first_val:], soda[1][-first_val:]
                else:
                    validX_atm, validX_oc, validY = noaaX, sodaX, soda[1]
            else:  # without val. set, just return the SODA set
                noaa_trainX, soda_trainX, soda_trainY = noaaX, sodaX, soda[1]
                validX_atm, validX_oc, validY = noaaX, sodaX, soda[1]
        else:
            raise ValueError('Validation dataset must be either CMIP5 or SODA')

        trainX_atm = np.concatenate((cmip5_atm_trainX, noaa_trainX), axis=0)
        trainX_oc = np.concatenate((cmip5_oc_trainX, soda_trainX), axis=0)
        trainY = np.concatenate((cmip5_trainY, soda_trainY), axis=0)
    else:
        print("Only SODA for training!")
        first_val = int(valid_split * len(soda[0]))
        trainX_atm, trainX_oc, trainY = noaaX[:-first_val], sodaX[:-first_val], soda[1][:-first_val]
        validX_atm, validX_oc, validY = noaaX[-first_val:], sodaX[-first_val:], soda[1][-first_val:]

    # Create dataset objects
    trainset = Xs_Dataset(trainX_atm, trainY, X_oc=trainX_oc)
    valset = Xs_Dataset(validX_atm, validY, X_oc=validX_oc) if validX_atm is not None else []
    testset = Xs_Dataset(era5X, godas[1], X_oc=godasX)

    if verbose:
        print('Train set:', len(trainset), 'Validation set:', len(valset), 'Test set:', len(testset))

    # Create DataLoaders
    train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_training, pin_memory=True)
    test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val = None if valset == [] else DataLoader(valset, batch_size=batch_size, shuffle=False)

    del trainset, valset, testset
    return train, val, test


def reformat_cnn_data(lead_months=3, window=3, use_heat_content=False,
                      lon_min=0, lon_max=360,
                      lat_min=-55, lat_max=60,
                      data_dir="data/",
                      # sample_file='CMIP5.input.36mn.1861_2001_og.nc',  # Input of training set
                      sample_file = 'CMIP5.input.36mn.1861_2001.nc',
                      # label_file='CMIP5.label.mldgron.12mn_3mv.1863_2003.nc',  # Label of training set
                      label_file = 'CMIP5.label.nino34.12mn_3mv.1863_2003.nc',
                      sst_key="sst",
                      get_valid_nodes_mask=False,
                      get_valid_coordinates=False
                      ):
    """
    :param lon_min, lon_max, lat_min, lat_max: all inclusive
    """
    import pandas as pd
    lat_p1, lat_p2 = int((lat_min + 55) / 5), min(int((lat_max + 55) / 5), 23)
    lon_p1, lon_p2 = int(lon_min / 5), min(int(lon_max / 5), 71)
    data = xa.open_dataset(f'{data_dir}/{sample_file}')
    labels = xa.open_dataset(f'{data_dir}/{label_file}')
    # Shape T' x 36 x |lat| x |lon|, want : T x 12 x |lat| x |lon|
    lat_sz = lat_p2 - lat_p1 + 1
    lon_sz = lon_p2 - lon_p1 + 1
    features = 2 if use_heat_content else 1
    feature_names = ["sst", "heat_content"] if use_heat_content else ["sst"]

    filtered_region = data.sel(
        {'lat': slice(lat_min, lat_max), 'lon': slice(lon_min, lon_max)}
    )
    filtered_region = filtered_region.rename({"lev": "window", "time": "year"})  # rename coordinate name
    X_all_target_mons = np.empty((data.sizes["time"], 12, features, window, lat_sz, lon_sz))
    Y_all_target_mons = np.empty((data.sizes["time"], 12))
    tg_mons = np.arange(0, 12)
    X_all_target_mons = xa.DataArray(X_all_target_mons, coords=[("year", data.get_index("time")),
                                                                ("tg-mon", tg_mons),
                                                                ("feature", feature_names),
                                                                ("window", np.arange(1, window + 1)),
                                                                ("lat", filtered_region.get_index("lat")),
                                                                ("lon", filtered_region.get_index("lon"))
                                                                ])
    if "CMIP5" not in label_file:
        X_all_target_mons.attrs["time"] = \
            [pd.Timestamp("1982-01-01") + pd.DateOffset(months=d_mon) for d_mon in
             range(len(data.get_index("time")) * 12)]

    X_all_target_mons.attrs["Lons"] = filtered_region.get_index('lon')
    X_all_target_mons.attrs["Lats"] = filtered_region.get_index('lat')
    print(f"file {sample_file} has {data.dims}")
    print(f"file {label_file} has {labels.dims}")
    for target_month in range(0, 12):
        '''
        target months are indices [25, 36)
        possible predictor months (for lead months<=24) are indices [0, 24]
        '''
        var_dict = {"ld_mn2": int(25 - lead_months + target_month) + 1,
                    "ld_mn1": int(25 - lead_months + target_month) + 1 - window}
        X_all_target_mons.loc[:, target_month, "sst", :, :, :] = \
            filtered_region.variables[sst_key][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        if use_heat_content:
            X_all_target_mons.loc[:, target_month, "heat_content", :, :, :] = \
                filtered_region.variables['t300'][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]

        Y_all_target_mons[:, target_month] = labels.variables['pr'][:, target_month, 0, 0]
    X_all_target_mons = X_all_target_mons.stack(time=["year", "tg-mon"])

    Y_time_flattened = Y_all_target_mons.reshape((-1,))
    X_node_flattened = X_all_target_mons.stack(cord=["lat", "lon"])
    X_time_and_node_flattened = X_node_flattened.transpose("time", "feature", "window", "cord")

    if get_valid_nodes_mask:
        sea = (np.count_nonzero(X_time_and_node_flattened[:, 0, 0, :], axis=0) > 0)
        if get_valid_coordinates:
            return X_time_and_node_flattened, Y_time_flattened, sea, X_time_and_node_flattened.get_index("cord")
        return X_time_and_node_flattened, Y_time_flattened, sea

    return X_time_and_node_flattened, Y_time_flattened

def reformat_atm_data(lead_months=3, window=3,
                      lon_min=0, lon_max=360,
                      lat_min=-55, lat_max=60,
                      data_dir="data/",
                      sample_file='CMIP5.input.36mn_3mv.1861_2001.nc',  # Input of training set
                      # label_file='CMIP5.label.mldgron.12mn_3mv.1863_2003.nc',  # Label of training set
                      label_file='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                      sp_key="sp",
                      get_valid_nodes_mask=False,
                      get_valid_coordinates=False
                      ):
    """
    :param lon_min, lon_max, lat_min, lat_max: all inclusive
    """
    import pandas as pd
    print(sample_file)
    lat_p1, lat_p2 = int((lat_min + 55) / 5), min(int((lat_max + 55) / 5), 23)
    lon_p1, lon_p2 = int(lon_min / 5), min(int(lon_max / 5), 71)
    data = xa.open_dataset(f'{data_dir}/{sample_file}')
    labels = xa.open_dataset(f'{data_dir}/{label_file}')
    # Shape T' x 36 x |lat| x |lon|, want : T x 12 x |lat| x |lon|
    lat_sz = lat_p2 - lat_p1 + 1
    lon_sz = lon_p2 - lon_p1 + 1
    features = 1
    feature_names = ["sp"]

    filtered_region = data.sel(
        {'lat': slice(lat_min, lat_max), 'lon': slice(lon_min, lon_max)}
    )
    filtered_region = filtered_region.rename({"lev": "window", "time": "year"})  # rename coordinate name
    X_all_target_mons = np.empty((data.sizes["time"], 12, features, window, lat_sz, lon_sz))
    Y_all_target_mons = np.empty((data.sizes["time"], 12))
    tg_mons = np.arange(0, 12)
    
    X_all_target_mons = xa.DataArray(X_all_target_mons, coords=[("year", data.get_index("time")),
                                                                ("tg-mon", tg_mons),
                                                                ("feature", feature_names),
                                                                ("window", np.arange(1, window + 1)),
                                                                ("lat", filtered_region.get_index("lat")),
                                                                ("lon", filtered_region.get_index("lon"))
                                                                ])
    if "CMIP5" not in label_file:
        X_all_target_mons.attrs["time"] = \
            [pd.Timestamp("1982-01-01") + pd.DateOffset(months=d_mon) for d_mon in
             range(len(data.get_index("time")) * 12)]

    X_all_target_mons.attrs["Lons"] = filtered_region.get_index('lon')
    X_all_target_mons.attrs["Lats"] = filtered_region.get_index('lat')
    for target_month in range(0, 12):
        '''
        target months are indices [25, 36)
        possible predictor months (for lead months<=24) are indices [0, 24]
        '''
        var_dict = {"ld_mn2": int(25 - lead_months + target_month) + 1,
                    "ld_mn1": int(25 - lead_months + target_month) + 1 - window}
        X_all_target_mons.loc[:, target_month, "sp", :, :, :] = \
            filtered_region.variables[sp_key][:, var_dict["ld_mn1"]:var_dict["ld_mn2"], :, :]
        
        Y_all_target_mons[:, target_month] = labels.variables['pr'][:, target_month, 0, 0]
    X_all_target_mons = X_all_target_mons.stack(time=["year", "tg-mon"])

    Y_time_flattened = Y_all_target_mons.reshape((-1,))
    X_node_flattened = X_all_target_mons.stack(cord=["lat", "lon"])
    X_time_and_node_flattened = X_node_flattened.transpose("time", "feature", "window", "cord")

    if get_valid_nodes_mask:
        sea = (np.count_nonzero(X_time_and_node_flattened[:, 0, 0, :], axis=0) > 0)
        if get_valid_coordinates:
            return X_time_and_node_flattened, Y_time_flattened, sea, X_time_and_node_flattened.get_index("cord")
        return X_time_and_node_flattened, Y_time_flattened, sea

    return X_time_and_node_flattened, Y_time_flattened

# def load_cnn_data(lead_months=3, window=3, use_heat_content=False,
#                   lon_min=0, lon_max=359,
#                   lat_min=-55, lat_max=60,
#                   data_dir="data/",
#                   cmip5_data='CMIP5.input.36mn.1861_2001.nc',  # Input of CMIP5 training set
#                   cmip5_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
#                   soda_data='SODA.input.36mn.1871_1970.nc',  # Input of SODA training set
#                   soda_label='SODA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
#                   godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS training set
#                   godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of training set
#                   truncate_GODAS=True,  # whether to truncate it to the 1984-2017 period the CNN paper used
#                   return_new_coordinates=False,
#                   return_mask=False,
#                   add_index_node=False,
#                   verbose=True, **kwargs
#                   ):
#     """

#     :param lead_months:
#     :param window:
#     :param use_heat_content:
#     :param lon_min:
#     :param lon_max:
#     :param lat_min:
#     :param lat_max:
#     :param data_dir:
#     :param cmip5_data:
#     :param cmip5_label:
#     :param soda_data:
#     :param soda_label:
#     :param godas_data:
#     :param godas_label:
#     :param truncate_GODAS:
#     :param return_new_coordinates:
#     :param return_mask:
#     :param target_months: if "all", the model will need to learn to give predictions for any target months,
#                             if an int in [1, 12], it can focus on that specific target month/season,
#                             where 1 translates to "JFM", ..., 12 to "DJF"
#     :return:
#     """
#     cmip5, cmip5_Y, m1 = reformat_cnn_data(lead_months=lead_months, window=window,
#                                            use_heat_content=use_heat_content,
#                                            lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                            data_dir=data_dir + "CMIP5_CNN/", sst_key="sst1",
#                                            sample_file=cmip5_data, label_file=cmip5_label,
#                                            get_valid_nodes_mask=True, get_valid_coordinates=False)
#     SODA, SODA_Y, m2 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
#                                          lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                          data_dir=data_dir + "SODA/", sample_file=soda_data, label_file=soda_label,
#                                          get_valid_nodes_mask=True, get_valid_coordinates=False)
#     GODAS, GODAS_Y, m3 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
#                                            lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                            data_dir=data_dir + "GODAS/", sample_file=godas_data, label_file=godas_label,
#                                            get_valid_nodes_mask=True, get_valid_coordinates=False)
#     if truncate_GODAS:
#         start_1984 = 24  # 2 * 12
#         GODAS, GODAS_Y, GODAS.attrs["time"] = GODAS[start_1984:], GODAS_Y[start_1984:], GODAS.attrs["time"][start_1984:]
#     # DUE to variations due to resolution = 5deg., there are some inconsistencies in which nodes are terrestrial
#     final_mask = np.logical_and(m1, np.logical_and(m2, m3))
#     cmip5, SODA, GODAS = cmip5[:, :, :, final_mask], SODA[:, :, :, final_mask], GODAS[:, :, :, final_mask]
#     if add_index_node:
#         cmip5, SODA, GODAS = add_ONI_node(cmip5), add_ONI_node(SODA), add_ONI_node(GODAS)
#         # cords = np.array(list(cords) + [(0, 205)])  # add coordinate for ONI
#         final_mask = np.append(final_mask, True)  # add
#         if verbose:
#             print('MASKING OUT', np.count_nonzero(np.logical_not(final_mask)), 'nodes')
#     cords = GODAS.indexes['cord']
#     if return_new_coordinates:
#         # cords = cords[final_mask]
#         if return_mask:
#             return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords, final_mask
#         return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), cords
#     if return_mask:
#         return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), final_mask
#     return (cmip5, cmip5_Y), (SODA, SODA_Y), (GODAS, GODAS_Y)

def load_data(lead_months=3, window=3, use_heat_content=False,
                  lon_min=0, lon_max=359,
                  lat_min=-55, lat_max=60,
                  data_dir="data/",
                  # cmip5_oc_data='CMIP5.input.36mn.1861_2001.nc',  # Input of CMIP5_oc training set 
                  cmip5_oc_data='CMIP5.input.36mn.1861_2001_og.nc',  # Input of CMIP5_oc training set 
                  cmip5_oc_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                  # cmip5_oc_label='CMIP5.label.mldgron.12mn_3mv.1863_2003.nc',  # Label of training set
              
                  cmip5_atm_data='CMIP5.input.36mn_3mv.1861_2001.nc',  # Input of CMIP5_atm training set
                  cmip5_atm_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
                  # cmip5_atm_label='CMIP5.label.mldgron.12mn_3mv.1863_2003.nc',  # Label of training set
              
                  soda_data='SODA.input.36mn.1871_1970.nc',  # Input of SODA training set
                  soda_label='SODA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
                  # soda_label='SODA.label.mldgron.12mn_3mv.1873_1972.nc',  # Label of training set
              
                  noaa_data='NOAA.input.36mn_3mv.1871_1970.nc',  # Input of NOAA training set
                  noaa_label='NOAA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
                  # noaa_label='NOAA.label.mldgron.12mn_3mv.1873_1972.nc',  # Label of training set
              
                  godas_data='GODAS.input.36mn.1980_2015.nc',  # Input of GODAS training set
                  # godas_data='GODAS.input.36mn.1980_2012.nc',  # Input of GODAS training set
                  godas_label='GODAS.label.12mn_3mv.1982_2017.nc',  # Label of training set
                  # godas_label='GODAS.label.mldgron.12mn_3mv.1982_2014.nc',  # Label of training set
              
                  # era5_data='ERA5.input.36mn_3mv.1980_2012.nc',  # Input of ERA5 training set
                  era5_data='ERA5.input.36mn_3mv.1980_2015.nc',  # Input of ERA5 training set
                  era5_label='ERA5.label.12mn_3mv.1982_2017.nc',  # Label of training set
                  # era5_label='ERA5.label.mldgron.12mn_3mv.1982_2014.nc',  # Label of training set
              
                  truncate_GODAS=True,  # whether to truncate it to the 1984-2017 period the CNN paper used
                  return_new_coordinates=False,
                  return_mask=False,
                  add_index_node=False,
                  verbose=True, **kwargs
                  ):
    """

    :param lead_months:
    :param window:
    :param use_heat_content:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param data_dir:
    :param cmip5_oc_data:
    :param cmip5_oc_label:
    :param cmip5_atm_data:
    :param cmip5_atm_label:
    :param soda_data:
    :param soda_label:
    :param noaa_data:
    :param noaa_label:
    :param godas_data:
    :param godas_label:
    :param era5_data:
    :param era5_label:
    :param truncate_GODAS:
    :param return_new_coordinates:
    :param return_mask:
    :param target_months: if "all", the model will need to learn to give predictions for any target months,
                            if an int in [1, 12], it can focus on that specific target month/season,
                            where 1 translates to "JFM", ..., 12 to "DJF"
    :return:
    """
    cmip5_oc, cmip5_oc_Y, m1 = reformat_cnn_data(lead_months=lead_months, window=window,
                                           use_heat_content=use_heat_content,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "CMIP5_CNN/", sst_key="sst1",
                                           sample_file=cmip5_oc_data, label_file=cmip5_oc_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)
    SODA, SODA_Y, m2 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                         lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                         data_dir=data_dir + "SODA/", sample_file=soda_data, label_file=soda_label,
                                         get_valid_nodes_mask=True, get_valid_coordinates=False)
    GODAS, GODAS_Y, m3 = reformat_cnn_data(lead_months=lead_months, window=window, use_heat_content=use_heat_content,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "GODAS/", sample_file=godas_data, label_file=godas_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)
    cmip5_atm, cmip5_atm_Y, m4 = reformat_atm_data(lead_months=lead_months, window=window,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "CMIP5/", sp_key="sp",
                                           sample_file=cmip5_atm_data, label_file=cmip5_atm_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)
    NOAA, NOAA_Y, m5 = reformat_atm_data(lead_months=lead_months, window=window,
                                         lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                         data_dir=data_dir + "NOAA/", sample_file=noaa_data, label_file=noaa_label,
                                         get_valid_nodes_mask=True, get_valid_coordinates=False)
    ERA5, ERA5_Y, m6 = reformat_atm_data(lead_months=lead_months, window=window,
                                           lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
                                           data_dir=data_dir + "ERA5/", sample_file=era5_data, label_file=era5_label,
                                           get_valid_nodes_mask=True, get_valid_coordinates=False)

    if truncate_GODAS:
        start_1984 = 24  # 2 * 12
        GODAS, GODAS_Y, GODAS.attrs["time"] = GODAS[start_1984:], GODAS_Y[start_1984:], GODAS.attrs["time"][start_1984:]
        ERA5, ERA5_Y, ERA5.attrs["time"] = ERA5[start_1984:], ERA5_Y[start_1984:], ERA5.attrs["time"][start_1984:]
    # DUE to variations due to resolution = 5deg., there are some inconsistencies in which nodes are terrestrial
    final_mask = np.logical_and.reduce([m1, m2, m3, m4, m5, m6])
    # cmip5_oc, SODA, GODAS = cmip5_oc[:, :, :, final_mask], SODA[:, :, :, final_mask], GODAS[:, :, :, final_mask]
    # cmip5_atm, NOAA, ERA5 = cmip5_atm[:, :, :, final_mask], NOAA[:, :, :, final_mask], ERA5[:, :, :, final_mask]
    if add_index_node:
        cmip5_oc, SODA, GODAS = add_ONI_node(cmip5_oc), add_ONI_node(SODA), add_ONI_node(GODAS)
        cmip5_atm, NOAA, ERA5 = add_ONI_node(cmip5_atm), add_ONI_node(NOAA), add_ONI_node(ERA5)
        # cords = np.array(list(cords) + [(0, 205)])  # add coordinate for ONI
        final_mask = np.append(final_mask, True)  # add
        if verbose:
            print('MASKING OUT', np.count_nonzero(np.logical_not(final_mask)), 'nodes')
    cords = GODAS.indexes['cord']
    if return_new_coordinates:
        # cords = cords[final_mask]
        if return_mask:
            return (cmip5_oc, cmip5_oc_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), (cmip5_atm, cmip5_atm_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y) , cords, final_mask
        return (cmip5_oc, cmip5_oc_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), (cmip5_atm, cmip5_atm_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y), cords
    if return_mask:
        return (cmip5_oc, cmip5_oc_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), (cmip5_atm, cmip5_atm_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y), final_mask
    return (cmip5_oc, cmip5_oc_Y), (SODA, SODA_Y), (GODAS, GODAS_Y), (cmip5_atm, cmip5_atm_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y)


# def load_atm_data(lead_months=3, window=3,
#                   lon_min=0, lon_max=359,
#                   lat_min=-55, lat_max=60,
#                   data_dir="data/",
#                   cmip5_data='CMIP5.input.36mn_3mv.1861_2001',  # Input of CMIP5 training set
#                   cmip5_label='CMIP5.label.nino34.12mn_3mv.1863_2003.nc',  # Label of training set
#                   noaa_data='NOAA.input.36mn.1871_1970.nc',  # Input of NOAA training set
#                   noaa_label='NOAA.label.nino34.12mn_3mv.1873_1972.nc',  # Label of training set
#                   era5_data='ERA5.input.36mn.1980_2015.nc',  # Input of ERA5 training set
#                   era5_label='ERA5.label.12mn_3mv.1982_2017.nc',  # Label of training set
#                   truncate_ERA5=True,  # whether to truncate it to the 1984-2017 period the CNN paper used
#                   return_new_coordinates=False,
#                   return_mask=False,
#                   add_index_node=False,
#                   verbose=True, **kwargs
#                   ):
#     """

#     :param lead_months:
#     :param window:
#     :param use_heat_content:
#     :param lon_min:
#     :param lon_max:
#     :param lat_min:
#     :param lat_max:
#     :param data_dir:
#     :param cmip5_data:
#     :param cmip5_label:
#     :param noaa_data:
#     :param noaa_label:
#     :param era5_data:
#     :param era5_label:
#     :param truncate_ERA5:
#     :param return_new_coordinates:
#     :param return_mask:
#     :param target_months: if "all", the model will need to learn to give predictions for any target months,
#                             if an int in [1, 12], it can focus on that specific target month/season,
#                             where 1 translates to "JFM", ..., 12 to "DJF"
#     :return:
#     """
#     cmip5, cmip5_Y, m1 = reformat_atm_data(lead_months=lead_months, window=window,
#                                            lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                            data_dir=data_dir + "CMIP5/", sp_key="sp",
#                                            sample_file=cmip5_data, label_file=cmip5_label,
#                                            get_valid_nodes_mask=True, get_valid_coordinates=False)
#     NOAA, NOAA_Y, m2 = reformat_atm_data(lead_months=lead_months, window=window,
#                                          lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                          data_dir=data_dir + "NOAA/", sample_file=noaa_data, label_file=noaa_label,
#                                          get_valid_nodes_mask=True, get_valid_coordinates=False)
#     ERA5, ERA5_Y, m3 = reformat_atm_data(lead_months=lead_months, window=window,
#                                            lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
#                                            data_dir=data_dir + "ERA5/", sample_file=era5_data, label_file=era5_label,
#                                            get_valid_nodes_mask=True, get_valid_coordinates=False)
#     if truncate_ERA5:
#         start_1984 = 24  # 2 * 12
#         ERA5, ERA5_Y, ERA5.attrs["time"] = ERA5[start_1984:], ERA5_Y[start_1984:], ERA5.attrs["time"][start_1984:]
#     # DUE to variations due to resolution = 5deg., there are some inconsistencies in which nodes are terrestrial
#     final_mask = np.logical_and(m1, np.logical_and(m2, m3))
#     cmip5, NOAA, ERA5 = cmip5[:, :, :, final_mask], NOAA[:, :, :, final_mask], ERA5[:, :, :, final_mask]
#     if add_index_node:
#         cmip5, NOAA, ERA5 = add_ONI_node(cmip5), add_ONI_node(NOAA), add_ONI_node(ERA5)
#         # cords = np.array(list(cords) + [(0, 205)])  # add coordinate for ONI
#         final_mask = np.append(final_mask, True)  # add
#         if verbose:
#             print('MASKING OUT', np.count_nonzero(np.logical_not(final_mask)), 'nodes')
#     cords = ERA5.indexes['cord']
#     if return_new_coordinates:
#         # cords = cords[final_mask]
#         if return_mask:
#             return (cmip5, cmip5_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y), cords, final_mask
#         return (cmip5, cmip5_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y), cords
#     if return_mask:
#         return (cmip5, cmip5_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y), final_mask
#     return (cmip5, cmip5_Y), (NOAA, NOAA_Y), (ERA5, ERA5_Y)




def add_ONI_node(data_array):
    """

    :param data_array: A xarray DataArray of shape (#time-steps, #features, window, #nodes)
    :return: A xarray DataArray of shape (#time-steps, #features, window, #nodes + 1)
    """
    _, mask = get_index_mask(data_array[:, 0, 0, :], 'ONI', flattened_too=True, is_data_flattened=True)
    mask = np.array(mask)
    oni_cord_index = pd.MultiIndex.from_tuples([(0, 205)], names=['lat', 'lon'])
    ONI_NODE = data_array[:, :, :, mask].mean(dim='cord', keepdims=True).assign_coords({'cord': oni_cord_index})
    data_array = xa.concat((data_array, ONI_NODE), dim='cord')
    return data_array

