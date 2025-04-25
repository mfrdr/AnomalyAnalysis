"""
Author: Salva Rühling Cachay
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval_gcn_2 import evaluate_preds

from utilities.data_wrangling_2 import load_data, to_dataloaders
from utilities.utils import get_euclidean_adj


def train_epoch(dataloader, model, criterion, optims, device, epoch, nth_step=100):
    if not isinstance(optims, list):
        optims = [optims]
    model.train()
    total_loss = 0

#     # Progress bar for the epoch
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True)

    for iter, (X_atm, X_oc, Y) in enumerate(dataloader, 1):
        X_atm, X_oc, Y = X_atm.to(device), X_oc.to(device), Y.to(device)
        for optim in optims:
            optim.zero_grad()
        # Reshape both inputs to match model's expected format
        X_atm = X_atm.reshape((X_atm.shape[0], -1, X_atm.shape[3])).transpose(1, 2)
        X_oc = X_oc.reshape((X_oc.shape[0], -1, X_oc.shape[3])).transpose(1, 2)
        
        preds = model(X_atm, X_oc)  # Ensure model takes both inputs
        loss = criterion(preds, Y)
        loss.backward()
        for optim in optims:
            optim.step()
        total_loss += loss.item()

        # # Update progress bar with loss information
        # progress_bar.set_postfix(loss=loss.item())

    num_edges = torch.count_nonzero(model.adj_oc.detach()).item()
    return total_loss / iter, num_edges



def evaluate(dataloader, model, device, return_preds=False):
    model.eval()
    total_loss_l2 = 0
    total_loss_l1 = 0
    preds = None
    Ytrue = None
    for i, (X_atm, X_oc, Y) in enumerate(dataloader, 1):
        assert len(X_atm.size()) == 4, "Expected X_atm to have shape (batch_size, #channels, window, #nodes)"
        assert len(X_oc.size()) == 4, "Expected X_oc to have shape (batch_size, #channels, window, #nodes)"
        X_atm, X_oc, Y = X_atm.to(device), X_oc.to(device), Y.to(device)
        X_atm = X_atm.reshape((X_atm.shape[0], -1, X_atm.shape[3])).transpose(1, 2)
        X_oc = X_oc.reshape((X_oc.shape[0], -1, X_oc.shape[3])).transpose(1, 2)

        with torch.no_grad():
            output = model(X_atm, X_oc)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss_l2 += F.mse_loss(output, Y).item()
        total_loss_l1 += F.l1_loss(output, Y).item()

    preds = preds.data.cpu().numpy()
    Ytest = Ytrue.data.cpu().numpy()
    oni_stats = evaluate_preds(Ytest, preds, return_dict=True)
    oni_stats['mae'] = total_loss_l1
    if return_preds:
        return total_loss_l2 / i, oni_stats, Ytest, preds
    else:
        return total_loss_l2 / i, oni_stats


def get_static_feats(params, net_params, coordinates, trainset):
    max_lat = max(params['lat_max'], params['lat_min'])
    static_feats = np.array([
        [lat / max_lat, (lon - 180) / 360] for lat, lon in coordinates
    ])  # (#nodes, 2) = (#nodes (lat, lon))
    trainset_sst = trainset[:, 0, 0, :].squeeze()  # take SSTs of the first timestep before prediction
    static_feats = np.concatenate((static_feats, trainset_sst.T), axis=1)  # (#nodes, 2 + len(trainset))
    if trainset.shape[1] == 2:
        trainset_hc = trainset[:, 1, 0, :].squeeze()  # take SSTs of the first timestep before prediction
        static_feats = np.concatenate((static_feats, trainset_hc.T), axis=1)  # (#nodes, 2 + 2*len(trainset))
    return static_feats


def get_dataloaders(params, net_params):
    # Load data
    load_data_kwargs = {
        'window': params['window'], 'lead_months': params['horizon'], 'lon_min': params['lon_min'],
        'lon_max': params['lon_max'], 'lat_min': params['lat_min'], 'lat_max': params['lat_max'],
        'data_dir': params['data_dir'], 'use_heat_content': params['use_heat_content'],
        'add_index_node': net_params['index_node']
    }
    cmip5_oc, SODA, GODAS, cmip5_atm, NOAA, ERA5, cords, cnn_mask = load_data(**load_data_kwargs, return_new_coordinates=True, return_mask=True)
    

    net_params['num_nodes'] = SODA[0].shape[3]
    if 'grid_edges' in params and params['grid_edges']:
        print('Using grid edges, i.e. based on spatial proximity!!!! ')
        adj_oc = get_euclidean_adj(GODAS[0], radius_lat=5, radius_lon=5, self_loop=True)
        adj_atm = get_euclidean_adj(ERA5[0], radius_lat=5, radius_lon=5, self_loop=True)
        static_feats_oc = None
        static_feats_atm = None
    else:
        adj_oc = None
        adj_atm = None
        # Static features for adj learning
        static_feats_oc = get_static_feats(params, net_params, cords, SODA[0])
        static_feats_atm = get_static_feats(params, net_params, cords, NOAA[0])
        # print("Statics shapes:", static_feats_oc.shape, static_feats_atm.shape)
        assert SODA[0].shape[3] == cmip5_oc[0].shape[3] and SODA[0].shape[3] == GODAS[0].shape[3]
        assert NOAA[0].shape[3] == cmip5_atm[0].shape[3] and NOAA[0].shape[3] == ERA5[0].shape[3]

    trainloader, valloader, testloader = \
        to_dataloaders(cmip5_oc, SODA, GODAS, cmip5_atm, ERA5, NOAA, 
                       batch_size=params['batch_size'],
                       valid_split=params['validation_frac'], concat_cmip5_and_soda=True,
                       shuffle_training=params['shuffle'], validation=params['validation_set'],
                       separate_inputs=True)
    del cmip5_oc, SODA, GODAS, cmip5_atm, NOAA, ERA5
    return (adj_atm, adj_oc, static_feats_atm, static_feats_oc, cords), (trainloader, valloader, testloader)


def get_dirs(params, net_params):
    suffix = params['ID'] + f"{net_params['num_nodes']}nodes_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    params['ID'] = suffix
    log_dir = f"{params['model_dir']}/logs/" + suffix
    ckpt_dir = f"{params['model_dir']}/checkpoints/" + suffix
    for d in [log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    return ckpt_dir, log_dir
