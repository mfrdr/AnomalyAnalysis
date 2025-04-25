"""
Author: Salva Rühling Cachay
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
import networkx as nx
import cartopy.crs as ccrs
from utilities.utils import mask_adj_out
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def init_world_fig(cm=180):
    fig = plt.figure(figsize=(19.20, 10.80))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=cm))
    minlon = -180 + cm
    maxlon = +179 + cm
    ax.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
    ax.coastlines()
    return fig, ax


def plot_centrality_from_adj(adj: np.ndarray, lats, lons, coordinates=None, format=None, save_to=None, plot_heatmap=True,
                             set_title=True,
                             min_weight=0.1, show=True, verbose=True, horizon=-1):
    graph = get_graph_from_adj(adj, coordinates, min_weight=min_weight)
    lat_len, lon_len = len(lats), len(lons)
    if verbose:
        print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}")
        print("Computing eigenvector...")
    try:
        centrality = nx.eigenvector_centrality(graph, max_iter=150)
    except nx.PowerIterationFailedConvergence:
        try:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality = nx.eigenvector_centrality(graph, max_iter=300)
        except nx.PowerIterationFailedConvergence:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality = nx.eigenvector_centrality(graph, max_iter=800)

    centrality_heat = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])
    for node, (lat_i, lon_i) in enumerate(coordinates):
        centrality_heat.loc[lat_i, lon_i] = centrality[node]

    cm = 180  # bring figure center to enso region
    fig, ax = init_world_fig(cm=cm)
    minlon = -180 + cm
    maxlon = +179 + cm
    ax.set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())

    if set_title:
        ax.set_title(f"Heatmap of eigenvector centrality for {horizon} lead months (thresh={min_weight})")

    if plot_heatmap:
        im = ax.pcolormesh(lons, lats, centrality_heat, cmap="Reds", transform=ccrs.PlateCarree())
        fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    else:
        im = ax.contourf(lons, lats, centrality_heat, transform=ccrs.PlateCarree(), alpha=0.85, cmap="Reds", levels=100)
        fig.colorbar(im, ax=ax, pad=0.01)

    ax.coastlines()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', format=format, dpi=500)
    if show:
        plt.show()
    else:
        print("Omitting plotting")

def plot_centrality_from_2_adj(adj_atm: np.ndarray, adj_oc: np.ndarray, lats, lons, coordinates=None, format=None, save_to=None,                                  plot_heatmap=True,set_title=True,min_weight=0.1, show=True, verbose=True, horizon=-1):
    
    graph_atm = get_graph_from_adj(adj_atm, coordinates, min_weight=min_weight)
    graph_oc = get_graph_from_adj(adj_oc, coordinates, min_weight=min_weight)
    lat_len, lon_len = len(lats), len(lons)
    if verbose:
        print(f"#Nodes_Atm: {graph_atm.number_of_nodes()}, #Edges_Atm: {graph_atm.number_of_edges()}")
        print(f"#Nodes_Oc: {graph_oc.number_of_nodes()}, #Edges_Oc: {graph_oc.number_of_edges()}")
        print("Computing eigenvector...")
    try:
        centrality_atm = nx.eigenvector_centrality(graph_atm, max_iter=150)
    except nx.PowerIterationFailedConvergence:
        try:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality_atm = nx.eigenvector_centrality(graph_atm, max_iter=300)
        except nx.PowerIterationFailedConvergence:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality_atm = nx.eigenvector_centrality(graph_atm)
    print("Centrality computation for Atm concluded")
    try:
        centrality_oc = nx.eigenvector_centrality(graph_oc, max_iter=150)
    except nx.PowerIterationFailedConvergence:
        try:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality_oc = nx.eigenvector_centrality(graph_oc, max_iter=300)
        except nx.PowerIterationFailedConvergence:
            print('EV centrality computation failed to converge, trying with more iterations..')
            centrality_oc = nx.eigenvector_centrality(graph_oc)
    print("Centrality computation for Oc concluded")
    
    centrality_heat_atm = xa.DataArray(np.zeros((lat_len, lon_len)), coords=[("lat", lats), ("lon", lons)])
    for node, (lat_i, lon_i) in enumerate(coordinates):
        centrality_heat_atm.loc[lat_i, lon_i] = centrality_atm[node]
        centrality_heat_oc.loc[lat_i, lon_i] = centrality_oc[node]

    cm = 180  # center the figure on ENSO region
    minlon, maxlon = -180 + cm, 179 + cm

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Supertitle
    if set_title:
        fig.suptitle(f"Heatmap of eigenvector centrality for {horizon} lead months (thresh={min_weight})", fontsize=14, fontweight="bold")

    # Plot Atmospheric centrality
    axs[0].set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
    axs[0].set_title("Atmospheric Nodes", fontsize=12)
    im1 = axs[0].pcolormesh(lons, lats, centrality_heat_atm, cmap="Reds", transform=ccrs.PlateCarree())
    fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.05)
    axs[0].coastlines()

    # Plot Oceanic centrality
    axs[1].set_extent([minlon, maxlon, -55, 60], ccrs.PlateCarree())
    axs[1].set_title("Oceanic Nodes", fontsize=12)
    im2 = axs[1].pcolormesh(lons, lats, centrality_heat_oc, cmap="Blues", transform=ccrs.PlateCarree())
    fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.05)
    axs[1].coastlines()

    # Save or show
    if save_to:
        plt.savefig(save_to, bbox_inches="tight", format=format, dpi=500)
    if show:
        plt.show()
    else:
        print("Plot omitted")

def get_graph_from_adj(adj, coordinates, min_weight=0.1, directed=True):
    graph = nx.DiGraph() if directed else nx.Graph()
    if coordinates is not None:
        for node, (lat_i, lon_i) in enumerate(coordinates):
            graph.add_nodes_from([(node, {"latitude": lat_i, "longitude:": lon_i})])  # attributes aren't used
    else:
        graph.add_nodes_from([(node, {}) for node in range(adj.shape[0])])  # attributes aren't used

    rows, cols = np.where(adj > min_weight)
    edges = zip(rows.tolist(), cols.tolist())
    graph.add_edges_from(edges)
    return graph


def plot_edges_from_adj(adj, coordinates, emph_short_edges=True, format=None, save_to=None, set_title=True,
                        min_weight=0.1, show=True, k_hops_is_short=1, arrows=False, horizon=-1, resolution=5):
    graph = get_graph_from_adj(adj, coordinates, min_weight=min_weight)
    print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}")

    cm = 180  # bring figure center to enso region
    fig, ax = init_world_fig(cm=cm)

    pos = {}
    for node, (lat_i, lon_i) in enumerate(coordinates):
        pos[node] = (lon_i - cm, lat_i)
    nx.draw_networkx_nodes(graph, pos, node_color="white", node_size=0.01)
    nx.draw_networkx_edges(graph, pos=pos, edge_color='darkgreen', alpha=0.05, arrows=False)

    if emph_short_edges:
        """ Draw 'short'-distance edges with extra emphasis """
        short_adj = mask_adj_out(adj, coordinates=coordinates, max_distance=resolution * k_hops_is_short)
        graph = get_graph_from_adj(short_adj, coordinates, min_weight=min_weight)
        nx.draw_networkx_edges(graph, pos=pos, edge_color='navy', alpha=0.7, arrows=arrows, width=0.8)

        # Now the short ones == 1 hop
        short_adj = mask_adj_out(adj, coordinates=coordinates, max_distance=resolution * 1)
        graph = get_graph_from_adj(short_adj, coordinates, min_weight=min_weight)
        nx.draw_networkx_edges(graph, pos=pos, edge_color='orange', alpha=0.94, arrows=arrows, arrowsize=5,
                               width=1.2 if arrows else 0.8)  # , width=1, arrowsize=1)

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', format=format, dpi=500)
    if show:
        plt.show()
    else:
        print("Omitting plotting")


def plot_time_series(data, *args, labels=["timeseries"], time_steps=None, data_std=None, linewidth=2,
                     timeaxis="time", ylabel="Nino3.4 index", plot_months=False, show=True, save_to=None):
    if time_steps is not None:
        time = time_steps
    elif isinstance(data, xa.DataArray):
        time = data.get_index(timeaxis)
    else:
        time = np.arange(0, data.shape[0], 1)
    series = np.array(data)
    plt.figure()
    plt.plot(time, series, label=labels[0], linewidth=linewidth)
    if data_std is not None:
        plt.fill_between(time, series - data_std, series + data_std, alpha=0.25)
    minimum, maximum = np.min(data), np.max(data)
    for i, arg in enumerate(args, 1):
        minimum, maximum = min(minimum, np.min(arg)), max(maximum, np.max(np.max(arg)))
        try:
            plt.plot(time, arg, label=labels[i], linewidth=linewidth)
        except ValueError as e:
            raise ValueError("Please align the timeseries to the same time axis.", e)
        except IndexError:
            raise IndexError("You must pass as many entries in labels, as there are time series to plot")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.yticks(np.arange(np.round(minimum - 0.5, 0), np.round(maximum + 0.51, 0), 0.5))
    nth_month = 10
    if plot_months and isinstance(time[0], pd._libs.tslibs.Timestamp):
        xticks, year_mon = time[::nth_month][:-1], [f"{date.year}-{date.month}" for date in time[::nth_month][:-1]]
        xticks = xticks.append(pd.Index([time[-1]]))
        year_mon.append(f"{time[-1].year}-{time[-1].month}")  # add last month
        plt.xticks(ticks=xticks, labels=year_mon, rotation=20)
    plt.legend()
    plt.grid()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')

    if show:
        plt.show()
    return time
