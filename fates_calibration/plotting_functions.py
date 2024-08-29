import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from ILAMB_observations_functions import average_obs_by_model
from FATES_calibration_constants import ILAMB_MODELS

def choose_subplot_dimensions(k):
    if k < 2:
        return k, 1
    elif k < 11:
        return math.ceil(k/2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(k/3), 3

def generate_subplots(k, row_wise=False):
    nrow, ncol = choose_subplot_dimensions(k)
    figure, axes = plt.subplots(nrow, ncol, figsize=(13, 6),
                                subplot_kw=dict(projection=ccrs.Robinson()),
                                layout='compressed')
    if not isinstance(axes, np.ndarray):
        return figure, [axes]
    else:
        axes = axes.flatten(order=('C' if row_wise else 'F'))
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)
            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)
        axes = axes[:k]
        return figure, axes
      
def map_function(ax, dat, title, cmap, vmax, vmin, div=False):

    if div:
        vmin = min(vmin, -1.0*vmax)
        vmax = max(vmax, np.abs(vmin))
    
    ax.set_title(title, loc='left', fontsize='large', fontweight='bold')
    ax.coastlines()
    ocean = ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                                                        facecolor='white'))
    pcm = ax.pcolormesh(dat.lon, dat.lat, dat,
                        transform=ccrs.PlateCarree(), shading='auto',
                        cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm

def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n*multiplier)/multiplier
  
def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n*multiplier)/multiplier

def get_by_lat(global_dat, var, models, units, cf=None):

    if cf is not None:
        globd = (global_dat[var])*global_dat.land_area*cf
        globd = globd.where(globd.model.isin(models), drop = True)
        by_lat = globd.sum(dim='lon')
    else:
        globd = global_dat[var]
        globd = globd.where(globd.model.isin(models), drop = True)
        by_lat = globd.mean(dim='lon')

    by_lat.attrs = {'units': units, 'long_name': global_dat[var].attrs['long_name']}
    
    return globd, by_lat
  
def plot_obs_bylat(data_bylat, var, varname, units):

    df = pd.DataFrame({'lat': np.tile(data_bylat.lat, len(data_bylat.model)),
                       'model': np.repeat(data_bylat.model, len(data_bylat.lat)),
                       var: data_bylat.values.flatten()})
    minval = df[var].min()
    if minval < 0:
        minvar = round_up(np.abs(minval))*-1.0
    else:
        minvar = truncate(minval)
    
    maxvar = round_up(df[var].max())

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r/255., g/255., b/255.)

    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xlim(minvar, maxvar)
    plt.ylim(-90, 90)

    plt.yticks(range(-90, 91, 15), [str(x) + "º" for x in range(-90, 91, 15)],
               fontsize=10)
    plt.xticks(fontsize=10)

    for y in range(-90, 91, 15):
        plt.plot(range(math.floor(minvar), math.ceil(maxvar) + 1),
                 [y] * len(range(math.floor(minvar), math.ceil(maxvar) + 1)),
                 "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(bottom=False, top=False, left=False, right=False)

    models = np.unique(df.model.values)
    for rank, model in enumerate(models):
        data = df[df.model == model]
        plt.plot(data[var].values, data.lat.values, lw=2, color=tableau20[rank],
                 label=model)

    plt.ylabel('Latitude (º)', fontsize=11)
    plt.xlabel(f'Annual {varname} ({units})', fontsize=11)
    plt.title(f"Observed Annual {varname}" + 
              " by latitude for different data products", fontsize=11)
    plt.legend(loc='upper right')

def plot_global(var_glob, varname, units, cmap, row_wise=False, div=False):

    vmin = var_glob.min().values
    vmax = var_glob.max().values
    models = var_glob.model.values
    n = len(models)
    figure, axes = generate_subplots(n, row_wise=False)
    if n > 1:
        axes = axes.flatten(order=('C' if row_wise else 'F'))
        for idx, ax in enumerate(axes):
            dat = var_glob.sel(model=models[idx])
            pcm = map_function(ax, dat, models[idx], cmap, vmax, vmin, div=div)
        cbar = figure.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation='horizontal')
    else:
        dat = var_glob.sel(model=models[0])
        pcm = map_function(axes[0], dat, models[0], cmap, vmax, vmin, div=div)
        cbar = figure.colorbar(pcm, ax=axes[0], shrink=0.5, orientation='horizontal')
    cbar.set_label(f'{varname} ({units})', size=10, fontweight='bold')

def get_biome_palette():
    
    # set the hue palette as a dict for custom mapping
    biome_names = ['Ice sheet', 'Tropical rain forest',
                'Tropical seasonal forest/savanna', 'Subtropical desert',
                'Temperate rain forest', 'Temperate seasonal forest',
                'Woodland/shrubland', 'Temperate grassland/desert',
                'Boreal forest', 'Tundra']
    colors = ["#ADADC9", "#317A22", "#A09700", "#DCBB50", "#75A95E", "#97B669",
            "#D16E3F", "#FCD57A", "#A5C790", "#C1E1DD"]
    
    palette = {}
    for i in range(len(colors)):
        palette[float(i)] = colors[i]
        
    return palette, biome_names

def plot_pft_grid(colors, names, obs_data, obs_df):
    
    cmap = matplotlib.colors.ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(13, 6),
                       subplot_kw=dict(projection=ccrs.Robinson()))
    ax.coastlines()
    ocean = ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m',
                                                    facecolor='white'))
    
    pcm = ax.pcolormesh(obs_data.lon, obs_data.lat, obs_data.biome,
                    transform=ccrs.PlateCarree(), shading='auto',
                    cmap=cmap, vmin=-0.5,
                    vmax=9.5)
    scatter = ax.scatter(obs_df.lon, obs_df.lat, s=10, c='none',
                         edgecolor='black', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03,
                    orientation='vertical')
    cbar.set_label('Biome', size=12, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cbar.set_ticklabels(names)
    ax.set_extent([-50, 180, -10, 10])
    
def plot_rel_sd_var(obs_ds, obs_var, name, points, pft='all'):

    obs = average_obs_by_model(obs_ds, ILAMB_MODELS[obs_var.upper()], obs_var)
    
    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    
    ax.coastlines()
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m', facecolor='white'))
    
    pcm = ax.pcolormesh(obs.lon, obs.lat, obs[f"{obs_var}_rel_sd"], transform=ccrs.PlateCarree(),
                        shading='auto', cmap='rainbow', vmin=0.0, vmax=6.0)
    if pft != 'all':
        points = points[points.pft == pft]
    
    ax.scatter(points.lon, points.lat, s=15, c='none', edgecolor='black', transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation='horizontal')
    cbar.set_label(f'Observed {name} Relative Standard Deviation', size=10,
                   fontweight='bold')
    
def plot_mean_var(obs_ds, obs_var, name, units, points, cmap, pft='all'):

    obs = average_obs_by_model(obs_ds, ILAMB_MODELS[obs_var.upper()], obs_var)
    vmin = obs[obs_var].min().values
    vmax = obs[obs_var].max().values
    
    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    
    ax.coastlines()
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '110m', facecolor='white'))
    
    pcm = ax.pcolormesh(obs.lon, obs.lat, obs[obs_var], transform=ccrs.PlateCarree(),
                        shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    if pft != 'all':
        points = points[points.pft == pft]
    
    ax.scatter(points.lon, points.lat, s=15, c='none', edgecolor='black', transform=ccrs.PlateCarree())
    
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation='horizontal')
    cbar.set_label(f'Observed {name} ({units})', size=10, fontweight='bold')

def plot_obs_hists(obs_df, pft, vars, names, units):
    
    palette, biome_names = get_biome_palette()
    
    pft_df = obs_df[obs_df.pft == pft]
    fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
    axes = axes.flatten(order=('C'))
    for i, ax in enumerate(axes):
        sns.histplot(data=pft_df, x=vars[i], hue='biome', stat='count',
                     edgecolor=None, palette=palette,
                     multiple="stack", ax=ax)
        ax.set_ylabel('Number of Gridcells', fontsize=11)
        ax.set_xlabel(f'Observed {names[i]} ({units[i]})', fontsize=11)
        ax.get_legend().remove()
    
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title='Biome',
               labels=np.flip([biome_names[int(b)] for b in np.unique(pft_df.biome)]))
    fig.suptitle(f'Observations for {pft} grids')
    fig.tight_layout()