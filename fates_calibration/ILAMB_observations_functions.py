import os
from datetime import date
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe

def month_wts(nyears: int) -> xr.DataArray:
    """Helper function for summing up a monthly variable by days

    Args:
        nyears (int): number of years in your Dataset

    Returns:
        xr.DataArray: A DataArray with number of days per month tiled by number of years
    """
    days_pm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return xr.DataArray(np.tile(days_pm, nyears), dims='time')

def adjust_lon(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Adjusts the longitude values of a dataset to be from 0-360 to -180 to 180

    Args:
        ds (xr.Dataset): Dataset
        lon_name (str): name of the longitude variable

    Returns:
        xr.Dataset: Dataset with the longitude values changes
    """

    # adjust lon values to make sure they are within (-180, 180)
    ds['_longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds
        .swap_dims({lon_name: '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
        .drop_vars(lon_name))

    ds = ds.rename({'_longitude_adjusted': lon_name})

    return ds
  
def get_annual_obs(ds: xr.Dataset, var: str, conversion_factor: float) -> xr.Dataset:
    """Sums the annual values for a variable, using a conversion factor

    Args:
        ds (xr.Dataset): Dataset
        var (str): variable to sum
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: output annual sum
    """
  
    # calculate annual values
    nyears = len(np.unique(ds['time.year']))
    annual = conversion_factor*(ds[var]*month_wts(nyears)).groupby('time.year').sum()
  
    return annual

def cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Given arrays of latitude and longitude, return cell areas in square meters

    Args:
        lat (np.ndarray): a 1D array of latitudes (cell centroids)
        lon (np.ndarray): a 1D array of longitudes (cell centroids)

    Returns:
        np.ndarray: a 2D array of cell areas [m2]
    """
    
    EARTH_RAD = 6.371e6

    x = np.zeros(lon.size + 1)
    x[1:-1] = 0.5*(lon[1:] + lon[:-1])
    x[0] = lon[0] - 0.5*(lon[1] - lon[0])
    x[-1] = lon[-1] + 0.5*(lon[-1] - lon[-2])
    
    if x.max() > 181:
        x -= 180
    x = x.clip(-180, 180)
    x *= np.pi/180.0

    y = np.zeros(lat.size + 1)
    y[1:-1] = 0.5*(lat[1:] + lat[:-1])
    y[0] = lat[0] - 0.5*(lat[1] - lat[0])
    y[-1] = lat[-1] + 0.5*(lat[-1] - lat[-2])
    
    y = y.clip(-90, 90)
    y *= np.pi/180.0

    dx = EARTH_RAD*(x[1:] - x[:-1])
    dy = EARTH_RAD*(np.sin(y[1:]) - np.sin(y[:-1]))
    areas = np.outer(dx, dy).T

    return areas

def filter_dataset(ds, var, tstart=None, tstop=None, max_val=None, min_val=None):
    
    # filter by min/max
    if max_val is not None:
        ds = ds.where(ds[var] <= max_val)
    if min_val is not None:
        ds = ds.where(ds[var] >= min_val)
        
    # subset by time
    if tstart is not None and tstop is not None:
      ds = ds.sel(time=slice(tstart, tstop))
      
    return ds

def get_biomass_ds(top_dir, sub_dir, model, filename, in_var, max_val=None, min_val=None):
    
    # read in dataset
    file_path = os.path.join(top_dir, sub_dir, model, filename)
    raw_ds = xr.open_dataset(file_path)
    
    # filter by min/max
    if max_val is not None:
        raw_ds = raw_ds.where(raw_ds[in_var] <= max_val)
    if min_val is not None:
        raw_ds = raw_ds.where(raw_ds[in_var] >= min_val)
    
    # only one time date
    raw_ds = raw_ds.isel(time=0).drop_vars(['time'])
        
    # calculate cell areas
    lons = raw_ds.lon.values
    lats = np.array(sorted(raw_ds.lat.values))
    areas = cell_areas(lats, lons)
    land_area = xr.DataArray(areas, coords={"lat": lats, "lon": lons})
    
    biomass = raw_ds[in_var]

    # add in land area
    biomass['land_area'] = land_area
    biomass['land_area'].attrs['units'] = 'm2'
    biomass['land_area'].attrs['long_name'] = 'land area'

    # update attributes
    biomass.attrs['Original'] = filename
    biomass.attrs['Date'] = str(date.today())
    biomass.attrs['Author'] = 'afoster@ucar.edu'
    
    return biomass

def get_annual_albedo(top_dir, in_var, out_var, units, longname, conversion_factor,
                  rsds_dict, rsus_dict, tstart=None, tstop=None):
    
    # read in datasets
    rsds_path = os.path.join(top_dir, rsds_dict['sub_dir'], rsds_dict['model'],
                           rsds_dict['filename'])
    rsus_path = os.path.join(top_dir, rsus_dict['sub_dir'], rsus_dict['model'],
                           rsus_dict['filename'])
    
    rsds = xr.open_dataset(rsds_path)
    rsus = xr.open_dataset(rsus_path)
    
    # filter by date and min/max
    rsds = filter_dataset(rsds, rsds_dict['in_var'], tstart, tstop, rsds_dict['max_val'],
                        rsds_dict['min_val'])
    rsus = filter_dataset(rsus, rsus_dict['in_var'], tstart, tstop, rsus_dict['max_val'],
                        rsus_dict['min_val'])
    
    # calculate albedo
    alb = albedo(rsds[rsds_dict['in_var']], rsus[rsus_dict['in_var']], 10).to_dataset(name=in_var)
    
    # calculate cell areas
    lons = alb.lon.values
    lats = np.array(sorted(alb.lat.values))
    areas = cell_areas(lats, lons)
    land_area = xr.DataArray(areas, coords={"lat": lats, "lon": lons})
    
    annual = get_annual_obs(alb, in_var, conversion_factor)
    
    # get average and sd across years
    annual_mean = annual.mean(dim='year').to_dataset(name=f'{out_var}')
    annual_mean[f'{out_var}'].attrs['units'] = units
    annual_mean[f'{out_var}'].attrs['long_name'] = f'average {longname}'
    
    annual_sd = annual.var(dim='year').to_dataset(name=f'{out_var}_iav')
    annual_sd[f'{out_var}_iav'].attrs['units'] = units
    annual_sd[f'{out_var}_iav'].attrs['long_name'] = f'interannual variation of {longname}'
    
    annual_ds = xr.merge([annual_mean, annual_sd])

    # add in land area
    annual_ds['land_area'] = land_area
    annual_ds['land_area'].attrs['units'] = 'm2'
    annual_ds['land_area'].attrs['long_name'] = 'land area'

    # update attributes
    annual_ds.attrs['Original'] = f"{rsds_path} and {rsus_path}"
    annual_ds.attrs['Date'] = str(date.today())
    annual_ds.attrs['Author'] = 'afoster@ucar.edu'
    
    return annual_ds
    
def get_annual_ef(top_dir, in_var, out_var, units, longname, conversion_factor,
                  le_dict, sh_dict, tstart=None, tstop=None):
    
    # read in datasets
    le_path = os.path.join(top_dir, le_dict['sub_dir'], le_dict['model'],
                           le_dict['filename'])
    sh_path = os.path.join(top_dir, sh_dict['sub_dir'], sh_dict['model'],
                           sh_dict['filename'])
    
    le = xr.open_dataset(le_path)
    sh = xr.open_dataset(sh_path)
    
    # filter by date and min/max
    le = filter_dataset(le, le_dict['in_var'], tstart, tstop, le_dict['max_val'],
                        le_dict['min_val'])
    sh = filter_dataset(sh, sh_dict['in_var'], tstart, tstop, sh_dict['max_val'],
                        sh_dict['min_val'])
    
    # calculate evaporative fraction
    ef = evapfrac(sh[sh_dict['in_var']], le[le_dict['in_var']], 20).to_dataset(name=in_var)
    
    # calculate cell areas
    lons = ef.lon.values
    lats = np.array(sorted(ef.lat.values))
    areas = cell_areas(lats, lons)
    land_area = xr.DataArray(areas, coords={"lat": lats, "lon": lons})
    
    annual = get_annual_obs(ef, in_var, conversion_factor)
    
    # get average and sd across years
    annual_mean = annual.mean(dim='year').to_dataset(name=f'{out_var}')
    annual_mean[f'{out_var}'].attrs['units'] = units
    annual_mean[f'{out_var}'].attrs['long_name'] = f'average {longname}'
    
    annual_sd = annual.var(dim='year').to_dataset(name=f'{out_var}_iav')
    annual_sd[f'{out_var}_iav'].attrs['units'] = units
    annual_sd[f'{out_var}_iav'].attrs['long_name'] = f'interannual variation of {longname}'
    
    annual_ds = xr.merge([annual_mean, annual_sd])

    # add in land area
    annual_ds['land_area'] = land_area
    annual_ds['land_area'].attrs['units'] = 'm2'
    annual_ds['land_area'].attrs['long_name'] = 'land area'

    # update attributes

    annual_ds.attrs['Original'] = f"{le_path} and {sh_path}"
    annual_ds.attrs['Date'] = str(date.today())
    annual_ds.attrs['Author'] = 'afoster@ucar.edu'
    
    return annual_ds
    
def get_annual_ds(top_dir, sub_dir, model, filename, in_var, out_var, conversion_factor,
                  units, longname, tstart=None, tstop=None, max_val=None,
                  min_val=None):

    # read in dataset
    file_path = os.path.join(top_dir, sub_dir, model, filename)
    raw_ds = xr.open_dataset(file_path)
    
    # filter dataset
    if model == 'MODIS':
        raw_ds = filter_dataset(raw_ds, in_var, None, None, max_val, min_val)
    else:
        raw_ds = filter_dataset(raw_ds, in_var, tstart, tstop, max_val, min_val)
    
    # calculate cell areas
    lons = raw_ds.lon.values
    lats = np.array(sorted(raw_ds.lat.values))
    areas = cell_areas(lats, lons)
    land_area = xr.DataArray(areas, coords={"lat": lats, "lon": lons})
    
    # calculate annual values
    annual = get_annual_obs(raw_ds, in_var, conversion_factor)
    
    # get average and sd across years
    annual_mean = annual.mean(dim='year').to_dataset(name=f'{out_var}')
    annual_mean[f'{out_var}'].attrs['units'] = units
    annual_mean[f'{out_var}'].attrs['long_name'] = f'average {longname}'
    
    annual_sd = annual.var(dim='year').to_dataset(name=f'{out_var}_iav')
    annual_sd[f'{out_var}_iav'].attrs['units'] = units
    annual_sd[f'{out_var}_iav'].attrs['long_name'] = f'interannual variation of {longname}'
    
    annual_ds = xr.merge([annual_mean, annual_sd])

    # add in land area
    annual_ds['land_area'] = land_area
    annual_ds['land_area'].attrs['units'] = 'm2'
    annual_ds['land_area'].attrs['long_name'] = 'land area'

    # update attributes

    annual_ds.attrs['Original'] = file_path
    annual_ds.attrs['Date'] = str(date.today())
    annual_ds.attrs['Author'] = 'afoster@ucar.edu'
    
    return annual_ds

def get_annual_obs_times_area(ds, var, conversion_factor, land_area):
    
    nyears = len(np.unique(ds['time.year']))
    ds[f'{var}_m2'] = ds[var]*land_area
    annual = conversion_factor*(ds[f'{var}_m2']*month_wts(nyears)).groupby('time.year').sum()
    
    return annual
   
def evapfrac(sh: xr.DataArray, le: xr.DataArray, energy_threshold: float) -> xr.DataArray:
    """Calculates evaporative fraction as le/(le + sh)

    Args:
        sh (xr.DataArray): sensible heat flux
        le (xr.DataArray): latent heat flux
        energy_threshold (float): energy threshold to prevent div/0s

    Returns:
        xr.DataArray: evaporative fraction [0-1]
    """
    sh = sh.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    le = le.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    ef = le/(le + sh)
    return ef
    
def albedo(rsds: xr.DataArray, rsus: xr.DataArray, energy_threshold: float) -> xr.DataArray:
    """Calculates albedo as rsus/rsds

    Args:
        rsds (xr.DataArray): downward shortwave radiation
        rsus (xr.DataArray): upward shortwave radiation
        energy_threshold (float): energy threshold to prevent div/0s

    Returns:
        xr.DataArray: albedo [0-1]
    """
    rsds = rsds.where(rsds > energy_threshold)
    rsus = rsus.where(rsus > energy_threshold)
    alb = rsus/rsds
    return alb

def regrid_ILAMB_ds(ds, target_grid, var, method="bilinear", interpolate=True):
    
    # interpolate to get rid of NAs from ocean/land mismatch
    if interpolate:
        ds_sorted = ds.sortby('lat')
        ds_interp = ds_sorted.interpolate_na(dim='lon', method='nearest', fill_value='extrapolate')
        ds_interp = ds_interp.interpolate_na(dim='late', method='nearest', fill_value='extrapolate')
    else:
        ds_interp = ds
        
    regridder = xe.Regridder(ds_interp, target_grid, method)
    ds_regrid = regridder(ds_interp[var]).to_dataset(name=var)      
    ds_regrid = ds_regrid*target_grid.mask
    ds_regrid[var].attrs = ds[var].attrs

    return ds_regrid

def get_dataset(top_dir, models, var):
    
    data_sets = []
    for model in models:
        file_name = os.path.join(top_dir, f"{model}_{var}_regridded.nc")
        data_sets.append(xr.open_dataset(file_name))
    obs = xr.concat(data_sets, 'model', data_vars='all')
    obs = obs.assign_coords(model=("model", models))
 
    return obs

def compile_observational_datasets(top_dir, land_area_file):
    
    files = sorted(os.listdir(top_dir))
    all_models = [f.split('_')[0] for f in files]
    vars = [f.split('_')[-2].replace('_regridded.nc', '') for f in files]
    unique_vars = np.unique(vars)
    
    data_sets = []
    for var in unique_vars:
        models = [all_models[i] for i in range(len(vars)) if vars[i] == var]
        data_sets.append(get_dataset(top_dir, models, var))

    # add in land area
    ds_grid = xr.open_dataset(land_area_file)
    land_area = ds_grid.landfrac*ds_grid.area
    land_area = land_area.to_dataset(name='land_area')
    land_area['land_area'].attrs['units'] = 'km2'
    data_sets.append(land_area)
   
    # merge together
    obs_ds = xr.merge(data_sets)
    
    return obs_ds

def average_obs_by_model(obs_ds, models, var_name):
    
    obs = obs_ds[var_name].where(obs_ds.model.isin(models), drop=True)
    
    obs_mean = obs.mean(dim='model').to_dataset(name=f'{var_name}')
    obs_var = obs.var(dim='model').to_dataset(name=f'{var_name}_var')
    
    obs_sd = obs.std(dim='model')
    obs_rel_sd = (obs_sd/obs_sd.mean()).to_dataset(name=f'{var_name}_rel_sd')

    obs_data = xr.merge([obs_mean, obs_var, obs_rel_sd])

    return obs_data

def extract_obs(obs_ds, var, var_models, lats, lons, pfts):
    
    # average observations by model
    obs = average_obs_by_model(obs_ds, var_models, var)
    
    # grab observational lat/lons
    obs_lats = obs['lat']
    obs_lons = obs['lon']
    
    # extract observations at the chosen gridcells
    var_mean = np.zeros(len(lats))
    var_var = np.zeros(len(lats))
    var_rel_sd = np.zeros(len(lats))
    for i in range(len(lats)):
            nearest_index_lat = np.abs(obs_lats - lats[i]).argmin()
            nearest_index_lon = np.abs(obs_lons - lons[i]).argmin()
            
            # grab data at correct lat/lon
            var_mean[i] = obs[f'{var}'][nearest_index_lat, nearest_index_lon]
            if len(var_models) > 1:
                var_var[i] = obs[f'{var}_var'][nearest_index_lat, nearest_index_lon]
                var_rel_sd[i] = obs[f'{var}_rel_sd'][nearest_index_lat, nearest_index_lon]
    
    obs_df = pd.DataFrame({'lat': lats, 'lon': lons, 'pft': pfts,
                           f'{var}': var_mean, f'{var}_var': var_var,
                           f'{var}_rel_sd': var_rel_sd})
    
    return obs_df

def filter_df(df, filter_vars, tol):
    
    for var in filter_vars:
        df = df.where(df[f"{var}_rel_sd"] < tol)
    
    df = df.dropna()
    
    return df
