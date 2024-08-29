import glob
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import *

def amean(da, conversion_factor=1/365):
    # annual mean
    months  = da['time.daysinmonth']
    xa = conversion_factor*(months*da).groupby('time.year').sum().compute()
    xa.name = da.name
    return xa

def preprocess(ds):
    data_vars = ['TLAI', 'GPP', 'TBOT', 'RAIN', 'SNOW', 'area', 'landfrac']
    return ds[data_vars]
  
def byhand(t, p):
    t1 = -5  #    tundra-boreal
    t2 =  3  #    boreal-temperate
    t3 = 20  # temperate-tropical

    tvals = [-np.inf, t1, t2, t3, np.inf]
    bvals = [9, 8, 4, 1]
    
    for i in range(4):
        if (t > tvals[i]) & (t <= tvals[i + 1]):
            b = bvals[i]

    td = 15
    pd=60
    bd=3  #desert
    if ( t> td) & (p < pd):
        b = bd

    return b
  
def get_bclass(n, ta, pr, gpp, data):

    ncell = len(ta)
    bclass = np.zeros(ncell) + np.nan

    for c in range(ncell):
        if (ta[c] < 0) & (gpp[c] == 0):
            bclass[c] = 0  #ice
        else:
            ptf = gpd.GeoDataFrame({'geometry': [Point(ta[c], pr[c])]})
            x = gpd.overlay(ptf, data, how='intersection')
            if len(x) > 0:
                bclass[c] = x.biome_id.values[0]
            else:
                bclass[c] = byhand(ta[c],pr[c])
    return bclass
  
def get_biome_map(clm_sim_dir, whit_shp, whitkey):

    # Load full grid CLM simulation at 2degree
    files = sorted(glob.glob(clm_sim_dir + '*h0*'))[-84:]
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time',
                           parallel=True, preprocess=preprocess,
                           decode_times=False)
    ds['time'] = xr.cftime_range('2007', periods=84, freq='MS',
                                 calendar='noleap')

    # calculate temperature and precip for each gridcell
    conversion_factor = 24*60*60*365/10
    tbot = amean(ds.TBOT).mean(dim='year') - 273.15  # degC
    rain = conversion_factor*amean(ds.RAIN).mean(dim='year')  # cm/yr
    snow = conversion_factor*amean(ds.SNOW).mean(dim='year')  # cm/yr
    prec = rain + snow

    landfrac = ds.isel(time=0).landfrac
    area = ds.isel(time=0).area

    ta = tbot.values.reshape(-1, 1)
    pr = prec.values.reshape(-1, 1)

    gpp_clim = amean(ds.GPP).mean(dim='year').compute()
    gpp = gpp_clim.values.reshape(-1, 1)

    data = gpd.read_file(whit_shp)
    data.biome_id = np.array([9, 8, 7, 6, 5, 4, 1, 2, 3])

    tmp = get_bclass(1, ta, pr, gpp, data)

    Biome_ID = tmp.reshape(np.shape(tbot)[0], np.shape(tbot)[1])

    ds_biomeID = xr.DataArray(Biome_ID, dims=('lat', 'lon'), coords={'lat': ds.lat, 'lon': ds.lon})

    ds_out = ds_biomeID.to_dataset(name='biome')
    ds_out['biome_name'] = whitkey.biome_name

    ds_out['landfrac'] = landfrac
    ds_out['area'] = area

    ds_out['biome'] = xr.where(ds_out.landfrac > 0.0, ds_out.biome, -9999)
    ds_masked = ds_out.where(ds_out['biome'] != -9999)

    return ds_masked