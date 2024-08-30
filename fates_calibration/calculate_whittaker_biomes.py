"""Functions to calculate Whittaker biomes globally
"""
import glob
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def calculate_annual_mean(
    data_array: xr.DataArray, conversion_factor: float = 1 / 365
) -> xr.DataArray:
    """Calculates annual mean of an input DataArray, applies a conversion factor if supplied

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float, optional): Conversion factor. Defaults to 1/365.

    Returns:
        xr.DataArray: output DataArray
    """

    months = data_array["time.daysinmonth"]
    annual_mean = conversion_factor * (months * data_array).groupby("time.year").sum()
    annual_mean.name = data_array.name
    return annual_mean


def preprocess(data_set: xr.Dataset) -> xr.Dataset:
    """Preprocesses and xarray Dataset by subsetting to specific variables - to be used on read-in

    Args:
        ds (xr.Dataset): input Dataset

    Returns:
        xr.Dataset: output Dataset
    """

    data_vars = ["TLAI", "GPP", "TBOT", "RAIN", "SNOW", "area", "landfrac"]
    return data_set[data_vars]


def calc_biome_by_hand(temperature: float, precip: float) -> int:
    """Calculates Whittaker biome by hand via input temperature and precipitation

    Args:
        temperature (float): annual temperature [degC]
        precip (float): annual precipitation [mm/c]

    Returns:
        int: Whittaker Biome class
    """

    temp_1 = -5.0  # tundra-boreal
    temp_2 = 3.0  # boreal-temperate
    temp_3 = 20.0  # temperate-tropical

    temperature_vals = [-np.inf, temp_1, temp_2, temp_3, np.inf]
    biome_vals = [9, 8, 4, 1]

    if (temperature > 15.0) & (precip < 60.0):
        # desert
        return 3

    for i in range(len(temperature_vals) - 1):
        if (temperature > temperature_vals[i]) & (
            temperature <= temperature_vals[i + 1]
        ):
            biome_int = biome_vals[i]

    return biome_int


def get_biome_class(
    temperature: np.array, precip: np.array, gpp: np.array, data
) -> np.array:
    """Calculates global biome class from input arrays of temperature, precipitation,
    and GPP, as well as a Whittaker biomes shapefile


    Args:
        temperature (np.array): global temperature [degC]
        precip (np.array): global precipitation [cm/yr]
        gpp (np.array): global GPP [gC/m2/yr]
        data (_type_): whittaker biomes shapefile

    Returns:
        np.array: biome class
    """

    ncell = len(temperature)
    biome_class = np.zeros(ncell) + np.nan

    for cell in range(ncell):
        if (temperature[cell] < 0.0) & (gpp[cell] == 0.0):
            # ice
            biome_class[cell] = 0
        else:
            ptf = gpd.GeoDataFrame(
                {"geometry": [Point(temperature[cell], precip[cell])]}
            )
            point = gpd.overlay(ptf, data, how="intersection")
            if len(point) > 0:
                biome_class[cell] = point.biome_id.values[0]
            else:
                biome_class[cell] = calc_biome_by_hand(temperature[cell], precip[cell])

    return biome_class


def get_global_vars(clm_sim: xr.Dataset):
    """Reads in and processes data needed for Whittaker biome calculation

    Args:
        clm_dir (xr.Dataset): CLM simulation dataset

    Returns:
        xr.DataArray: temperature, precipitation, and gpp
    """

    # calculate temperature, precipitation, and gpp for each gridcell
    tbot = calculate_annual_mean(clm_sim.TBOT).mean(dim="year") - 273.15  # degC

    conversion_factor = 24 * 60 * 60 * 365 / 10
    rain = conversion_factor * calculate_annual_mean(clm_sim.RAIN).mean(
        dim="year"
    )  # cm/yr
    snow = conversion_factor * calculate_annual_mean(clm_sim.SNOW).mean(
        dim="year"
    )  # cm/yr
    prec = rain + snow

    gpp = calculate_annual_mean(clm_sim.GPP).mean(dim="year")

    # reshape arrays
    tbot_reshape = tbot.values.reshape(-1, 1)
    prec_reshape = prec.values.reshape(-1, 1)
    gpp_reshape = gpp.values.reshape(-1, 1)

    return tbot_reshape, prec_reshape, gpp_reshape


def read_in_clm_sim(clm_dir: str):
    """Reads in CLM simulation needed to calculate Whittaker biomes

    Args:
        clm_dir (str): path to CLM simulation

    Returns:
        xr.Dataset: CLM simulation dataset
    """

    # load full grid CLM simulation at 2degree
    files = sorted(glob.glob(clm_dir + "*h0*"))[-84:]
    clm_sim = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        preprocess=preprocess,
        decode_times=False,
    )
    clm_sim["time"] = xr.cftime_range("2007", periods=84, freq="MS", calendar="noleap")

    return clm_sim


def whittaker_biomes(
    clm_dir: str, whit_shp_file: str, whitkey: xr.DataArray
) -> xr.Dataset:
    """Calculates global Whittaker biomes

    Args:
        clm_dir (str): path to CLM simulation (for input data)
        whit_shp_file (str): path to Whittaker biomes shapefile
        whitkey (xr.DataArray): data array with biome integer-key pairs

    Returns:
        xr.Dataset: Biome Dataset
    """

    # get data needed to calculate Whittaker biomes
    clm_sim = read_in_clm_sim(clm_dir)
    tbot, prec, gpp = get_global_vars(clm_sim)

    # read in the whittaker biomes shapefile
    whittaker_shapefile = gpd.read_file(whit_shp_file)
    whittaker_shapefile.biome_id = np.array([9, 8, 7, 6, 5, 4, 1, 2, 3])

    # calculate biome class
    biome_class = get_biome_class(tbot, prec, gpp, whittaker_shapefile)

    # rehape and turn into a DataSet
    biome_id = biome_class.reshape(np.shape(tbot)[0], np.shape(tbot)[1])

    da_biome_id = xr.DataArray(
        biome_id, dims=("lat", "lon"), coords={"lat": clm_sim.lat, "lon": clm_sim.lon}
    )
    ds_out = da_biome_id.to_dataset(name="biome")
    ds_out["biome_name"] = whitkey.biome_name
    ds_out["landfrac"] = clm_sim.isel(time=0).landfrac
    ds_out["area"] = clm_sim.isel(time=0).area
    ds_out["biome"] = xr.where(ds_out.landfrac > 0.0, ds_out.biome, -9999)
    ds_masked = ds_out.where(ds_out["biome"] != -9999)

    return ds_masked
