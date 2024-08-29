import os
import numpy as np
import xarray as xr
import cftime
import sys
import pandas as pd
import glob
import dask
import argparse
from dask_jobqueue import PBSCluster
from dask.distributed import Client

def get_data_dict(var_df: pd.DataFrame, var_out: str) -> dict:
    """Returns a dictionary with TRENDY-relevant output information for specified TRENDY
    output variable name. Based on input data frame

    Args:
        var_df (pd.DataFrame): pandas data frame with information about each output variable
        var_out (str): output variable

    Returns:
        dict: data dictionary
    """

    # subset to this output variable
    df_sub = var_df[var_df['output_varname'] == var_out]

    # grab relevant info
    data_dict = {'vars_in': df_sub['CLM_varname'].values[0].split('-'),
                 'long_name': df_sub['long_name'].values[0],
                 'output_units': df_sub['output_units'].values[0],
                 'fill_val': df_sub['NA_val'].values[0],
                 'conversion_factor': df_sub['conversion_factor'].values[0],
                 'frequency': df_sub['frequency'].values[0],
                 'dims': df_sub['CMOR_dims'].values[0],
                 'dimension': df_sub['dimension'].values[0],
                 'output_function': df_sub['output_function'].values[0]}

    # h0 or h1 streams
    if 'PFT' in data_dict['dims']:
        data_dict['stream'] = 'h1'
    else:
        data_dict['stream'] = 'h0'

    return data_dict

def get_var_files(tseries_dir: str, pft_dir: str, case_tag: str, exp: str, stream: str, var: str) -> list[str]:
    """Gets a list of files to read in

    Args:
        tseries_dir (str): single-variable time series directory for all CLM variables
        pft_dir (str): directory with regridded PFT-level single-variable time series files
        case_tag (str): case name
        exp (str): TRENDY experiment
        stream (str): file stream (h0 or h1)
        var (str): CLM variable

    Returns:
        list[str]: list of files
    """

    if stream == 'h0':
        var_files = sorted(glob.glob(f'{tseries_dir}/*{stream}.{var}.*'))
    elif stream == 'h1':
        var_files = sorted(glob.glob(f"{pft_dir}/{case_tag}.clm2_{exp}_{var}_pft*.nc"))
    else:
      print(f"Unknown stream {stream}.")

    return var_files

def read_in_ds(files: list[str], stream: str, min_year: int=1700, max_year: int=2023) -> xr.Dataset:
    """Reads in an xarray dataset (usually a single-variable dataset) for TRENDY output
       Assumes pft-level (h1) files are single-variable and single-pft, so all pft files
       for the variable are read in

    Args:
        files (list[str]): list files to read in
        stream (str): file stream (h0 or h1)
        min_year (int): minimum year
        max_year (int): maximum year

    Returns:
        xr.Dataset: xarray dataset from input files
    """
  
    # read in dataset from input set of files
    if stream == 'h0':
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time',
                                   parallel=True, autoclose=True, chunks={'time': 10})
    elif stream == 'h1':
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='PFT',
                                   parallel=True, autoclose=True, chunks={'time': 10})
        ds = ds.transpose('time', 'PFT', 'lat', 'lon')
    else:
      print(f"Unknown stream {stream}.")

    # reset time 
    yr0 = ds['time.year'][0].values
    ds['time'] = xr.cftime_range(str(yr0), periods=len(ds.time), freq='MS',
                                 calendar='noleap')
    
    # subset to correct years
    ds = ds.sel(time=slice(f'{min_year}-01-01', f'{max_year}-12-31'))
    
    return ds
  
def read_in_all_vars(vars_in: list[str], tseries_dir: str, pft_dir: str,
                     case_tag: str, exp: str, data_dict: dict) -> xr.Dataset:
    """Read in all CLM variables required for this TRENDY output variable and merges them
    into one xarray dataset

    Args:
        vars_in (list[str]): list of CLM history variables
        tseries_dir (str): single-variable time series directory for all CLM variables
        pft_dir (str): directory with regridded PFT-level single-variable time series files
        case_tag (str): case name
        exp (str): TRENDY experiment
        data_dict (dict): dictionary with TRENDY-relevant information about this output variable

    Returns:
        xr.Dataset: xarray Dataset with all the relevant CLM history variables
    """
    
    # read in all the data we need for this output variable
    ds_list = []
    for var in vars_in:
        
        if var in ['FLDS', 'FSDS']:
            # these don't exist on the h1 streams but we need them for some PFT-level calculculations
            stream = 'h0'
        else: 
            stream = data_dict['stream']
        
        # get files and read in each individual variable
        files = get_var_files(tseries_dir, pft_dir, case_tag, exp,
                              stream, var)
        ds_var = read_in_ds(files, stream)
        
        # add in pftname for h1 streams
        if stream == 'h0':
            ds_var = ds_var[[var]]
        else:
            ds_var = ds_var[[var, 'pftname']]
        
        ds_list.append(ds_var)
    
    ds = xr.merge(ds_list)

    return ds

def sum_vars(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Sums all variables in vars_in 

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of variables to sum
        var_out (str): name of output variable

    Returns:
        xr.Dataset: dataset with new variable
        str: a string to add to the attributes describing how this 
        new variable was made
    """

    ds[var_out] = xr.full_like(ds[vars_in[0]], 0.0)
    for var in vars_in:
        ds[var_out] += ds[var]

    clm_orig_var_string = ' + '.join(vars_in)

    return ds, clm_orig_var_string

def get_cpool(ds: xr.Dataset, var_out: str, active_var: str, slow_var: str,
              passive_var: str) -> xr.Dataset:
    """Reconfigures carbon pool history variable output to be dimensioned by carbon pool

    Args:
        ds (xr.Dataset): input dataset
        var_out (str): output variable string
        active_var (str): name of active history variable
        slow_var (str): name of slow history variable
        passive_var (str): name of passive history variable

    Returns:
        xr.Dataset: output dataset reconfigured correctly
    """
    
    active = ds[active_var].to_dataset(name=var_out)
    slow = ds[slow_var].to_dataset(name=var_out)
    passive = ds[passive_var].to_dataset(name=var_out)

    ds_pools = xr.concat([active, slow, passive], dim='Pool', data_vars='all')
    ds_pools = ds_pools.assign_coords(Pool=("Pool", ['active', 'slow', 'passive']))
    ds_pools = ds_pools.transpose('time', 'Pool', 'lat', 'lon')

    return ds_pools

def find_cpool_var(vars_in, cpool_string):
    return vars_in[np.argwhere([cpool_string in var for var in vars_in])[0][0]]

def carbon_pool(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates the TRENDY cpool output variable using several CLM history variables

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of CLM history variables
        var_out (str): output variable name

    Returns:
        xr.Dataset: dataset with new variable
        str: a string to add to the attributes describing how this 
        new variable was made
    """
  
    # find matching strings
    active_var = find_cpool_var(vars_in, "ACT")
    slow_var = find_cpool_var(vars_in, "SLO")
    passive_var = find_cpool_var(vars_in, "PAS")
    
    # reorganize by carbon pool
    ds = get_cpool(ds, var_out, active_var, slow_var, passive_var)

    clm_orig_var_string = ', '.join(vars_in)

    return ds, clm_orig_var_string
  
def rh_pool(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates the rhpool TRENDY variable

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of CLM variables used 
        var_out (str): output variable name

    Returns:
        xr.Dataset: output dataset with new variable
        str: a string to add to the attributes describing how this 
          new variable was made
    """
    
    # sum appropriate history variables
    ds['FROM_ACT'] = ds['SOM_ACT_C_TO_SOM_PAS_C'] + ds['SOM_ACT_C_TO_SOM_SLO_C']
    ds['FROM_SLO'] = ds['SOM_SLO_C_TO_SOM_ACT_C'] + ds['SOM_SLO_C_TO_SOM_PAS_C']
    ds['FROM_PAS'] = ds['SOM_PAS_C_TO_SOM_ACT_C']

    # reorganize by carbon pool
    ds_pools, _ = get_cpool(ds, var_out, 'FROM_ACT', 'FROM_SLO', 'FROM_PAS')

    clm_orig_var_string = 'active: SOM_ACT_C_TO_SOM_PAS_C, SOM_ACT_C_TO_SOM_SLO_C; slow: SOM_SLO_C_TO_SOM_ACT_C, SOM_SLO_C_TO_SOM_PAS_C; passive: SOM_PAS_C_TO_SOM_ACT_C'

    return ds_pools, clm_orig_var_string
  
def black_sky_albedo(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates black sky albedo

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of CLM variables used 
        var_out (str): output variable name

    Returns:
        xr.Dataset: output dataset with new variable
        str: a string to add to the attributes describing how this 
          new variable was made
    """

    bs_alb = (ds['FSRND'] + ds['FSRVD'])/(ds['FSDSND'] + ds['FSDSVD'])
    data_out = bs_alb.to_dataset(name=var_out)

    clm_orig_var_string = '(FSRND + FSRVD)/(FSDSND + FSDSVD)'

    return data_out, clm_orig_var_string
  
def white_sky_albedo(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates white sky albedo

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of CLM variables used 
        var_out (str): output variable name

    Returns:
        xr.Dataset: output dataset with new variable
        str: a string to add to the attributes describing how this 
          new variable was made
    """

    ws_alb = (ds['FSRNI'] + ds['FSRVI'])/(ds['FSDSNI'] + ds['FSDSVI'])
    data_out = ws_alb.to_dataset(name=var_out)

    clm_orig_var_string = '(FSRNI + FSRVI)/(FSDSNI + FSDSVI)'
    
    return data_out, clm_orig_var_string

def net_radiation(ds: xr.Dataset, vars_in: list[str], var_out: str):
    """Calculates net radiation

    Args:
        ds (xr.Dataset): input dataset
        vars_in (list[str]): list of CLM variables used 
        var_out (str): output variable name

    Returns:
        xr.Dataset: output dataset with new variable
        str: a string to add to the attributes describing how this 
          new variable was made
    """
    
    rn = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
    data_out = rn.to_dataset(name=var_out)
    data_out['pftname'] = ds.pftname

    clm_orig_var_string = 'FLDS - FIRE + FSDS - FSR'
    
    return data_out, clm_orig_var_string
  
def calculate_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Calculates annual mean of a variable

    Args:
        da (xr.DataArray): input data array

    Returns:
        xr.DataArray: annual mean data array
    """
    
    months = da['time.daysinmonth']
    annual_mean = 1/365*(months*da).groupby('time.year').sum()
    annual_mean.name = da.name
    annual_mean.attrs = da.attrs
  
    return annual_mean
  
def make_annual(ds: xr.Dataset, var_out: str, fill_val: float) -> xr.Dataset:
    """Calculates annual output for an input dataset and variable

    Args:
        ds (xr.Dataset): input dataset
        var_out (str): variable to calculate annual mean for
        fill_val (float): fill value

    Returns:
        xr.Dataset: _description_
    """

    annual_mean = calculate_annual_mean(ds[var_out])
    data_out = annual_mean.to_dataset(name=var_out)
    data_out = data_out.rename({'year': 'time'})

    data_out['time'].attrs = {
        'long_name': 'year',
        'units': 'yr',
        '_FillValue': fill_val,
    }

    return data_out

def post_process_var(tseries_dir: str, var_out: str, data_dict: dict, func_dict: dict,
                     pft_dir: str, case_tag: str, exp: str) -> xr.Dataset:
    """Calculate TRENDY output variable

    Args:
        tseries_dir (str): single-variable time series directory for all CLM variables
        var_out (str): output variable name
        data_dict (dict): dictionary with TRENDY-relevant information about this output variable
        func_dict (dict): dictionary with output functions 
        pft_dir (str): directory with regridded PFT-level single-variable time series files
        case_tag (str): case name
        exp (str): TRENDY experiment

    Returns:
        xr.Dataset: output dataset
    """

    vars_in = data_dict['vars_in']

    # read in all the data we need for this output variable
    ds = read_in_all_vars(vars_in, tseries_dir, pft_dir, case_tag, exp, data_dict)
    ds_raw = ds

    # create new variable based on function
    ds, clm_orig_var_string = func_dict[data_dict['output_function']](ds, vars_in, var_out)

    # # convert to correct units
    ds[var_out] = ds[var_out]*data_dict['conversion_factor']

    # convert to annual if desired
    if data_dict['frequency'] == 'annual':
        data_out = make_annual(ds, var_out, data_dict['fill_val'])
    elif data_dict['frequency'] == 'monthly':
        data_out = ds[var_out].to_dataset(name=var_out)
    else:
        print(f'unknown frequency for variable {var_out}')
        return

    # rename dimensions for soil layers
    if data_dict['dimension'] == 'soil_layer':
        if var_out == 'tsl':
            data_out = data_out.rename({'levgrnd': 'stlayer'})
        elif var_out == 'msl':
            data_out = data_out.rename({'levgrnd': 'smlayer'})
        else:
            print(f'unknown output dimension for variable {var_out}')

    # set attributes
    data_out[var_out].attrs = {
        'long_name': data_dict['long_name'],
        'units': data_dict['output_units'],
        '_FillValue': data_dict['fill_val'],
        'CLM-TRENDY_unit_conversion_factor': data_dict['conversion_factor']
    }

    # add information about CLM original variables
    data_out[var_out].attrs['CLM_orig_var_name'] = clm_orig_var_string
    for var in vars_in:
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_units"] = ds_raw[var].attrs['units']
        data_out[var_out].attrs[f"CLM_orig_attr_{var}_long_name"] = ds_raw[var].attrs['long_name']

    # transpose PFT
    if data_dict['stream'] == 'h1':
        data_out['pftname'] = ds.pftname
        data_out['pftname'].attrs = {'long_name': 'pft name'}
        data_out['PFT'].attrs = {'long_name': 'pft index'}
        data_out = data_out.transpose('time', 'PFT', 'lat', 'lon')

    # fill na values
    data_out = data_out.fillna(data_dict['fill_val'])

    return data_out

def create_trendy_var(var_df: pd.DataFrame, func_dict: dict, var_out: str,
                      tseries_dir: str, pft_dir: str, case_tag: str, exp: str):
    """Creates the TRENDY output dataset for the specified variable and supples an 
    encoding for writing

    Args:
        var_df (pd.DataFrame): pandas data frame with information about each output variable
        func_dict (dict): dictionary with output functions 
        var_out (str): output variable name
        tseries_dir (str): single-variable time series directory for all CLM variables
        pft_dir (str): directory with regridded PFT-level single-variable time series files
        case_tag (str): case name
        exp (str): TRENDY experiment

    Returns:
        xr.Dataset: output dataset
        dict:       encoding dictionary for writing
    """
    
    data_dict = get_data_dict(var_df, var_out)
    data_out = post_process_var(tseries_dir, var_out, data_dict, func_dict, pft_dir, 
                                case_tag, exp)
    encoding = {
        var_out: {
            'dtype': 'float32',
        },
        'lat': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'lon': {
            'dtype': 'float32',
            '_FillValue': -9999.0
        },
        'time': {
            'dtype': 'int32'
        }
    }
    if data_dict['stream'] == 'h1':
        encoding['PFT'] = {'dtype': 'int32',
                           '_FillValue': -9999}
        encoding['pftname'] = {'dtype': '<U16'}

    return data_out, encoding
  
def create_dask_client(n_scale):
    print("Creating cluster... ", end="")
    cluster = PBSCluster(
        cores=1,
        memory='30GB',
        processes=1,
        queue='casper',
        resource_spec='select=1:ncpus=1:mem=2GB',
        account='P93300041',
        walltime='04:00:00',
        log_directory="/glade/derecho/scratch/afoster/dask_logs/"
    )
    cluster.scale(n_scale)
    dask.config.set({
        'distributed.dashboard.link': 'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    })
    client = Client(cluster)
    
    return client

def commandline_args():
    """Parse and return command-line arguments"""

    description = """

    Typical usage:
      python make_TRENDY_outputs --exp S0

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="S0",
        help="Experiment to post-process\n",
    )
    parser.add_argument(
        "--ntasks",
        type=int,
        default=20,
        help="Number of dask jobs to scale\n",
    )
    parser.add_argument(
        "--clobber",
        action='store_true',
        help="Overwrite files\n",
    )

    args = parser.parse_args()

    return args
  
def main():
    
    args = commandline_args()
    
    client = create_dask_client(args.ntasks)
    print(client)
    
    func_dict = {
      'sum': sum_vars,
      'carbon_pool': carbon_pool,
      'rh_pool': rh_pool,
      'black_sky_albedo': black_sky_albedo,
      'white_sky_albedo': white_sky_albedo,
      'rn': net_radiation
    }
    
    # output directories and case names
    case_tag = "TRENDY2024_f09_clm60"
    out_dir = '/glade/derecho/scratch/afoster/TRENDY_outputs'
    
    # location of time series directory and history directory
    top_dir = '/glade/derecho/scratch/afoster/TRENDY_outputs'
    tseries_dir = f'/glade/derecho/scratch/afoster/TRENDY_outputs/{args.exp}/month_1/'
    pft_dir = os.path.join(top_dir, f"{args.exp}/pft")
    
    # information about postprocessing
    var_df = pd.read_csv('/glade/work/afoster/TRENDY_2024/post_processing/TRENDY_CLM_vars.csv')
    all_vars = [var for var in var_df['output_varname'].values if var not in ['oceanCoverFrac', 'landCoverFrac']]
    
    for var_out in all_vars:
        
        file_name = os.path.join(out_dir, 'to_trendy', f"CLM6.0_{args.exp}_{var_out}.nc")

        if os.path.isfile(file_name) and not args.clobber:
            print(f"File {file_name} exists, skipping...")
            continue
        else:
            if os.path.isfile(file_name):
                os.remove(file_name)
            data_out, encoding = create_trendy_var(var_df, func_dict, var_out,
                                                    tseries_dir, pft_dir, case_tag,
                                                    args.exp)
            data_out.to_netcdf(file_name, encoding=encoding)
            
if __name__ == "__main__":
    main()
    