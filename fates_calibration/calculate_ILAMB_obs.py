import os
import xarray as xr
import pandas as pd

from ILAMB_observations_functions import get_annual_ef, get_biomass_ds, get_annual_ds
from ILAMB_observations_functions import get_annual_albedo
from ILAMB_obs_dict import ILAMB_DATA_DICT

def main(out_dir, tstart, tstop, clobber=False):
  
    ILAMB_DIR = '/glade/campaign/cesm/community/lmwg/diag/ILAMB/DATA'
  
    for dataset, attributes in ILAMB_DATA_DICT.items():
        
        model = attributes['model']
        out_var = attributes['out_var']
        file_name = f'{model}_{out_var.upper()}.nc'
        out_file = os.path.join(out_dir, file_name)
        
        if os.path.isfile(out_file):
            if clobber:
                os.remove(out_file)
            else:
                print(f"File {out_file} for {dataset} exists, skipping, set clobber=True if rewrite desired")
                continue
        
        # get annual values
        if attributes['out_var'] == 'ef':
            
            le_dict = ILAMB_DATA_DICT[f'{model}_LE']
            sh_dict = ILAMB_DATA_DICT[f'{model}_SH']
            ds = get_annual_ef(ILAMB_DIR, attributes['in_var'], out_var,
                               attributes['units'], attributes['longname'], 
                               attributes['conversion_factor'], le_dict, sh_dict,
                               tstart, tstop)
        
        elif attributes['out_var'] == 'albedo':
            rsds_dict = ILAMB_DATA_DICT[f'{model}_RSDS']
            rsus_dict = ILAMB_DATA_DICT[f'{model}_FSR'] 
            ds = get_annual_albedo(ILAMB_DIR, attributes['in_var'], out_var,
                                   attributes['units'], attributes['longname'],
                                   attributes['conversion_factor'], rsds_dict, rsus_dict,
                                   tstart, tstop)
        
        elif attributes['out_var'] == 'biomass':
            
            ds = get_biomass_ds(ILAMB_DIR, attributes['sub_dir'], model,
                                attributes['filename'], attributes['in_var'],
                                max_val=attributes['max_val'],
                                min_val=attributes['min_val'])
        else:

            ds = get_annual_ds(ILAMB_DIR, attributes['sub_dir'], model,
                        attributes['filename'], attributes['in_var'], attributes['out_var'], 
                        attributes['conversion_factor'], attributes['units'], 
                        attributes['longname'], tstart=tstart, tstop=tstop, 
                        max_val=attributes['max_val'], min_val=attributes['min_val'])
        
        # write to file
        ds.to_netcdf(out_file)

if __name__ == "__main__":
    
    out_dir = '/glade/work/afoster/FATES_calibration/ILAMB_data'
    tstart = '2005-01-01'
    tstop = '2014-12-31'
    main(out_dir, tstart, tstop)