import os
import xarray as xr
from fates_calibration.ILAMB_observations_functions import regrid_ILAMB_ds
from fates_calibration.ILAMB_obs_dict import ILAMB_DATA_DICT

def create_target_grid(file):
    ds_sp = xr.open_dataset(file)
    target_grid = ds_sp.TLAI.mean(dim='time')
    target_grid['mask'] = ds_sp.landmask
    
    return target_grid

def main(in_dir, out_dir, target_grid_file, clobber=False):
    
    target_grid = create_target_grid(target_grid_file)
    files = [f for f in os.listdir(in_dir) if f.endswith('.nc')]
    
    for file in files:
        file_basename = f'{file.replace(".nc", "")}'
        file_out = os.path.join(out_dir, f'{file_basename}_regridded.nc')
        
        if os.path.isfile(file_out):
          if clobber:
              os.remove(file_out)
          else:
              print(f"File {file_out}, skipping, set clobber=True if rewrite desired")
              continue
      
        ds = xr.open_dataset(os.path.join(in_dir, file))
        
        var = ILAMB_DATA_DICT[file_basename]['out_var']
        
        if var == 'biomass':
            ds_regrid = regrid_ILAMB_ds(ds, target_grid, var)
        else:
            ds_regrid_mean = regrid_ILAMB_ds(ds, target_grid, f'{var}')
            ds_regrid_var = regrid_ILAMB_ds(ds, target_grid, f'{var}_iav')
            ds_regrid = xr.merge([ds_regrid_mean, ds_regrid_var])
        
        ds_regrid.to_netcdf(file_out)
    
if __name__ == "__main__":
    
    in_dir = '/glade/work/afoster/FATES_calibration/ILAMB_data'
    out_dir = '/glade/work/afoster/FATES_calibration/ILAMB_data/regridded'
    
    target_grid_file = '/glade/work/linnia/LAI_SP_ctsm51d115/run/LAI_SP_ctsm51d115.clm2.h0.2000-02-01-00000.nc'
    
    # create the directory if it doesn't exist
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    main(in_dir, out_dir, target_grid_file, clobber=True)