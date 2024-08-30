import os
import xarray as xr
from ILAMB_observations_functions import compile_observational_datasets


def main(top_dir, out_dir, file_name, grid_file):
    ds = compile_observational_datasets(top_dir, grid_file)
    ds.to_netcdf(os.path.join(out_dir, file_name))


if __name__ == "__main__":

    grid_file = "/glade/work/linnia/LAI_SP_ctsm51d115/run/LAI_SP_ctsm51d115.clm2.h0.2000-02-01-00000.nc"
    top_dir = "/glade/work/afoster/FATES_calibration/ILAMB_data/regridded"
    out_dir = "/glade/u/home/afoster/FATES_Calibration/FATES_SP/observations"
    file_name = "ILAMB_obs.nc"
    main(top_dir, out_dir, file_name, grid_file)
