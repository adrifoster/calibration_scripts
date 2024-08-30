"""Calculates observations needed for FATES calibration from the raw ILAMB data repository.

Update ILAMB_DATA_DICT in ILAMB_obs_dict to update any relevant information about
each ILAMB dataset, or add more ILAMB datasets

Typical usage example:

  python calculate_ILAMB_obs.py

"""
import argparse
import os
import xarray  # pylint: disable=unused-import

from fates_calibration.ILAMB_observations_functions import (
    get_annual_ef,
    get_biomass_ds,
    get_annual_ds,
)
from fates_calibration.ILAMB_observations_functions import get_annual_albedo
from fates_calibration.ILAMB_obs_dict import ILAMB_DATA_DICT


def commandline_args():
    """Parses and return command-line arguments"""

    description = """
    Typical usage: python calculate_ILAMB_obs.py 

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2005-01-01",
        help="Starting date of ILAMB data\n",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2014-01-01",
        help="Ending date of ILAMB data\n",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/glade/work/afoster/FATES_calibration/ILAMB_data",
        help="Directory to write ILAMB data to\n",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Whether or not to overwrite files\n",
    )
    parser.add_argument(
        "--ILAMB_dir",
        type=str,
        default="/glade/campaign/cesm/community/lmwg/diag/ILAMB/DATA",
        help="Location of ILAMB data repository\n",
    )

    args = parser.parse_args()

    return args


def main():
    """Runs the main program"""

    args = commandline_args()

    for dataset, attributes in ILAMB_DATA_DICT.items():

        model = attributes["model"]
        out_var = attributes["out_var"]
        file_name = f"{model}_{out_var.upper()}.nc"
        out_file = os.path.join(args.out_dir, file_name)

        if os.path.isfile(out_file):
            if args.clobber:
                os.remove(out_file)
            else:
                print(f"File {out_file} for {dataset} exists, skipping")
                print("Use --clobber if rewrite desired.")
                continue

        # get annual values
        if attributes["out_var"] == "ef":

            le_dict = ILAMB_DATA_DICT[f"{model}_LE"]
            sh_dict = ILAMB_DATA_DICT[f"{model}_SH"]
            ilamb_data = get_annual_ef(
                args.ILAMB_dir,
                attributes["in_var"],
                out_var,
                attributes["units"],
                attributes["longname"],
                attributes["conversion_factor"],
                le_dict,
                sh_dict,
                args.start_date,
                args.end_date,
            )

        elif attributes["out_var"] == "albedo":
            rsds_dict = ILAMB_DATA_DICT[f"{model}_RSDS"]
            rsus_dict = ILAMB_DATA_DICT[f"{model}_FSR"]
            ilamb_data = get_annual_albedo(
                args.ILAMB_dir,
                attributes["in_var"],
                out_var,
                attributes["units"],
                attributes["longname"],
                attributes["conversion_factor"],
                rsds_dict,
                rsus_dict,
                args.start_date,
                args.end_datetop,
            )

        elif attributes["out_var"] == "biomass":

            ilamb_data = get_biomass_ds(
                args.ILAMB_dir,
                attributes["sub_dir"],
                model,
                attributes["filename"],
                attributes["in_var"],
                max_val=attributes["max_val"],
                min_val=attributes["min_val"],
            )
        else:

            ilamb_data = get_annual_ds(
                args.ILAMB_dir,
                attributes["sub_dir"],
                model,
                attributes["filename"],
                attributes["in_var"],
                attributes["out_var"],
                attributes["conversion_factor"],
                attributes["units"],
                attributes["longname"],
                tstart=args.start_date,
                tstop=args.end_date,
                max_val=attributes["max_val"],
                min_val=attributes["min_val"],
            )

        # write to file
        ilamb_data.to_netcdf(out_file)


if __name__ == "__main__":

    main()
