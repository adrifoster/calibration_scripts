import os
import pandas as pd
import emulation_functions as emf
import argparse
from mpi4py import MPI

from fates_calibration.FATES_calibration_constants import FATES_PFT_IDS, FATES_INDEX

DEFAULT_PARS = {
    'broadleaf_evergreen_tropical_tree': ['fates_turb_leaf_diameter', 'fates_turb_z0mr',
                                          'fates_maintresp_leaf_atkin2017_baserate',
                                          'fates_rad_leaf_clumping_index',
                                          'fates_nonhydro_smpsc', 'fates_nonhydro_smpso',
                                          'fates_leaf_slatop', 'fates_allom_fnrt_prof_a',
                                          'fates_allom_fnrt_prof_b', 'fates_turb_displar']
}

def choose_params(sample_df, sens_df, vars, implausibility_tol, sens_tol):

    # subset out anything over implausibility tolerance
    implaus_vars = [f"{var}_implausibility" for var in vars]
    sample_df['implaus_sum'] = emf.calculate_implaus_sum(sample_df, implaus_vars)
    sample_sub = emf.subset_sample(sample_df, implaus_vars, implausibility_tol)

    # grab only the sensitive parameters
    sensitive_pars = emf.find_sensitive_parameters(sens_df, vars, sens_tol)

    if sample_sub.shape[0] > 0 and len(sensitive_pars) > 0:
        best_sample = emf.find_best_parameter_sets(sample_sub)
        sample_out = best_sample.loc[:, [sensitive_pars, 'implaus_sum']]
    
        return sample_out.reset_index(drop=True)
    else:
        return None
    
def calibration_wave(emulators, param_names, n_samp, obs_df, pft_id, out_dir, wave,
                     implausibility_tol, sens_tol, update_vars=None, default_pars=None,
                     plot_figs=False):
    
    sens_df = emf.sensitivity_analysis(emulators, param_names, pft_id, out_dir, wave,
                                   update_vars=update_vars, default_pars=default_pars,
                                   plot_figs=plot_figs)
    
    sample_df = emf.sample_emulators(emulators, param_names, n_samp, obs_df, out_dir, pft_id,
                     update_vars=update_vars, default_pars=default_pars,
                     plot_figs=plot_figs)
    
    best_sample = choose_params(sample_df, sens_df, list(emulators.keys()),
                                implausibility_tol, sens_tol)

    return best_sample

def find_best_parameters(num_waves, emulators, param_names, n_samp, obs_df, pft_id, out_dir,
        implausibility_tol, sens_tol, default_pars=None):

    update_vars = None
    for wave in range(num_waves):
        if wave == 0:
            best_sample = calibration_wave(emulators, param_names, n_samp,
                                           obs_df, pft_id, out_dir, wave,
                                           implausibility_tol, sens_tol,
                                           update_vars=None, default_pars=default_pars)
        else:
            if best_sample is not None:
                if update_vars is None:
                    update_vars = best_sample
                else:
                    update_vars = pd.concat([update_vars, best_sample], axis=1)
                best_sample = calibration_wave(emulators, param_names, n_samp,
                                               obs_df, pft_id, out_dir, wave,
                                               implausibility_tol, sens_tol,
                                               update_vars=update_vars, 
                                               default_pars=default_pars)
            else:
                return update_vars, wave
    return update_vars, wave

def run_calibration(out_dir, pft, vars, emulator_dir, lhckey, obs_file, n_samp,
                    implausibility_tol, sens_tol, num_waves):
    
    pft_id = FATES_PFT_IDS[pft]
    
    lhkey_df = pd.read_csv(lhckey)
    lhkey_df = lhkey_df.drop(columns=['ensemble'])
    param_names = lhkey_df.columns
    
    obs_df_all = pd.read_csv(obs_file)
    obs_df = obs_df_all[obs_df_all.pft == pft]
    
    emulators = emf.load_all_emulators(pft_id, emulator_dir, vars)
    
    min_max_pars = pd.read_csv('/glade/u/home/afoster/FATES_Calibration/FATES_SP/FATES_LH_min_max_crops.csv')
    
    default_pars = DEFAULT_PARS[pft]
    default_parvals = emf.make_default_values(default_pars, min_max_pars, FATES_INDEX[pft])

    best_param_set, wave = find_best_parameters(num_waves, emulators, param_names, n_samp,
                                          obs_df, pft_id, out_dir, implausibility_tol,
                                          sens_tol, default_pars=default_parvals)
    
    best_param_set['wave'] = wave
    
    return best_param_set

def commandline_args():
    """Parse and return command-line arguments"""

    description = """
    Typical usage: python calibrate_emulate_sample --pft broadleaf_evergreen_tropical_tree

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--pft",
        type=str,
        default=0,
        help="PFT to calibrate\n",
    )
    parser.add_argument(
        "--nsamp",
        type=int,
        default=10000,
        help="Number of samples to emulate\n",
    )
    parser.add_argument(
        "--imp_tol",
        type=float,
        default=1.0,
        help="Implausibility tolerance\n",
    )
    parser.add_argument(
        "--sens_tol",
        type=float,
        default=0.01,
        help="Sensitivity tolerance\n",
    )
    parser.add_argument(
        "--num_waves",
        type=int,
        default=4,
        help="Number of emulated waves\n",
    )
    parser.add_argument(
        "--lhkey",
        type=str,
        default='/glade/u/home/afoster/FATES_Calibration/FATES_SP/LH/lh_key.csv',
        help="path to Latin Hypercube parameter key\n",
    )
    parser.add_argument(
        "--obs_file",
        type=str,
        default='/glade/work/afoster/FATES_calibration/mesh_files/dominant_pft_grid.csv',
        help="path to observations data frame\n",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="Number of times to run calibration\n",
    )

    args = parser.parse_args()

    return args

def main():
    
    comm = MPI.COMM_WORLD
    
    vars = ['GPP', 'EFLX_LH_TOT', 'FSH', 'EF']
    emulator_dir = '/glade/u/home/afoster/FATES_Calibration/FATES_SP/pft_output/emulators'
    top_dir = "/glade/u/home/afoster/FATES_Calibration/FATES_SP/pft_output"
        
    args = commandline_args()
    
    pft_id = FATES_PFT_IDS[args.pft]
    
    out_dir = os.path.join(top_dir, f"{pft_id}_output")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    sample_dir = os.path.join(out_dir, 'samples')
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)
    
    best_sets = []
    for _ in range(args.bootstraps):
        best_param_set = run_calibration(out_dir, args.pft, vars, emulator_dir, args.lhkey,
                                        args.obs_file, args.nsamp, args.imp_tol, 
                                        args.sens_tol, args.num_waves)
        best_sets.append(best_param_set)
    
    file_name = f"param_vals_{str(comm.rank)}.csv"
    best_params = pd.concat(best_sets)
    best_params.to_csv(os.path.join(sample_dir, file_name))
        
if __name__ == "__main__":
    main()


    