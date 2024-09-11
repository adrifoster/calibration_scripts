#!/bin/bash

#PBS -N FATES_calib
#PBS -q casper
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=10G
#PBS -l walltime=12:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -m abe
#PBS -M afoster@ucar.edu

pft=needleleaf_evergreen_extratrop_tree
pft_id=NEET

module load conda
conda activate ml_analysis

pft_dir=/glade/u/home/afoster/FATES_Calibration/pft_output/${pft_id}_outputs/
sample_dir=${pft_dir}/samples

if [ ! -d "${pft_dir}" ]; then
  mkdir ${pft_dir}
fi
if [ ! -d "${sample_dir}" ]; then
  mkdir ${sample_dir}
fi

cd /glade/u/home/afoster/FATES_Calibration/scripts
mpiexec -n 16 python calibrate_emulate_sample.py --pft ${pft} --bootstraps 100
