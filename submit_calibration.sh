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

pft=broadleaf_evergreen_tropical_tree

module load conda
conda activate ml_analysis

mpiexec -n 16 python calibrate_emulate_sample.py --pft ${pft} --num_samples 100