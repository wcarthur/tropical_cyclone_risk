#PBS -A UPSU0032
#PBS -N era5_hadgem-lm
#PBS -l walltime=14:00:00
#PBS -M ajb8224@psu.edu
#PBS -l select=1:ncpus=1
#PBS -m ae
#PBS -q casper

script=run.py
VAR=NA

source ~/.bashrc
conda activate tc_risk
python3 ${script} ${VAR}

