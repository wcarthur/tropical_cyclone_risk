#PBS -A UPSU0032
#PBS -N download_EC-Earth3P
#PBS -l walltime=12:00:00
#PBS -M ajb8224@psu.edu
#PBS -l select=1:ncpus=1
#PBS -m ae
#PBS -q casper

script=download_cmip6.py

source ~/.bashrc
conda activate tc_risk
python3 ${script}

