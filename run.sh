#PBS -Pw85
#PBS -qnormal
#PBS -N tcr
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=12:00:00
#PBS -lmem=380GB,ncpus=96,jobfs=4000MB
#PBS -joe
#PBS -W umask=002
#PBS -lstorage=gdata/w85+scratch/w85+gdata/hh5+gdata/rt52

umask 0002

module purge
module load pbs
module load dot
module use /g/data/hh5/public/modules
module load conda/analysis3

cd /g/data/w85/software/tcr

export PYTHONPATH=$PYTHONPATH:/scratch/$PROJECT/$USER/python/lib/python3.10/site-packages

# Set the number of processors for use in Dask to match the number available on the job
sed -i "s/n_procs = [0-9]\+/n_procs = $PBS_NCPUS/" namelist.py
start=$(date +%s)
python3 run.py GL 10 > tcr.nwaves.log 2>&1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
