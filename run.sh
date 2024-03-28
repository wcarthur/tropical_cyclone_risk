#PBS -Pw85
#PBS -qnormal
#PBS -N tcr
#PBS -m ae
#PBS -M craig.arthur@ga.gov.au
#PBS -lwalltime=1:00:00
#PBS -lmem=190GB,ncpus=48,jobfs=4000MB
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

start=$(date +%s)
mpirun -np $PBS_NCPUS python3 run.py GL > tcr.log 2>&1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
