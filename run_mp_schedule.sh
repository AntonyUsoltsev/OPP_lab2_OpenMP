#!/bin/sh

#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=4:ompthreads=4:mem=10000m,place=scatter
#PBS -m n
#PBS -o out_sch.txt
#PBS -e err_sch.txt

cd $PBS_O_WORKDIR
echo
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
./task_sch