#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=16:00:00
#SBATCH --nodes=10
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

#SBATCH -J cu_inj_frontend/v1_csnga
#SBATCH --mail-user=cmayes@stanford.edu
#SBATCH --mail-type=ALL

source activate lume
export HDF5_USE_FILE_LOCKING=FALSE
srun -n 320  python -m mpi4py.futures -m xopt.mpi.run xopt.yaml
