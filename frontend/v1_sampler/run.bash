source activate lume2
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
srun -n 1024 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml
#srun -n 32 python -m mpi4py.futures -m xopt.mpi.run xopt.yaml
#python run.py xopt.yaml
#python -m xopt.run xopt.yaml
