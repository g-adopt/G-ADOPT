#PBS -N RobinBegins   
#PBS -P xd2
#PBS -q normalsr 
#PBS -l walltime=24:00:00
#PBS -l mem=6500GB
#PBS -l ncpus=1352
#PBS -l jobfs=5200GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#PBS -M siavash.ghelichkhan@anu.edu.au
#PBS -m abe
#### Load relevant modules:

module load python3/3.10.4 openmpi/4.0.7

export PETSC_DIR="/tmp/firedrake-prefix"
# export PETSC_OPTIONS="-log_view"
export PETSC_ARC=""
export OMP_NUM_THREADS=1
export LD_PRELOAD=${INTEL_MKL_ROOT}/lib/intel64/libmkl_sequential.so:${INTEL_MKL_ROOT}/lib/intel64/libmkl_core.so:${LD_PRELOAD}
mpiexec --map-by ppr:1:node -np $PBS_NNODES tar -C $PBS_JOBFS -xf /g/data/xd2/sg8812/FD-IMAGES/fd-prefix-2024_04_08-full-openmpi407-opt-64bitInt-pyop2fixed_update2024Apr24.tar.gz
mpiexec --map-by ppr:1:node -np $PBS_NNODES ln -s $PBS_JOBFS/firedrake-prefix /tmp/firedrake-prefix

# Setting Python directories
# Each core needs a python instance
export PYTHONUSERBASE=/tmp/firedrake-prefix
export XDG_CACHE_HOME=$PBS_JOBFS/xdg
export MPLCONFIGDIR=$PBS_JOBFS/firedrake-prefix

# This is to make sure we only compile on rank 0
export PYOP2_CACHE_DIR=/scratch/xd2/sg8812/g-adopt/demos/adjoint_spherical/runs/00_taylor_test/pyop/
export PYOP2_NODE_LOCAL_COMPILATION=0
export OMPI_MCA_io="ompio"

# Making sure all nodes have matplotlib
mpiexec --map-by ppr:1:node -np $PBS_NNODES  python3 -c "import matplotlib.pyplot as plt"

mpiexec -np $PBS_NCPUS python3 ./adjoint.py 0 >output.log 2>error.log
