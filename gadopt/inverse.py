from os.path import join, isdir
from firedrake import CheckpointFile, utils
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.optimization.rol_solver import (
    ROLVector, ROLObjective, ROLSolver)
from pyadjoint.tape import no_annotations
import shutil
import os
import ROL
from mpi4py import MPI


minimisation_parameters = {
    "General": {
        "Print Verbosity": 1 if MPI.comm.rank == 0 else 0,
        "Output Level": 1 if MPI.comm.rank == 0 else 0,
        "Krylov": {
            "Iteration Limit": 10,
            "Absolute Tolerance": 1e-4,
            "Relative Tolerance": 1e-2,
        },
        "Secant": {
            "Type": "Limited-Memory BFGS",
            "Maximum Storage": 10,
            "Use as Hessian": True,
            "Barzilai-Borwein": 1,
        },
    },
    "Step": {
        "Type": "Trust Region",
        "Trust Region": {
            "Lin-More": {
                "Maximum Number of Minor Iterations": 10,
                "Sufficient Decrease Parameter": 1e-2,
                "Relative Tolerance Exponent": 1.0,
                "Cauchy Point": {
                    "Maximum Number of Reduction Steps": 10,
                    "Maximum Number of Expansion Steps": 10,
                    "Initial Step Size": 1.0,
                    "Normalize Initial Step Size": True,
                    "Reduction Rate": 0.1,
                    "Expansion Rate": 10.0,
                    "Decrease Tolerance": 1e-8,
                },
                "Projected Search": {
                    "Backtracking Rate": 0.5,
                    "Maximum Number of Steps": 20,
                },
            },
            "Subproblem Model": "Lin-More",
            "Initial Radius": 1.0,
            "Maximum Radius": 1e20,
            "Step Acceptance Threshold": 0.05,
            "Radius Shrinking Threshold": 0.05,
            "Radius Growing Threshold": 0.9,
            "Radius Shrinking Rate (Negative rho)": 0.0625,
            "Radius Shrinking Rate (Positive rho)": 0.25,
            "Radius Growing Rate": 10.0,
            "Sufficient Decrease Parameter": 1e-2,
            "Safeguard Size": 100,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 0,
        "Iteration Limit": 100,
    },
}


"""
    LinMore optimisation with checkpointing
"""


_vector_registry = []


class _ROLCheckpointManager_(object):
    def __init__(self):
        # directory to output checkpoints
        self._ROL_checkpoint_dir_ = './'
        self._ROL_mesh_file_ = None
        self._ROL_mesh_name_ = None
        self._index_ = 0

    def set_mesh(self, mesh_file_name, mesh_name):
        self._ROL_mesh_file_ = mesh_file_name
        self._ROL_mesh_name_ = mesh_name

    def set_checkpoint_dir(self, checkpoint_dir):
        # make sure we have the direcotory
        self._makedir_(checkpoint_dir)

        self._ROL_checkpoint_dir_ = checkpoint_dir

    def set_iteration(self, iteration):
        self._index_ = iteration

    def _makedir_(self, dirname):
        if MPI.COMM_WORLD.rank == 0 and \
                not os.path.isdir(dirname):
            os.mkdir(dirname)

        MPI.COMM_WORLD.Barrier()

    def increment_iteration(self):
        self._index_ += 1

    def get_mesh_name(self):
        if None in [self._ROL_mesh_file_, self._ROL_mesh_name_]:
            raise ValueError(
                "First use set_mesh to set a mesh that is checkpointable")
        return self._ROL_mesh_file_, self._ROL_mesh_name_

    def get_checkpoint_dir(self):

        if self._ROL_checkpoint_dir_ is None:
            raise ValueError(
                "set_checkpoint_dir to set the directory")

        subdir_name = join(self._ROL_checkpoint_dir_,
                           f"iteration_{self._index_}")

        self._makedir_(subdir_name)

        return subdir_name

    def get_stale_checkpoint_dir(self):
        """
            Gives the checkpoint directory
            before the current one.
            Primarily used to free up space
        """

        if self._ROL_checkpoint_dir_ is None:
            raise ValueError(
                "set_checkpoint_dir to set the directory")

        subdir_name = join(self._ROL_checkpoint_dir_,
                           f"iteration_{self._index_ - 1}")

        if isdir(subdir_name):
            return subdir_name
        else:
            return None


ROLCheckpointManager = _ROLCheckpointManager_()


class CheckpointedROLVector(ROLVector):
    def __init__(self, dat, inner_product="L2"):
        super().__init__(dat, inner_product)

        with CheckpointFile(ROLCheckpointManager.get_mesh_name()[0],
                            mode='r') as fi:
            self.mesh = fi.load_mesh(ROLCheckpointManager.get_mesh_name()[1])

    def load(self):
        """Load our data once self.dat is populated"""
        with CheckpointFile(self.fname, mode='r') as ckpnt:
            for i, _ in enumerate(self.dat):
                self.dat[i] = \
                    ckpnt.load_function(self.mesh, name=f'dat_{i}')

    def save(self, fname):
        with CheckpointFile(fname, mode='w') as ckpnt:
            ckpnt.save_mesh(self.mesh)
            for i, f in enumerate(self.dat):
                ckpnt.save_function(f, name=f"dat_{i}")

    def clone(self):
        dat = []
        for x in self.dat:
            dat.append(x._ad_copy())
        res = CheckpointedROLVector(dat, inner_product=self.inner_product)
        res.scale(0.0)
        return res

    def __setstate__(self, state):
        """Set the state from unpickling

        Requires self.dat to be separately set, then self.load()
        can be called.
        """

        # initialise C++ state
        super().__init__(state)

        self.fname, self.inner_product = state

        with CheckpointFile(ROLCheckpointManager.get_mesh_name()[0],
                            mode='r') as fi:
            self.mesh = fi.load_mesh(ROLCheckpointManager.get_mesh_name()[1])

        _vector_registry.append(self)

    def __getstate__(self):
        """Return a state tuple suitable for pickling"""

        fname = join(ROLCheckpointManager.get_checkpoint_dir(),
                     "vector_checkpoint_{}.h5".format(utils._new_uid()))
        self.save(fname)

        return (fname, self.inner_product)


class CheckPointedROLSolver(ROLSolver):
    def __init__(self, problem, parameters, inner_product="L2"):
        super().__init__(problem, parameters)
        OptimizationSolver.__init__(self, problem, parameters)
        self.rolobjective = ROLObjective(problem.reduced_functional)
        x = [p.tape_value() for p in self.problem.reduced_functional.controls]
        self.rolvector = CheckpointedROLVector(x, inner_product=inner_product)
        self.params_dict = parameters

        # self.bounds = super(CheckPointedROLSolver, self).__get_bounds()
        # self.constraints = self.__get_constraints()


class LinMoreOptimiser(object):
    def __init__(self, minimisation_problem, parameters, callback=None):

        self.rol_solver = CheckPointedROLSolver(
            minimisation_problem, parameters, inner_product='L2')

        self.rol_parameters = ROL.ParameterList(
            parameters, "Parameters")

        self.rol_secant = ROL.InitBFGS(
            parameters.get('General').get('Secant').get('Maximum Storage'))

        self.rol_algorithm = ROL.LinMoreAlgorithm(
            self.rol_parameters, self.rol_secant)

        self.rol_algorithm.setStatusTest(
            self.StatusTest(self.rol_parameters,
                            self.rol_solver.rolvector,
                            self),
            False)

        self.callback = callback

    # solving the optimisation problem
    def run(self):
        self.rol_algorithm.run(
            self.rol_solver.rolvector,
            self.rol_solver.rolobjective,
            self.rol_solver.bounds)

    #
    def checkpoint(self):

        ROL.serialise_secant(self.rol_secant,
                             MPI.COMM_WORLD.rank,
                             ROLCheckpointManager.get_checkpoint_dir())

        ROL.serialise_algorithm(self.rol_algorithm,
                                MPI.COMM_WORLD.rank,
                                ROLCheckpointManager.get_checkpoint_dir())

        with CheckpointFile(join(ROLCheckpointManager.get_checkpoint_dir(),
                                 "solution_checkpoint.h5"),
                            mode='w') as ckpnt:
            ckpnt.save_mesh(self.rol_solver.rolvector.mesh)
            for i, f in enumerate(self.rol_solver.rolvector.dat):
                ckpnt.save_function(f, name=f"dat_{i}")

    def reload(self, iteration):
        ROLCheckpointManager.set_iteration(iteration)

        ROL.load_secant(self.rol_secant,
                        MPI.COMM_WORLD.rank,
                        ROLCheckpointManager.get_checkpoint_dir())
        ROL.load_algorithm(self.rol_algorithm,
                           MPI.COMM_WORLD.rank,
                           ROLCheckpointManager.get_checkpoint_dir())

        # Reloading the solution
        self.rol_solver.rolvector.fname = join(
            ROLCheckpointManager.get_checkpoint_dir(),
            "solution_checkpoint.h5")
        self.rol_solver.rolvector.load()

        vec = self.rol_solver.rolvector.dat
        for v in _vector_registry:
            x = [p.copy(deepcopy=True) for p in vec]
            v.dat = x
            v.load()

    class StatusTest(ROL.StatusTest):
        def __init__(self, params, vector, parent_optimiser):
            super().__init__(params)

            # This is to access outer object
            self.parent_optimiser = parent_optimiser

            # Keep track of the vector that is being passed to StatusCheck
            self.vector = vector

            self.my_idx = 0

        @no_annotations
        def check(self, status):

            # Checkpointing
            self.parent_optimiser.checkpoint()

            # Free up space from previous checkpoint
            if (ROLCheckpointManager.get_stale_checkpoint_dir()
                    is not None and MPI.COMM_WORLD.rank == 0):
                shutil.rmtree(ROLCheckpointManager.get_stale_checkpoint_dir())

            # Barriering
            MPI.COMM_WORLD.Barrier()

            # If there is a user defined
            # callback function call it
            if self.parent_optimiser.callback is not None:
                self.parent_optimiser.callback()

            # Write out the solution
            self.my_idx += 1
            ROLCheckpointManager.increment_iteration()

            return ROL.StatusTest.check(self, status)
