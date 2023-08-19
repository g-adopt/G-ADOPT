from firedrake import CheckpointFile
import firedrake.utils
from firedrake.adjoint import *  # noqa: F401
from pathlib import Path
from mpi4py import MPI
from pyadjoint.optimization.optimization_solver import OptimizationSolver
import pyadjoint.optimization.rol_solver as pyadjoint_rol
import ROL

# emulate the previous behaviour of firedrake_adjoint by automatically
# starting the tape
continue_annotation()

_vector_registry = []


class ROLSolver(pyadjoint_rol.ROLSolver):
    def __init__(self, problem, parameters, inner_product="L2", vector_class=pyadjoint_rol.ROLVector, vector_args=[]):
        OptimizationSolver.__init__(self, problem, parameters)
        self.rolobjective = pyadjoint_rol.ROLObjective(problem.reduced_functional)
        x = [p.tape_value() for p in self.problem.reduced_functional.controls]
        self.rolvector = vector_class(x, *vector_args, inner_product=inner_product)
        self.params_dict = parameters

        self.bounds = self.__get_bounds()
        self.constraints = self.__get_constraints()


class CheckpointedROLVector(pyadjoint_rol.ROLVector):
    def __init__(self, dat, optimiser, inner_product="L2"):
        super().__init__(dat, inner_product)

        self._optimiser = optimiser

    def clone(self):
        dat = []
        for x in self.dat:
            dat.append(x._ad_copy())
        res = CheckpointedROLVector(dat, self._optimiser, inner_product=self.inner_product)
        res.scale(0.0)
        return res

    def save(self, checkpoint_path):
        """Checkpoint the data within this vector to disk.

        Called when this object is pickled as part of the ROL
        state serialisation.
        """

        with CheckpointFile(str(checkpoint_path), "w") as f:
            for i, func in enumerate(self.dat):
                f.save_function(func, name=f"dat_{i}")

    def load(self, mesh):
        """Load the checkpointed data for this vector from disk.

        Called by the parent Optimiser after the ROL state has
        been deserialised. The pickling routine will register
        this vector within the registry.
        """

        with CheckpointFile(str(self.checkpoint_path), "r") as f:
            for i in range(len(self.dat)):
                self.dat[i] = f.load_function(mesh, name=f"dat_{i}")

    def __setstate__(self, state):
        """Set the state from the result of unpickling.

        This happens during the restoration of a checkpoint. self.dat needs to be
        separately set, then self.load() can be called.
        """

        # initialise C++ state
        super().__init__(state)
        self.checkpoint_path, self.inner_product = state

        _vector_registry.append(self)

    def __getstate__(self):
        """Return a state tuple suitable for pickling"""

        checkpoint_filename = f"vector_checkpoint_{firedrake.utils._new_uid()}.h5"
        checkpoint_path = self._optimiser.checkpoint_dir / checkpoint_filename
        self.save(checkpoint_path)

        return (checkpoint_path, self.inner_product)


class LinMoreOptimiser:
    def __init__(self, problem, parameters, checkpoint_dir=None, auto_checkpoint=True):
        """The management class for Lin-More trust region optimisation using ROL.

        This class sets up ROL to use the Lin-More trust region method, with a limited-memory
        BFGS secant for determining the steps. A pyadjoint problem has to be set up first,
        containing the optimisation functional and other constraints (like bounds).

        This optimiser also supports checkpointing ROL's state, to allow resumption of
        a previous optimisation without having to refill the L-BFGS memory. The underlying
        objects will be configured for checkpointing if `checkpoint_dir` is specified,
        and optionally the automatic checkpointing each iteration can be disabled by the
        `auto_checkpoint` paramater.

        Params:
            problem (pyadjoint.MinimizationProblem): The actual problem to solve.
            parameters (dict): A dictionary containing the parameters governing ROL.
            checkpoint_dir (Optional[str]): A path to hold any checkpoints that are saved.
            auto_checkpoint (Optional[bool]): Whether to automatically checkpoint each iteration.
        """

        self.iteration = -1

        solver_kwargs = {}
        if checkpoint_dir is not None:
            self._base_checkpoint_dir = Path(checkpoint_dir)
            self._ensure_checkpoint_dir()

            self._mesh = problem.reduced_functional.controls[0].control.function_space().mesh()
            solver_kwargs["vector_class"] = CheckpointedROLVector
            solver_kwargs["vector_args"] = [self]

            self.auto_checkpoint = auto_checkpoint
        else:
            self._base_checkpoint_dir = None
            self.auto_checkpoint = False

        self.rol_solver = ROLSolver(problem, parameters, inner_product="L2", **solver_kwargs)
        self.rol_parameters = ROL.ParameterList(parameters, "Parameters")

        try:
            self.rol_secant = ROL.lBFGS(parameters["General"]["Secant"]["Maximum Storage"])
        except KeyError:
            # Use the default storage value
            self.rol_secant = ROL.lBFGS()

        self.rol_algorithm = ROL.LinMoreAlgorithm(self.rol_parameters, self.rol_secant)
        self.callbacks = []

        self._add_statustest()

    def _ensure_checkpoint_dir(self):
        if MPI.COMM_WORLD.rank == 0:
            self.checkpoint_dir.mkdir(exist_ok=True)

        MPI.COMM_WORLD.Barrier()

    @property
    def checkpoint_dir(self):
        if self.iteration == -1:
            return self._base_checkpoint_dir

        return self._base_checkpoint_dir / str(self.iteration)

    def checkpoint(self):
        """Checkpoint the current ROL state to disk."""

        ROL.serialise_algorithm(self.rol_algorithm, MPI.COMM_WORLD.rank, str(self.checkpoint_dir))

        checkpoint_path = self.checkpoint_dir / "solution_checkpoint.h5"
        with CheckpointFile(str(checkpoint_path), "w") as f:
            for i, func in enumerate(self.rol_solver.rolvector.dat):
                f.save_function(func, name=f"dat_{i}")

    def restore(self, iteration):
        self.iteration = iteration
        self.rol_algorithm = ROL.load_algorithm(MPI.COMM_WORLD.rank, str(self.checkpoint_dir))
        self._add_statustest()

        self.rol_solver.rolvector.checkpoint_path = self.checkpoint_dir / "solution_checkpoint.h5"
        self.rol_solver.rolvector.load(self._mesh)

        # The various ROLVector objects can load all their metadata, but can't actually
        # restore from the Firedrake checkpoint. They register themselves, so we can access
        # them through a flat list.
        vec = self.rol_solver.rolvector.dat
        for v in _vector_registry:
            x = [p.copy(deepcopy=True) for p in vec]
            v.dat = x
            v.load(self._mesh)
            v._optimiser = self

        _vector_registry.clear()

    def run(self):
        self.rol_algorithm.run(
            self.rol_solver.rolvector,
            self.rol_solver.rolobjective,
            self.rol_solver.bounds,
        )

    def _add_statustest(self):
        class StatusTest(ROL.StatusTest):
            def check(inner_self, status):
                self.iteration += 1

                if self.auto_checkpoint:
                    self._ensure_checkpoint_dir()
                    self.checkpoint()

                for callback in self.callbacks:
                    callback()

                return super().check(status)

        # Don't chain with the default status test
        self.rol_algorithm.setStatusTest(StatusTest(self.rol_parameters), False)

    def add_callback(self, callback):
        self.callbacks.append(callback)


minimisation_parameters = {
    "General": {
        "Print Verbosity": 1 if MPI.COMM_WORLD.rank == 0 else 0,
        "Output Level": 1 if MPI.COMM_WORLD.rank == 0 else 0,
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