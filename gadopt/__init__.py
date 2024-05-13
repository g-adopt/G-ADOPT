from firedrake import *
from firedrake.output import VTKFile
from .time_stepper import CrankNicolsonRK, ImplicitMidpoint
from .limiter import VertexBasedP1DGLimiter
from .utility import log, ParameterLog, TimestepAdaptor, LayerAveraging, timer_decorator, InteriorBC
from .diagnostics import GeodynamicalDiagnostics
from .momentum_equation import StokesEquations
from .scalar_equation import EnergyEquation
from .stokes_integrators import StokesSolver, create_stokes_nullspace
from .energy_solver import EnergySolver
from .approximations import BoussinesqApproximation, ExtendedBoussinesqApproximation, AnelasticLiquidApproximation, TruncatedAnelasticLiquidApproximation
from .preconditioners import SPDAssembledPC

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()
