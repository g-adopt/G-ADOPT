"""Provides a set of classes and functions enabling multi-material capabilities

The `interface_thickness` and `conservative_level_set` functions provide default
strategies to initialise a level-set field. The `material_field` function enables
defining a UFL expression for a physical field depending on material properties. The
`LevelSetSolver` class instantiates the advection and reinitialisation systems for a
single level set. The `entrainment` function enables users to easily calculate material
entrainment.
"""

import operator
import re
from collections.abc import Callable
from typing import Any

import firedrake as fd
import numpy as np
import shapely as sl
from mpi4py import MPI

from . import scalar_equation as scalar_eq
from .equations import Equation
from .time_stepper import eSSPRKs3p3, eSSPRKs10p3
from .transport_solver import GenericTransportSolver
from .utility import node_coordinates


def signed_distance(
    level_set: fd.Function,
    /,
    interface_geometry: str,
    interface_coordinates: list[list[float]] | list[list[float], float] | None = None,
    *,
    interface_callable: Callable | str | None = None,
    interface_args: tuple[Any] | None = None,
    boundary_coordinates: list[list[float]] | np.ndarray | None = None,
) -> list:
    """Generates signed-distance function values at level-set nodes.

    Three scenarios are currently implemented:
    - The material interface is described by a mathematical function y = f(x). In this
      case, `interface_geometry` should be "curve" and `interface_callable` must be
      provided along with any `interface_args` to implement the aforementioned
      mathematical function.
    - The material interface is a polygon, and `interface_geometry` takes the value
      "polygon". In this case, `interface_coordinates` must exclude the polygon sides
      that do not act as a material interface and coincide with domain boundaries. The
      coordinates of these sides should be provided using the `boundary_coordinates`
      argument such that the concatenation of the two coordinate objects describes a
      closed polygonal chain.
    - The material interface is a circle, and `interface_geometry` takes the value
      "circle". In this case, `interface_coordinates` is a list holding the coordinates
      of the circle's centre and the circle radius. No other arguments are required.

    Geometrical objects underpinning material interfaces are generated using Shapely.

    Implemented interface geometry presets and associated arguments:
    | Curve     |                     Arguments                      |
    | :-------- | :------------------------------------------------: |
    | line      | slope, intercept                                   |
    | cosine    | amplitude, wavelength, vertical_shift, phase_shift |
    | rectangle | ref_vertex_coords, edge_sizes                      |

    Args:
      level_set:
        A Firedrake function for the targeted level-set field
      interface_geometry:
        A string specifying the geometry to create
      interface_coordinates:
        A sequence or an array-like with shape (N, 2) of numeric coordinate pairs
        defining the interface or a list containing centre coordinates and radius
      interface_callable:
        A callable implementing the mathematical function depicting the interface or a
        string matching an implemented callable preset
      interface_args:
        A tuple of arguments provided to the interface callable
      boundary_coordinates:
        A sequence of numeric coordinate pairs or an array-like with shape (N, 2)

    Returns:
        A list of signed-distance function values at the level-set nodes
    """

    def stack_coordinates(func):
        """Decorator to stack coordinates when the material interface is a curve.

        Args:
          func:
            A callable implementing the mathematical function depicting the interface

        Returns:
          A callable that can stack interface coordinates
        """

        def wrapper(*args):
            if isinstance(interface_coords_x := args[0], (int, float)):
                return func(*args)
            else:
                return np.column_stack((interface_coords_x, func(*args)))

        return wrapper

    def line(x, slope, intercept) -> float | np.ndarray:
        """Straight line equation"""
        return slope * x + intercept

    def cosine(
        x, amplitude, wavelength, vertical_shift, phase_shift=0
    ) -> float | np.ndarray:
        """Cosine function with an amplitude and a vertical shift."""
        cosine = np.cos(2 * np.pi / wavelength * x + phase_shift)

        return amplitude * cosine + vertical_shift

    def rectangle(
        ref_vertex_coords: tuple[float], edge_sizes: tuple[float]
    ) -> list[tuple[float]]:
        """Material interface defined by a rectangle.

        Edges are aligned with Cartesian directions and do not overlap domain boundaries.

        Args:
          ref_vertex_coords:
            A tuple holding the coordinates of the lower-left vertex
          edge_sizes:
            A tuple holding the edge sizes

        Returns:
          A list of tuples representing the coordinates of the rectangle's vertices
        """
        interface_coords = [
            (ref_vertex_coords[0], ref_vertex_coords[1]),
            (ref_vertex_coords[0] + edge_sizes[0], ref_vertex_coords[1]),
            (
                ref_vertex_coords[0] + edge_sizes[0],
                ref_vertex_coords[1] + edge_sizes[1],
            ),
            (ref_vertex_coords[0], ref_vertex_coords[1] + edge_sizes[1]),
            (ref_vertex_coords[0], ref_vertex_coords[1]),
        ]

        return interface_coords

    callable_presets = {"cosine": cosine, "line": line, "rectangle": rectangle}
    if isinstance(interface_callable, str):
        interface_callable = callable_presets[interface_callable]

    if interface_callable is not None:
        if interface_geometry == "curve":
            interface_callable = stack_coordinates(interface_callable)
        interface_coordinates = interface_callable(*interface_args)

    match interface_geometry:
        case "curve":
            interface = sl.LineString(interface_coordinates)

            signed_distance = [
                (1 if y > interface_callable(x, *interface_args[1:]) else -1)
                * interface.distance(sl.Point(x, y))
                for x, y in node_coordinates(level_set)
            ]
        case "polygon":
            if boundary_coordinates is None:
                interface = sl.Polygon(interface_coordinates)
                sl.prepare(interface)

                signed_distance = [
                    (1 if interface.contains(sl.Point(x, y)) else -1)
                    * interface.boundary.distance(sl.Point(x, y))
                    for x, y in node_coordinates(level_set)
                ]
            else:
                interface = sl.LineString(interface_coordinates)
                interface_with_boundaries = sl.Polygon(
                    np.vstack((interface_coordinates, boundary_coordinates))
                )
                sl.prepare(interface_with_boundaries)

                signed_distance = [
                    (1 if interface_with_boundaries.contains(sl.Point(x, y)) else -1)
                    * interface.distance(sl.Point(x, y))
                    for x, y in node_coordinates(level_set)
                ]
        case "circle":
            centre, radius = interface_coordinates
            interface = sl.Point(centre).buffer(radius)
            sl.prepare(interface)

            signed_distance = [
                (1 if interface.contains(sl.Point(x, y)) else -1)
                * interface.boundary.distance(sl.Point(x, y))
                for x, y in node_coordinates(level_set)
            ]
        case _:
            raise ValueError(
                "`interface_geometry` must be 'curve', 'polygon', or 'circle'."
            )

    return signed_distance


def interface_thickness(level_set: fd.Function, scale: float = 0.25) -> fd.Function:
    """Default strategy for the thickness of the conservative level set profile.

    Args:
      level_set:
        A Firedrake function for the level-set field
      scale:
        A float to control thickness values relative to cell sizes

    Returns:
      A Firedrake function holding the interface thickness values
    """
    epsilon = fd.Function(level_set, name="Interface thickness")
    epsilon.interpolate(scale * level_set.ufl_domain().cell_sizes)

    return epsilon


def conservative_level_set(
    signed_distance: list | np.ndarray, epsilon: float | fd.Function
) -> np.ndarray:
    """Returns the conservative level set profile for a given signed-distance function.

    Args:
      signed_distance:
        A list or NumPy array holding the signed-distance function values
      epsilon:
        A float or Firedrake function representing the interface thickness

    Returns:
      A NumPy array holding the conservative level-set node values
    """
    if isinstance(epsilon, fd.Function):
        epsilon = epsilon.dat.data

    return (1 + np.tanh(np.asarray(signed_distance) / 2 / epsilon)) / 2


def reinitialisation_term(
    eq: Equation, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
) -> fd.Form:
    """Term for the conservative-level-set reinitialisation equation.

    Implements terms on the right-hand side of Equation 17 from
    Parameswaran, S., & Mandal, J. C. (2023).
    A stable interface-preserving reinitialization equation for conservative level set
    method.
    European Journal of Mechanics-B/Fluids, 98, 40-63.
    """
    sharpen_term = -trial * (1 - trial) * (1 - 2 * trial) * eq.test * eq.dx
    balance_term = (
        eq.epsilon
        * (1 - 2 * trial)
        * fd.sqrt(fd.inner(eq.level_set_grad, eq.level_set_grad))
        * eq.test
        * eq.dx
    )

    return sharpen_term + balance_term


reinitialisation_term.required_attrs = {"epsilon", "level_set_grad"}
reinitialisation_term.optional_attrs = set()


class LevelSetSolver:
    """Solver for the conservative level-set approach.

    Advects and reinitialises a level-set field.

    Attributes:
      solution:
        The Firedrake function holding current level-set values
      solution_old:
        The Firedrake function holding previous level-set values
      solution_grad:
        The Firedrake function holding current level-set gradient values
      solution_space:
        The Firedrake function space where the level set lives
      mesh:
        The Firedrake mesh representing the numerical domain
      advection:
        A boolean specifying whether advection is set up
      reinitialisation:
        A boolean specifying whether reinitialisation is set up
      adv_kwargs:
        A dictionary holding the parameters used to set up advection
      reini_kwargs:
        A dictionary holding the parameters used to set up reinitialisation
      adv_solver:
        A G-ADOPT GenericTransportSolver tackling advection
      proj_solver:
        A Firedrake LinearVariationalSolver to project the level-set gradient
      reini_integrator:
        A G-ADOPT time integrator tackling reinitialisation
      step:
        An integer representing the number of advection steps already made
    """

    def __init__(
        self,
        level_set: fd.Function,
        /,
        *,
        adv_kwargs: dict[str, Any] | None = None,
        reini_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialises the solver instance.

        Valid keys for `adv_kwargs`:
        |    Argument     | Required |                   Description                   |
        | :-------------- | :------: | :---------------------------------------------: |
        | u               | True     | Velocity field (`Function`)                     |
        | timestep        | True     | Integration time step (`Constant`)              |
        | time_integrator | False    | Time integrator (class in `time_stepper.py`)    |
        | bcs             | False    | Boundary conditions (`dict`, G-ADOPT API)       |
        | solver_params   | False    | Solver parameters (`dict`, PETSc API)           |
        | subcycles       | False    | Advection iterations in one time step (`int`)   |

        Valid keys for `reini_kwargs`:
        |    Argument     | Required |                   Description                   |
        | :-------------- | :------: | :---------------------------------------------: |
        | epsilon         | True     | Interface thickness (`float` or `Function`)     |
        | timestep        | False    | Integration step in pseudo-time (`int`)         |
        | time_integrator | False    | Time integrator (class in `time_stepper.py`)    |
        | solver_params   | False    | Solver parameters (`dict`, PETSc API)           |
        | steps           | False    | Pseudo-time integration steps (`int`)           |
        | frequency       | False    | Advection steps before reinitialisation (`int`) |

        Args:
          level_set:
            The Firedrake function holding the level-set field
          adv_kwargs:
            A dictionary with parameters used to set up advection
          reini_kwargs:
            A dictionary with parameters used to set up reinitialisation
        """
        self.solution = level_set
        self.solution_old = fd.Function(self.solution)
        self.solution_space = level_set.function_space()
        self.advection = False
        self.reinitialisation = False

        if isinstance(adv_kwargs, dict):
            self.advection = True
            self.adv_kwargs = adv_kwargs

            self.set_default_advection_args()

        if isinstance(reini_kwargs, dict):
            self.reinitialisation = True
            self.reini_kwargs = reini_kwargs

            self.mesh = self.solution.ufl_domain()

            self.set_default_reinitialisation_args()

            self.proj_solver = self.gradient_L2_proj()

        self._solvers_ready = False

    def set_default_advection_args(self) -> None:
        """Set default values of optional arguments if not provided."""
        if "time_integrator" not in self.adv_kwargs:
            self.adv_kwargs["time_integrator"] = eSSPRKs10p3
        if "bcs" not in self.adv_kwargs:
            self.adv_kwargs["bcs"] = {}
        if "solver_params" not in self.adv_kwargs:
            self.adv_kwargs["solver_params"] = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        if "subcycles" not in self.adv_kwargs:
            self.adv_kwargs["subcycles"] = 1

    def set_default_reinitialisation_args(self) -> None:
        """Set default values of optional parameters if not provided."""
        if "timestep" not in self.reini_kwargs:
            self.reini_kwargs["timestep"] = 0.02
        if "time_integrator" not in self.reini_kwargs:
            self.reini_kwargs["time_integrator"] = eSSPRKs3p3
        if "solver_params" not in self.reini_kwargs:
            self.reini_kwargs["solver_params"] = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        if "steps" not in self.reini_kwargs:
            self.reini_kwargs["steps"] = 1
        if "frequency" not in self.reini_kwargs and self.mesh.cartesian:
            self.reini_kwargs["frequency"] = self.reinitialisation_frequency()

    def reinitialisation_frequency(self) -> int:
        """Implements default strategy for the reinitialisation frequency.

        Reinitialisation becomes less frequent as mesh resolution increases, with the
        underlying assumption that the minimum cell size occurs along the material
        interface. The current strategy is to apply reinitialisation at every time step
        up to a certain cell size and then scale the frequency with the decrease in cell
        size.
        """
        max_coords = self.mesh.coordinates.dat.data.max(axis=0)
        min_coords = self.mesh.coordinates.dat.data.min(axis=0)
        for i in range(len(max_coords)):
            max_coords[i] = self.mesh.comm.allreduce(max_coords[i], MPI.MAX)
            min_coords[i] = self.mesh.comm.allreduce(min_coords[i], MPI.MIN)
        domain_size = np.sqrt(np.sum((max_coords - min_coords) ** 2))

        epsilon = self.reini_kwargs["epsilon"]
        if isinstance(epsilon, fd.Function):
            epsilon = self.mesh.comm.allreduce(epsilon.dat.data.min(), MPI.MIN)

        return max(1, round(4.9e-3 * domain_size / epsilon - 0.25))

    def gradient_L2_proj(self) -> fd.LinearVariationalSolver:
        """Constructs a projection solver.

        Projects the level-set gradient from a discontinuous function space to the
        equivalent continuous one.

        Returns:
          A Firedrake solver capable of projecting a discontinuous gradient field on a
          continuous function space
        """
        grad_name = "Level-set gradient"
        if number_match := re.search(r"\s#\d+$", self.solution.name()):
            grad_name += number_match.group()

        gradient_space = fd.VectorFunctionSpace(
            self.mesh, "CG", self.solution.ufl_element().degree()
        )
        self.solution_grad = fd.Function(gradient_space, name=grad_name)

        test = fd.TestFunction(gradient_space)
        trial = fd.TrialFunction(gradient_space)

        mass_term = fd.inner(test, trial) * fd.dx(domain=self.mesh)
        residual_element = self.solution * fd.div(test) * fd.dx(domain=self.mesh)
        residual_boundary = (
            self.solution
            * fd.dot(test, fd.FacetNormal(self.mesh))
            * fd.ds(domain=self.mesh)
        )
        residual = -residual_element + residual_boundary

        problem = fd.LinearVariationalProblem(mass_term, residual, self.solution_grad)
        solver_params = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        return fd.LinearVariationalSolver(problem, solver_parameters=solver_params)

    def set_up_solvers(self) -> None:
        """Sets up time integrators for advection and reinitialisation as required."""
        if self.advection:
            self.adv_solver = GenericTransportSolver(
                "advection",
                self.solution,
                self.adv_kwargs["timestep"] / self.adv_kwargs["subcycles"],
                self.adv_kwargs["time_integrator"],
                solution_old=self.solution_old,
                eq_attrs={"u": self.adv_kwargs["u"]},
                bcs=self.adv_kwargs["bcs"],
                solver_parameters=self.adv_kwargs["solver_params"],
            )

        if self.reinitialisation:
            reinitialisation_equation = Equation(
                fd.TestFunction(self.solution_space),
                self.solution_space,
                reinitialisation_term,
                mass_term=scalar_eq.mass_term,
                eq_attrs={
                    "level_set_grad": self.solution_grad,
                    "epsilon": self.reini_kwargs["epsilon"],
                },
            )

            self.reini_integrator = self.reini_kwargs["time_integrator"](
                reinitialisation_equation,
                self.solution,
                self.reini_kwargs["timestep"],
                solution_old=self.solution_old,
                solver_parameters=self.reini_kwargs["solver_params"],
            )

        self.step = 0
        self._solvers_ready = True

    def update_level_set_gradient(self) -> None:
        """Calls the gradient projection solver.

        Can be provided as a callback to time integrators.
        """
        self.proj_solver.solve()

    def reinitialise(self) -> None:
        """Performs reinitialisation steps."""
        for _ in range(self.reini_kwargs["steps"]):
            self.reini_integrator.advance(
                update_forcings=self.update_level_set_gradient
            )

    def solve(
        self,
        disable_advection: bool = False,
        disable_reinitialisation: bool = False,
    ) -> None:
        """Updates the level-set function by means of advection and reinitialisation.

        Args:
          disable_advection:
            A boolean to disable the advection solve.
          disable_reinitialisation:
            A boolean to disable the reinitialisation solve.
        """
        if not self._solvers_ready:
            self.set_up_solvers()

        if self.advection and not disable_advection:
            for _ in range(self.adv_kwargs["subcycles"]):
                self.adv_solver.solve()
                self.step += 1

                if self.reinitialisation and not disable_reinitialisation:
                    if self.step % self.reini_kwargs["frequency"] == 0:
                        self.reinitialise()

        elif self.reinitialisation and not disable_reinitialisation:
            if self.step % self.reini_kwargs["frequency"] == 0:
                self.reinitialise()


def material_field_from_copy(
    level_set: fd.Function | list[fd.Function],
    field_values: list,
    interface: str,
) -> fd.ufl.algebra.Sum | fd.ufl.algebra.Product | fd.ufl.algebra.Division:
    """Generates UFL algebra describing a physical property across the domain.

    Ensures that the correct expression is assigned to each material based on the
    level-set functions. Property transitions across material interfaces are expressed
    according to the provided strategy.

    Args:
      level_set:
        A Firedrake function for the level set (or a list of these)
      field_values:
        A list of physical property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL algebra representing the physical property throughout the domain

    """
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:  # Directly specify material value on only one side of the interface
        match interface:
            case "sharp":
                heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                return field_values.pop() * heaviside + material_field_from_copy(
                    level_set, field_values, interface
                ) * (1 - heaviside)
            case "arithmetic":
                return field_values.pop() * ls + material_field_from_copy(
                    level_set, field_values, interface
                ) * (1 - ls)
            case "geometric":
                return field_values.pop() ** ls * material_field_from_copy(
                    level_set, field_values, interface
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / field_values.pop()
                    + (1 - ls)
                    / material_field_from_copy(level_set, field_values, interface)
                )
    else:  # Final level set; specify values for both sides of the interface
        match interface:
            case "sharp":
                heaviside = (ls - 0.5 + abs(ls - 0.5)) / 2 / (ls - 0.5)
                return field_values[0] * (1 - heaviside) + field_values[1] * heaviside
            case "arithmetic":
                return field_values[0] * (1 - ls) + field_values[1] * ls
            case "geometric":
                return field_values[0] ** (1 - ls) * field_values[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / field_values[0] + ls / field_values[1])


def material_field(
    level_set: fd.Function | list[fd.Function],
    field_values: list,
    interface: str,
) -> fd.ufl.algebra.Sum | fd.ufl.algebra.Product | fd.ufl.algebra.Division:
    """Generates UFL algebra describing a physical property across the domain.

    Calls `material_field_from_copy` using a copy of the level-set list, preventing the
    original one from being consumed by the function call.

    Args:
      level_set:
        A Firedrake function for the level set (or a list of these)
      field_values:
        A list of physical property values specific to each material
      interface:
        A string specifying how property transitions between materials are calculated

    Returns:
      UFL algebra representing the physical property throughout the domain

    Raises:
      ValueError: Incorrect interface strategy supplied
    """
    if not isinstance(level_set, list):
        level_set = [level_set]

    _impl_interface = ["sharp", "arithmetic", "geometric", "harmonic"]
    if interface not in _impl_interface:
        raise ValueError(f"Interface must be one of {_impl_interface}.")

    return material_field_from_copy(level_set.copy(), field_values, interface)


def entrainment(
    level_set: fd.Function, material_area: float, entrainment_height: float
) -> float:
    """Calculates the entrainment diagnostic.

    Determines the proportion of a material located above a given height.

    Args:
      level_set:
        A Firedrake function for the level-set field
      material_area:
        A float representing the total area occupied by a material
      entrainment_height:
        A float representing the height above which to calculate entrainment

    Returns:
      A float corresponding to the calculated entrainment diagnostic
    """
    mesh_coords = fd.SpatialCoordinate(level_set.ufl_domain())
    target_region = mesh_coords[1] >= entrainment_height
    material_entrained = fd.conditional(level_set < 0.5, 1, 0)

    return (
        fd.assemble(fd.conditional(target_region, material_entrained, 0) * fd.dx)
        / material_area
    )


def min_max_height(
    level_set: fd.Function, epsilon: float | fd.Function, *, side: int, mode: str
) -> float:
    """Calculates the maximum or minimum height of a material interface.

    Args:
      level_set:
        A Firedrake function for the level set field
      epsilon:
        A float or Firedrake function denoting the thickness of the material interface
      side:
        An integer (0 or 1) indicating the level-set side of the target material
      mode:
        A string ("min" or "max") specifying which extremum height is sought

    Returns:
      A float corresponding to the material interface extremum height
    """

    match side:
        case 0:
            comparison = operator.le
        case 1:
            comparison = operator.ge
        case _:
            raise ValueError("'side' must be 0 or 1.")

    match mode:
        case "min":
            arg_finder = np.argmin
            irrelevant_data = np.inf
            mpi_comparison = MPI.MIN
        case "max":
            arg_finder = np.argmax
            irrelevant_data = -np.inf
            mpi_comparison = MPI.MAX
        case _:
            raise ValueError("'mode' must be 'min' or 'max'.")

    mesh = level_set.ufl_domain()
    if not mesh.cartesian:
        raise ValueError("Only Cartesian meshes are currently supported.")

    coords_space = fd.VectorFunctionSpace(mesh, level_set.ufl_element())
    coords = fd.Function(coords_space).interpolate(mesh.coordinates)
    coords_data = coords.dat.data_ro_with_halos
    ls_data = level_set.dat.data_ro_with_halos
    if isinstance(epsilon, float):
        eps_data = epsilon * np.ones_like(ls_data)
    else:
        eps_data = epsilon.dat.data_ro_with_halos

    mask_ls = comparison(ls_data, 0.5)
    if mask_ls.any():
        ind_inside = arg_finder(coords_data[mask_ls, -1])
        height_inside = coords_data[mask_ls, -1][ind_inside]

        if not mask_ls.all():
            hor_coords = coords_data[mask_ls, :-1][ind_inside]
            hor_dist_vec = coords_data[~mask_ls, :-1] - hor_coords
            hor_dist = np.sqrt(np.sum(hor_dist_vec**2, axis=1))

            mask_hor_coords = hor_dist < eps_data[~mask_ls]

            if mask_hor_coords.any():
                ind_outside = abs(
                    coords_data[~mask_ls, -1][mask_hor_coords] - height_inside
                ).argmin()
                height_outside = coords_data[~mask_ls, -1][mask_hor_coords][ind_outside]

                ls_inside = ls_data[mask_ls][ind_inside]
                eps_inside = eps_data[mask_ls][ind_inside]
                sdls_inside = eps_inside * np.log(ls_inside / (1 - ls_inside))

                ls_outside = ls_data[~mask_ls][mask_hor_coords][ind_outside]
                eps_outside = eps_data[~mask_ls][mask_hor_coords][ind_outside]
                sdls_outside = eps_outside * np.log(ls_outside / (1 - ls_outside))

                sdls_dist = sdls_outside / (sdls_outside - sdls_inside)
                height = sdls_dist * height_inside + (1 - sdls_dist) * height_outside
            else:
                height = height_inside

        else:
            height = height_inside

    else:
        height = irrelevant_data

    height_global = level_set.comm.allreduce(height, mpi_comparison)

    return height_global
