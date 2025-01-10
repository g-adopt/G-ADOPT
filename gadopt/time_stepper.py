r"""This module provides several classes to perform integration of time-dependent
equations. Users choose if they require an explicit or implicit time integrator, and
they instantiate one of the implemented algorithm class, for example, `ERKEuler`, by
providing relevant parameters defined in the parent class (i.e. `ERKGeneric` or
`DIRKGeneric`). Then, they call the `advance` method to request a solver update.

"""

import operator
from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable

import firedrake as fd
import numpy as np

from .equations import Equation
from .utility import ensure_constant


class TimeIntegratorBase(ABC):
    """Defines the API for all time integrators."""

    @abstractmethod
    def advance(self, update_forcings: Callable | None = None, t: float | None = None):
        """Advances equations for one time step.

        Args:
          update_forcings:
            Callable updating any time-dependent equation forcings
          t:
            Current simulation time

        """

    @abstractmethod
    def initialise(self, init_solution):
        """Initialises the time integrator.

        Args:
          init_solution: Firedrake function representing the initial solution.

        """


class TimeIntegrator(TimeIntegratorBase):
    """Time integrator object that marches a single equation.

    Args:
      equation:
        G-ADOPT equation to integrate
      solution:
        Firedrake function representing the equation's solution
      dt:
        Integration time step
      solution_old:
        Firedrake function representing the equation's solution at the previous timestep
      solver_parameters:
        Dictionary of solver parameters provided to PETSc
      strong_bcs:
        List of Firedrake Dirichlet boundary conditions

    """

    def __init__(
        self,
        equation: Equation,
        solution: fd.Function,
        dt: fd.Constant | float,
        /,
        *,
        solution_old: fd.Function | None = None,
        solver_parameters: dict[str, str | Number] = {},
        strong_bcs: list[fd.DirichletBC] = [],
    ) -> None:
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        self.solution = solution
        self.dt = float(dt)
        self.dt_const = ensure_constant(dt)
        self.solution_old = solution_old or fd.Function(
            solution, name="Old" + solution.name()
        )
        self.solver_parameters = solver_parameters
        self.strong_bcs = strong_bcs

        self.hom_bcs = [
            bci.__class__(bci.function_space(), 0, bci.sub_domain)
            for bci in self.strong_bcs
        ]

        # unique identifier used in solver
        self.name = "-".join(
            [self.__class__.__name__, self.equation.__class__.__name__]
        )


class RungeKuttaTimeIntegrator(TimeIntegrator):
    """Abstract base class for all Runge-Kutta time integrators"""

    @abstractmethod
    def get_final_solution(self):
        """Evaluates the final solution"""

    @abstractmethod
    def solve_stage(
        self,
        i_stage: int,
        update_forcings: Callable | None = None,
        t: float | None = None,
    ):
        """Solves a single stage of step from t to t+dt.

        All functions that the equation depends on must be at right state corresponding
        to each sub-step.
        """

    def advance(
        self, update_forcings: Callable | None = None, t: float | None = None
    ) -> None:
        """Advances equations for one time step."""
        if not self._initialised:
            self.initialise(self.solution)

        for i in range(self.n_stages):
            self.solve_stage(i, update_forcings, t)

        self.get_final_solution()


class ERKGeneric(RungeKuttaTimeIntegrator):
    """Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form. All terms in the equation are treated explicitly.

    Args:
      equation:
        G-ADOPT equation to solve
      solution:
        Firedrake function reperesenting the equation's solution
      dt:
        Integration time step
      solution_old:
        Firedrake function representing the equation's solution at the previous timestep
      solver_parameters:
        Dictionary of solver parameters provided to PETSc
      strong_bcs:
        List of Firedrake Dirichlet boundary conditions

    """

    def __init__(
        self,
        equation: Equation,
        solution: fd.Function,
        dt: fd.Constant | float,
        /,
        **kwargs,
    ) -> None:
        super(ERKGeneric, self).__init__(equation, solution, dt, **kwargs)

        self._initialised = False

        V = solution.function_space()
        assert V == equation.trial_space

        self.tendency = []
        for i in range(self.n_stages):
            k = fd.Function(V, name="tendency{:}".format(i))
            self.tendency.append(k)

        # fully explicit evaluation
        self.a_rk = self.equation.mass(fd.TrialFunction(V))
        self.l_rk = self.dt_const * self.equation.residual(self.solution)

        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        if self._nontrivial:
            self.sol_expressions = []
            for i_stage in range(self.n_stages):
                tendency = self.tendency[:i_stage]
                a = self.a[i_stage][:i_stage]
                self.sol_expressions.append(sum(map(operator.mul, tendency, a)))

            self.final_sol_expr = sum(map(operator.mul, self.tendency, self.b))

        self.update_solver()

    def update_solver(self) -> None:
        """Create solver objects"""
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = fd.LinearVariationalProblem(
                    self.a_rk, self.l_rk, self.tendency[i], bcs=self.hom_bcs
                )
                solver = fd.LinearVariationalSolver(
                    prob,
                    options_prefix=self.name + "_k{:}".format(i),
                    solver_parameters=self.solver_parameters,
                )
                self.solver.append(solver)

    def initialise(self, solution) -> None:
        self.solution_old.assign(solution)
        self._initialised = True

    def update_solution(self, i_stage) -> None:
        """Computes the solution of the i-th stage

        Tendencies must have been evaluated first.

        """
        self.solution.assign(self.solution_old)
        if self._nontrivial and i_stage > 0:
            self.solution += self.sol_expressions[i_stage]

    def solve_tendency(self, i_stage, update_forcings, t) -> None:
        """Evaluates the tendency of i-th stage"""
        if self._nontrivial:
            if update_forcings is not None and t is not None:
                update_forcings(t + self.c[i_stage] * self.dt)
            elif update_forcings is not None:
                update_forcings()

            self.solver[i_stage].solve()

    def get_final_solution(self) -> None:
        self.solution.assign(self.solution_old)
        if self._nontrivial:
            self.solution += self.final_sol_expr

        self.solution_old.assign(self.solution)

    def solve_stage(self, i_stage, update_forcings, t) -> None:
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, update_forcings, t)


class DIRKGeneric(RungeKuttaTimeIntegrator):
    """Generic implementation of Diagonally Implicit Runge Kutta schemes.

    All derived classes must define the Butcher tableau coefficients :attr:`a`,
    :attr:`b`, :attr:`c`.

    Args:
      equation:
        G-ADOPT equation to solve
      solution:
        Firedrake function reperesenting the equation's solution
      dt:
        Integration time step
      solution_old:
        Firedrake function representing the equation's solution at the previous timestep
      solver_parameters:
        Dictionary of solver parameters provided to PETSc
      strong_bcs:
        List of Firedrake Dirichlet boundary conditions
    """

    def __init__(
        self,
        equation: Equation,
        solution: fd.Function,
        dt: fd.Constant | float,
        /,
        **kwargs,
    ) -> None:
        super(DIRKGeneric, self).__init__(equation, solution, dt, **kwargs)

        self.solver_parameters.setdefault("snes_type", "newtonls")
        self._initialised = False

        fs = solution.function_space()
        assert fs == equation.trial_space

        mixed_space = len(fs) > 1

        # Allocate tendency fields
        self.k = []
        for i in range(self.n_stages):
            fname = "{:}_k{:}".format(self.name, i)
            self.k.append(fd.Function(fs, name=fname))

        # construct variational problems
        self.F = []
        if not mixed_space:
            for i in range(self.n_stages):
                for j in range(i + 1):
                    if j == 0:
                        u = self.solution_old + self.a[i][j] * self.dt_const * self.k[j]
                    else:
                        u += self.a[i][j] * self.dt_const * self.k[j]

                self.F.append(self.equation.mass(self.k[i]) - self.equation.residual(u))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in range(self.n_stages):
                for j in range(i + 1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(
                            fd.split(self.solution_old), fd.split(self.k[j])
                        ):
                            u.append(s + self.a[i][j] * self.dt_const * k)
                    else:
                        for l, k in enumerate(fd.split(self.k[j])):
                            u[l] += self.a[i][j] * self.dt_const * k

                self.F.append(self.equation.mass(self.k[i]) - self.equation.residual())

        self.update_solver()

        # construct expressions for stage solutions
        self.sol_expressions = []
        for i_stage in range(self.n_stages):
            k = self.k[: i_stage + 1]
            a = self.a[i_stage][: i_stage + 1]
            self.sol_expressions.append(sum(map(operator.mul, k, self.dt_const * a)))

        self.final_sol_expr = self.solution_old + sum(
            map(operator.mul, self.k, self.dt_const * self.b)
        )

    def update_solver(self) -> None:
        """Create solver objects"""
        self.solver = []
        for i in range(self.n_stages):
            p = fd.NonlinearVariationalProblem(self.F[i], self.k[i], bcs=self.hom_bcs)
            sname = "{:}_stage{:}_".format(self.name, i)
            self.solver.append(
                fd.NonlinearVariationalSolver(
                    p, solver_parameters=self.solver_parameters, options_prefix=sname
                )
            )

    def initialise(self, init_cond) -> None:
        self.solution_old.assign(init_cond)
        self._initialised = True

    def update_solution(self, i_stage) -> None:
        """Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.

        """
        self.solution.assign(self.solution_old + self.sol_expressions[i_stage])

    def solve_tendency(self, i_stage, update_forcings, t) -> None:
        """Evaluates the tendency of i-th stage"""
        if i_stage == 0:
            # NOTE: solution may have changed in coupled system
            for bci in self.strong_bcs:
                bci.apply(self.solution)
            self.solution_old.assign(self.solution)

        if not self._initialised:
            raise ValueError("Time integrator {:} is not initialised".format(self.name))

        if update_forcings is not None and t is not None:
            update_forcings(t + self.c[i_stage] * self.dt)
        elif update_forcings is not None:
            update_forcings()

        self.solver[i_stage].solve()

    def get_final_solution(self) -> None:
        self.solution.assign(self.final_sol_expr)

    def solve_stage(self, i_stage, update_forcings, t) -> None:
        self.solve_tendency(i_stage, update_forcings, t)
        self.update_solution(i_stage)


CFL_UNCONDITIONALLY_STABLE = -1


class AbstractRKScheme(ABC):
    """Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """

    @property
    @abstractmethod
    def a(self):
        """Runge-Kutta matrix :math:`a_{i,j}` of the Butcher tableau"""

    @property
    @abstractmethod
    def b(self):
        """weights :math:`b_{i}` of the Butcher tableau"""

    @property
    @abstractmethod
    def c(self):
        """nodes :math:`c_{i}` of the Butcher tableau"""

    @property
    @abstractmethod
    def cfl_coeff(self):
        """CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.

        """

    def __init__(self) -> None:
        super().__init__()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), "Butcher tableau must be lower diagonal"
        assert np.allclose(
            np.sum(self.a, axis=1), self.c
        ), "Inconsistent Butcher tableau: Row sum of a is not c"

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()


def shu_osher_butcher(α_or_λ, β_or_μ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produces a Butcher tableau from a Shu-Osher form.

    Generate arrays composing the Butcher tableau of a Runge-Kutta method from the
    coefficient arrays of the equivalent, original or modified, Shu-Osher form.
    Code adapted from RK-Opt written in MATLAB by David Ketcheson.
    See also Ketcheson, Macdonald, and Gottlieb (2009).

    Function Args:
    α_or_λ : array_like, shape (n + 1, n)
    β_or_μ : array_like, shape (n + 1, n)
    """

    X = np.identity(α_or_λ.shape[1]) - α_or_λ[:-1]
    A = np.linalg.solve(X, β_or_μ[:-1])
    b = np.transpose(β_or_μ[-1] + np.dot(α_or_λ[-1], A))
    c = np.sum(A, axis=1)

    return A, b, c


class ForwardEulerAbstract(AbstractRKScheme):
    """Forward Euler method"""

    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ERKLSPUM2Abstract(AbstractRKScheme):
    """ERKLSPUM2, 3-stage, 2nd order Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [[0, 0, 0], [5.0 / 6.0, 0, 0], [11.0 / 24.0, 11.0 / 24.0, 0]]
    b = [24.0 / 55.0, 1.0 / 5.0, 4.0 / 11.0]
    c = [0, 5.0 / 6.0, 11.0 / 12.0]
    cfl_coeff = 1.2


class ERKLPUM2Abstract(AbstractRKScheme):
    """ERKLPUM2, 3-stage, 2nd order Explicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014).
    Optimized strong stability preserving IMEX Runge-Kutta methods.
    Journal of Computational and Applied Mathematics 272(2014) 116-140.
    http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [[0, 0, 0], [1.0 / 2.0, 0, 0], [1.0 / 2.0, 1.0 / 2.0, 0]]
    b = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    c = [0, 1.0 / 2.0, 1.0]
    cfl_coeff = 2.0


class ERKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0], [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33Abstract(AbstractRKScheme):
    r"""3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau

    .. math::
        \begin{array}{c|ccc}
            0 &                 \\
            1 & 1               \\
          1/2 & 1/4 & 1/4 &     \\ \hline
              & 1/6 & 1/6 & 2/3
        \end{array}

    CFL coefficient is 1.0
    """

    a = [[0, 0, 0], [1.0, 0, 0], [0.25, 0.25, 0]]
    b = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class eSSPRKs3p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [[0.0, 0.0, 0.0], [2 / 3, 0.0, 0.0], [2 / 9, 4 / 9, 0.0]]
    b = [0.25, 0.1875, 0.5625]
    c = [0, 2 / 3, 2 / 3]
    cfl_coeff = 3 / 4


class eSSPRKs4p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0],
        [11 / 20, 0.0, 0.0, 0.0],
        [11 / 32, 11 / 32, 0.0, 0.0],
        [55 / 288, 55 / 288, 11 / 36, 0.0],
    ]
    b = [0.24517906, 0.13774105, 0.22038567, 0.39669421]
    c = [0, 11 / 20, 11 / 16, 11 / 16]
    cfl_coeff = 20 / 11


class eSSPRKs5p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.37949799, 0.0, 0.0, 0.0, 0.0],
        [0.35866028, 0.35866028, 0.0, 0.0, 0.0],
        [0.23456423, 0.23456423, 0.24819211, 0.0, 0.0],
        [0.15340527, 0.15340527, 0.16231792, 0.24819211, 0.0],
    ]
    b = [0.20992362, 0.1975535, 0.1217419, 0.18614938, 0.28463159]
    c = [0.0, 0.37949799, 0.71732056, 0.71732057, 0.71732057]
    cfl_coeff = 2.63506005


class eSSPRKs6p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.28422072, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.28422072, 0.28422072, 0.0, 0.0, 0.0, 0.0],
        [0.23301578, 0.23301578, 0.23301578, 0.0, 0.0, 0.0],
        [0.16684082, 0.16532461, 0.16532461, 0.20165449, 0.0, 0.0],
        [0.21178186, 0.102324, 0.10202706, 0.12444738, 0.17540162, 0.0],
    ]
    b = [0.21181784, 0.10241434, 0.10198818, 0.12438557, 0.17531451, 0.28407956]
    c = [0.0, 0.28422072, 0.56844144, 0.69904734, 0.69914453, 0.71598192]
    cfl_coeff = 3.51839231


class eSSPRKs7p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23333473, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23333473, 0.23333473, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.23144338, 0.23144338, 0.23144338, 0.0, 0.0, 0.0, 0.0],
        [0.17322863, 0.17322863, 0.17322863, 0.17464425, 0.0, 0.0, 0.0],
        [0.13071968, 0.12941249, 0.12941249, 0.13047004, 0.17431545, 0.0, 0.0],
        [0.16655731, 0.16570664, 0.08421603, 0.08490424, 0.11343693, 0.15184412, 0.0],
    ]
    b = [
        0.16655731,
        0.16570664,
        0.08421603,
        0.08490424,
        0.11343693,
        0.15184412,
        0.23333473,
    ]
    c = [0.0, 0.23333473, 0.46666946, 0.69433014, 0.69433014, 0.69433015, 0.76666527]
    cfl_coeff = 4.28568865


class eSSPRKs8p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.19580402, 0.19580402, 0.19580402, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.15369244, 0.15369244, 0.15369244, 0.15369244, 0.0, 0.0, 0.0, 0.0],
        [0.11656615, 0.11656615, 0.11656615, 0.11656615, 0.14850516, 0.0, 0.0, 0.0],
        [
            0.12960593,
            0.09738344,
            0.09738344,
            0.09738344,
            0.12406641,
            0.16358153,
            0.0,
            0.0,
        ],
        [
            0.12970594,
            0.09753214,
            0.09723632,
            0.09723632,
            0.12387897,
            0.16333439,
            0.1955082,
            0.0,
        ],
    ]
    b = [
        0.1462899,
        0.12218849,
        0.12196689,
        0.10127077,
        0.09279782,
        0.1223539,
        0.14645532,
        0.14667691,
    ]
    c = [
        0.0,
        0.19580402,
        0.39160804,
        0.58741206,
        0.61476976,
        0.61476976,
        0.70940419,
        0.90443228,
    ]
    cfl_coeff = 5.10714756


class eSSPRKs9p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.13333333,
            0.13333333,
            0.13333333,
            0.13333333,
            0.13333333,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [0.14166667, 0.1, 0.1, 0.1, 0.1, 0.125, 0.0, 0.0, 0.0],
        [
            0.15,
            0.12222222,
            0.06666667,
            0.06666667,
            0.06666667,
            0.08333333,
            0.11111111,
            0.0,
            0.0,
        ],
        [
            0.15,
            0.12222222,
            0.06666667,
            0.06666667,
            0.06666667,
            0.08333333,
            0.11111111,
            0.16666667,
            0.0,
        ],
    ]
    b = [
        0.15,
        0.12222222,
        0.06666667,
        0.06666667,
        0.06666667,
        0.08333333,
        0.11111111,
        0.16666667,
        0.16666667,
    ]
    c = [
        0.0,
        0.16666667,
        0.33333334,
        0.50000001,
        0.66666668,
        0.66666665,
        0.66666667,
        0.66666667,
        0.83333334,
    ]
    cfl_coeff = 6.0


class eSSPRKs10p3Abstract(AbstractRKScheme):
    """Explicit SSP Runge-Kutta method with nondecreasing abscissas.
    See Isherwood, Grant, and Gottlieb (2018)."""

    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.14737756, 0.14737756, 0.14737756, 0.14737756, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.11790205,
            0.11790205,
            0.11790205,
            0.11790205,
            0.11790205,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.10906732,
            0.10906703,
            0.10906703,
            0.10906703,
            0.10906703,
            0.13633378,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.11862231,
            0.11856848,
            0.08186453,
            0.08186453,
            0.08186453,
            0.10233067,
            0.11062,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.12708168,
            0.12704369,
            0.08137218,
            0.0577812,
            0.0577812,
            0.07222649,
            0.07807723,
            0.10402125,
            0.0,
            0.0,
        ],
        [
            0.1270886,
            0.12705062,
            0.08139469,
            0.05776149,
            0.05776149,
            0.07220186,
            0.07805061,
            0.10398578,
            0.1473273,
            0.0,
        ],
    ]
    b = [
        0.1270886,
        0.12705062,
        0.08139469,
        0.05776149,
        0.05776149,
        0.07220186,
        0.07805061,
        0.10398578,
        0.1473273,
        0.14737756,
    ]
    c = [
        0.0,
        0.14737756,
        0.29475512,
        0.44213268,
        0.58951024,
        0.58951025,
        0.68166922,
        0.69573505,
        0.70538492,
        0.85262244,
    ]
    cfl_coeff = 6.78529356


class BackwardEulerAbstract(AbstractRKScheme):
    """Backward Euler method"""

    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class ImplicitMidpointAbstract(AbstractRKScheme):
    r"""Implicit midpoint method, second order.

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|c}
        0.5 & 0.5 \\ \hline
            & 1.0
        \end{array}

    """

    a = [[0.5]]
    b = [1.0]
    c = [0.5]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class CrankNicolsonAbstract(AbstractRKScheme):
    """
    Crank-Nicolson scheme
    """

    a = [[0.0, 0.0], [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22Abstract(AbstractRKScheme):
    r"""2-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
        \gamma &   \gamma &       0 \\
              1 & 1-\gamma & \gamma \\ \hline
                &       1/2 &     1/2
        \end{array}

    with :math:`\gamma = (2 + \sqrt{2})/2`.

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = (2.0 + np.sqrt(2.0)) / 2.0
    a = [[gamma, 0], [1 - gamma, gamma]]
    b = [1 - gamma, gamma]
    c = [gamma, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK23Abstract(AbstractRKScheme):
    r"""2-stage, 3rd order Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    .. math::
        \begin{array}{c|cc}
          \gamma &    \gamma &       0 \\
        1-\gamma & 1-2\gamma & \gamma \\ \hline
                  &        1/2 &     1/2
        \end{array}

    with :math:`\gamma = (3 + \sqrt{3})/6`.

    From DIRK(2,3,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = (3 + np.sqrt(3)) / 6
    a = [[gamma, 0], [1 - 2 * gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1 - gamma]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK33Abstract(AbstractRKScheme):
    """3-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    gamma = 0.4358665215
    b1 = -3.0 / 2.0 * gamma**2 + 4 * gamma - 1.0 / 4.0
    b2 = 3.0 / 2.0 * gamma**2 - 5 * gamma + 5.0 / 4.0
    a = [[gamma, 0, 0], [(1 - gamma) / 2, gamma, 0], [b1, b2, gamma]]
    b = [b1, b2, gamma]
    c = [gamma, (1 + gamma) / 2, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK43Abstract(AbstractRKScheme):
    """4-stage, 3rd order, L-stable Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)

    Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
    time-dependent partial differential equations. Applied Numerical
    Mathematics, 25:151-167. http://dx.doi.org/10.1137/0732037
    """

    a = [
        [0.5, 0, 0, 0],
        [1.0 / 6.0, 0.5, 0, 0],
        [-0.5, 0.5, 0.5, 0],
        [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5],
    ]
    b = [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5]
    c = [0.5, 2.0 / 3.0, 0.5, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRKLSPUM2Abstract(AbstractRKScheme):
    """DIRKLSPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [
        [2.0 / 11.0, 0, 0],
        [205.0 / 462.0, 2.0 / 11.0, 0],
        [2033.0 / 4620.0, 21.0 / 110.0, 2.0 / 11.0],
    ]
    b = [24.0 / 55.0, 1.0 / 5.0, 4.0 / 11.0]
    c = [2.0 / 11.0, 289.0 / 462.0, 751.0 / 924.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.82


class DIRKLPUM2Abstract(AbstractRKScheme):
    """DIRKLPUM2, 3-stage, 2nd order, L-stable Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """

    a = [
        [2.0 / 11.0, 0, 0],
        [41.0 / 154.0, 2.0 / 11.0, 0],
        [289.0 / 847.0, 42.0 / 121.0, 2.0 / 11.0],
    ]
    b = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    c = [2.0 / 11.0, 69.0 / 154.0, 67.0 / 77.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.09


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ERKEuler(ERKGeneric, ForwardEulerAbstract):
    pass


class SSPRK33(ERKGeneric, SSPRK33Abstract):
    pass


class eSSPRKs3p3(ERKGeneric, eSSPRKs3p3Abstract):
    pass


class eSSPRKs4p3(ERKGeneric, eSSPRKs4p3Abstract):
    pass


class eSSPRKs5p3(ERKGeneric, eSSPRKs5p3Abstract):
    pass


class eSSPRKs6p3(ERKGeneric, eSSPRKs6p3Abstract):
    pass


class eSSPRKs7p3(ERKGeneric, eSSPRKs7p3Abstract):
    pass


class eSSPRKs8p3(ERKGeneric, eSSPRKs8p3Abstract):
    pass


class eSSPRKs9p3(ERKGeneric, eSSPRKs9p3Abstract):
    pass


class eSSPRKs10p3(ERKGeneric, eSSPRKs10p3Abstract):
    pass


class BackwardEuler(DIRKGeneric, BackwardEulerAbstract):
    pass


class ImplicitMidpoint(DIRKGeneric, ImplicitMidpointAbstract):
    pass


class CrankNicolsonRK(DIRKGeneric, CrankNicolsonAbstract):
    pass


class DIRK22(DIRKGeneric, DIRK22Abstract):
    pass


class DIRK23(DIRKGeneric, DIRK23Abstract):
    pass


class DIRK33(DIRKGeneric, DIRK33Abstract):
    pass


class DIRK43(DIRKGeneric, DIRK43Abstract):
    pass


class DIRKLSPUM2(DIRKGeneric, DIRKLSPUM2Abstract):
    pass


class DIRKLPUM2(DIRKGeneric, DIRKLPUM2Abstract):
    pass
