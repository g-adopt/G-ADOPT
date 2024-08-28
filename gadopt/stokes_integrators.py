r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

from numbers import Number
from typing import Optional

import firedrake as fd

from .approximations import BaseApproximation
from .momentum_equation import StokesEquations
from .utility import DEBUG, INFO, depends_on, ensure_constant, log_level, upward_normal, InteriorBC

iterative_stokes_solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-5,
        "ksp_max_it": 1000,
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_mg_levels_pc_type": "sor",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
        "assembled_pc_gamg_coarse_eq_limit": 1000,
        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-4,
        "ksp_max_it": 200,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "ksp",
        "Mp_ksp_ksp_rtol": 1e-5,
        "Mp_ksp_ksp_type": "cg",
        "Mp_ksp_pc_type": "sor",
    }
}
"""Default iterative solver parameters for solution of stokes system.

We configure the Schur complement approach as described in Section of
4.3 of Davies et al. (2022), using PETSc's fieldsplit preconditioner
type, which provides a class of preconditioners for mixed problems
that allows a user to apply different preconditioners to different
blocks of the system.

The fieldsplit_0 entries configure solver options for the velocity
block. The linear systems associated with this matrix are solved using
a combination of the Conjugate Gradient (cg) method and an algebraic
multigrid preconditioner (GAMG).

The fieldsplit_1 entries contain solver options for the Schur
complement solve itself. For preconditioning, we approximate the Schur
complement matrix with a mass matrix scaled by viscosity, with the
viscosity provided through the optional mu keyword argument to Stokes
solver. Since this preconditioner step involves an iterative solve,
the Krylov method used for the Schur complement needs to be of
flexible type, and we use FGMRES by default.

We note that our default solver parameters can be augmented or
adjusted by accessing the solver_parameter dictionary, for example:

```
   stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
   stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
   stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
   stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
   stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2
```

Note:
  G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of Stokes system.

Configured to use LU factorisation, using the MUMPS library.

Note:
  G-ADOPT defaults to direct solvers in 2-D.
"""

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
}
"""Default solver parameters for non-linear systems.

We use a setup based on Newton's method (newtonls) with a secant line
search over the L2-norm of the function.
"""


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: Optional[list[int]] = None,
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z: Firedrake mixed function space associated with the Stokes system
      closed: Whether to include a constant pressure null space
      rotational: Whether to include all rotational modes
      translations: List of translations to include

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    V, W = Z.subfunctions

    if rotational:
        if dim == 2:
            rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(V).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(V).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(V).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    if closed:
        p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = W

    return fd.MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])


class StokesSolver:
    """Solves the Stokes system in place.

    Arguments:
      z: Firedrake function representing mixed Stokes system
      T: Firedrake function representing temperature
      approximation: Approximation describing system of equations
      bcs: Dictionary of identifier-value pairs specifying boundary conditions
      mu: Firedrake function representing dynamic viscosity
      quad_degree: Quadrature degree. Default value is `2p + 1`, where
                   p is the polynomial degree of the trial space
      solver_parameters: Either a dictionary of PETSc solver parameters or a string
                         specifying a default set of parameters defined in G-ADOPT
      J: Firedrake function representing the Jacobian of the system
      constant_jacobian: Whether the Jacobian of the system is constant

    """

    name = "Stokes"

    def __init__(
        self,
        z: fd.Function,
        T: fd.Function,
        approximation: BaseApproximation,
        bcs: dict[int, dict[str, Number]] = {},
        mu: fd.Function | Number = 1,
        quad_degree: int = 6,
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        J: Optional[fd.Function] = None,
        constant_jacobian: bool = False,
        **kwargs,
    ):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.solution_old = None
        self.approximation = approximation
        self.mu = ensure_constant(mu)
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)

        self.solver_kwargs = kwargs
        u, p = fd.split(self.solution)
        self.k = upward_normal(self.Z.mesh())
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'source': self.approximation.buoyancy(p, T) * self.k,
            'rho_continuity': self.approximation.rho_continuity(),
        }

        self.weak_bcs = {}
        self.strong_bcs = []
        for id, bc in bcs.items():
            weak_bc = {}
            for bc_type, value in bc.items():
                if bc_type == 'u':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0), value, id))
                elif bc_type == 'ux':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(0), value, id))
                elif bc_type == 'uy':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(1), value, id))
                elif bc_type == 'uz':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(2), value, id))
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if isinstance(solver_parameters, dict):
            self.solver_parameters = solver_parameters
        else:
            if self.linear:
                self.solver_parameters = {"snes_type": "ksponly"}
            else:
                self.solver_parameters = newton_stokes_solver_parameters.copy()

            if INFO >= log_level:
                self.solver_parameters["snes_monitor"] = None

            if isinstance(solver_parameters, str):
                match solver_parameters:
                    case "direct":
                        self.solver_parameters.update(direct_stokes_solver_parameters)
                    case "iterative":
                        self.solver_parameters.update(
                            iterative_stokes_solver_parameters
                        )
                    case _:
                        raise ValueError(
                            f"Solver type '{solver_parameters}' not implemented."
                        )
            elif self.mesh.topological_dimension() == 2 and self.mesh.cartesian:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)

                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None

        # solver object is set up later to permit editing default solver parameters specified above
        self._solver_setup = False

    def setup_solver(self):
        """Sets up the solver."""
        # mu used in MassInvPC:
        mu_over_rho = self.mu / self.approximation.rho_continuity()
        if self.constant_jacobian:
            z_tri = fd.TrialFunction(self.Z)
            F_stokes_lin = fd.replace(self.F, {self.solution: z_tri})
            a, L = fd.lhs(F_stokes_lin), fd.rhs(F_stokes_lin)
            self.problem = fd.LinearVariationalProblem(a, L, self.solution,
                                                       bcs=self.strong_bcs,
                                                       constant_jacobian=True)
            self.solver = fd.LinearVariationalSolver(self.problem,
                                                     solver_parameters=self.solver_parameters,
                                                     options_prefix=self.name,
                                                     appctx={"mu": mu_over_rho},
                                                     **self.solver_kwargs)
        else:
            self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                          bcs=self.strong_bcs, J=self.J)
            self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                        solver_parameters=self.solver_parameters,
                                                        options_prefix=self.name,
                                                        appctx={"mu": mu_over_rho},
                                                        **self.solver_kwargs)
        self._solver_setup = True

    def solve(self):
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()
        self.solution_old = self.solution.copy(deepcopy=True)
        self.solver.solve()


class BoundaryNormalStressSolver:
    """ A class for calculating surface forces acting on boundary

    Args:
        stokes_solver: gadopt StokesSolver, which provides
            the necessary fields for calculating stress
        subdomain_id: str | int, the subdomain id of a physical boundary
        solver_parameters: Optional, dictionary of parameters for the
            the simple variational problem
    """
    # Direct solve parameters for dynamic topography
    direct_solve_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    # Iterative solve parameters
    iterative_solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_rtol": 1e-5,
        "pc_type": "sor",
    }

    def __init__(self,
                 stokes_solver: StokesSolver,
                 subdomain_id: int | str,
                 solver_parameters: Optional[dict[str, str | Number]] = None):

        # pressure and velocity together with viscosity are needed
        self.u, self.p, *self.eta = stokes_solver.solution.subfunctions
        self.mu = stokes_solver.mu

        # geometry
        self.mesh = stokes_solver.mesh
        self.dim = self.mesh.geometric_dimension()

        # approximation tells us if we need to consider compressible formulation or not
        self.approximation = stokes_solver.approximation

        # physical boundary id
        self.subdomain_id = subdomain_id

        # setting solver parameters
        if solver_parameters is None:
            if self.dim == 3:
                self.solver_parameters = BoundaryNormalStressSolver.iterative_solver_parameters
            else:
                self.solver_parameters = BoundaryNormalStressSolver.direct_solve_parameters
        else:
            self.solver_parameters = solver_parameters

        # when to know the solver
        self._solver_is_made = False

    def solve(self):        # Solve a linear system
        if not self._solver_is_made:
            self.setup_solver()

        # solve for the source
        self.solver.solve()

        # take the average out
        vave = fd.assemble(self.force * self.ds)
        self.force.assign(self.force - vave)

        # re-apply the zero condition everywhere except for the
        self.interior_null_bc.apply(self.force)

        return self.force

    def setup_solver(self):

        # Since p will be alway in lower dimensions than u
        # it makes sense to define the solution in the lower dimension
        # as dynamic topography uses p
        Q = fd.FunctionSpace(self.mesh, self.p.ufl_element())

        self.force = fd.Function(Q, name=f"force_{self.subdomain_id}")

        # normal vector
        n = fd.FacetNormal(self.mesh)

        # test and trial functions
        phi = fd.TestFunction(Q)
        v = fd.TrialFunction(Q)

        # stress, compressible formulation has an additional term
        stress = -self.p * fd.Identity(self.dim) + self.mu * 2 * fd.sym(fd.grad(self.u))

        if self.approximation.compressible:
            stress -= 2/3 * self.mu * self.Identity(self.dim) * fd.div(self.u)

        # surface integral for extruded mesh is different
        # are we dealing with extruded mesh?
        extruded_mesh = self.mesh.extruded

        # choosing surfce integral
        if extruded_mesh and self.subdomain_id in ["top", "bottom"]:
            self.ds = {"top": fd.ds_t, "bottom": fd.ds_b}.get(self.subdomain_id)
        else:
            self.ds = fd.ds(self.subdomain_id)

        a = phi * v * self.ds
        L = - phi * fd.dot(fd.dot(stress, n), n) * self.ds

        # setting up boundary condition, problem and solver
        self.interior_null_bc = InteriorBC(Q, 0., [self.subdomain_id])
        self.problem = fd.LinearVariationalProblem(a, L, self.force,
                                                   bcs=self.interior_null_bc,
                                                   constant_jacobian=True)
        self.solver = fd.LinearVariationalSolver(self.problem, solver_parameters=self.solver_parameters)

        self._solver_is_made = True
