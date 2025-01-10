from gadopt import *
from explicit_free_surface import ExplicitFreeSurfaceModel
from test_free_surface import run_benchmark


class ImplicitFreeSurfaceModel(ExplicitFreeSurfaceModel):
    # Test case from Section 3.1.1 of `An implicit free surface algorithm
    # for geodynamical simulations', Kramer et al 2012.

    name = "implicit"
    bottom_free_surface = False
    direct = True
    iterative = True

    def __init__(self, dt_factor, iterative_2d=False, **kwargs):
        self.solver_parameters = 'iterative' if iterative_2d else 'direct'
        super().__init__(dt_factor, **kwargs)

    def setup_function_space(self):
        self.Z = MixedFunctionSpace([self.V, self.W, self.W])  # Mixed function space for velocity, pressure and eta.

    def setup_variables(self):
        # Function to store the solutions:
        self.z = Function(self.Z)  # a field over the mixed function space Z.
        self.stokes_vars = self.z.subfunctions

        # Next rename for output:
        self.stokes_vars[0].rename("Velocity")
        self.stokes_vars[1].rename("Pressure")
        self.stokes_vars[2].rename("eta")

    def initialise_free_surfaces(self):
        self.F0 = Constant(1000 / self.L0)  # initial free surface amplitude (dimensionless)
        self.stokes_vars[2].interpolate(self.F0 * cos(self.kk * self.X[0]))  # Initial free surface condition
        self.eta_analytical = Function(self.stokes_vars[2], name="eta analytical")

    def setup_bcs(self):
        # No normal flow except on the free surface
        self.stokes_bcs = {
            self.boundary.top: {'free_surface': {}},  # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
            self.boundary.bottom: {'un': 0},
            self.boundary.left: {'un': 0},
            self.boundary.right: {'un': 0},
        }

    def setup_solver(self):
        self.stokes_solver = StokesSolver(self.z, self.T, self.approximation, bcs=self.stokes_bcs,
                                          free_surface_dt=self.dt, solver_parameters=self.solver_parameters, nullspace=self.Z_nullspace,
                                          transpose_nullspace=self.Z_nullspace, near_nullspace=self.Z_near_nullspace)

    def calculate_error(self):
        local_error = assemble(pow(self.stokes_vars[2]-self.eta_analytical, 2)*self.ds(self.boundary.top))
        self.error += local_error*self.dt

    def write_file(self):
        self.output_file.write(self.stokes_vars[0], self.stokes_vars[1], self.stokes_vars[2], self.eta_analytical)

    def advance_timestep(self):
        # Solve Stokes sytem:
        self.stokes_solver.solve()


if __name__ == "__main__":
    run_benchmark(ImplicitFreeSurfaceModel)
