from test_free_surface import run_benchmark

from gadopt import *
from gadopt.equations import Equation
from gadopt.free_surface_equation import free_surface_term, mass_term


class ExplicitFreeSurfaceModel:
    # Test case from Section 3.1.1 of `An implicit free surface algorithm
    # for geodynamical simulations', Kramer et al 2012.

    name = "explicit"
    bottom_free_surface = False
    direct = True
    iterative = False

    def __init__(self, dt_factor, nx=80, do_write=False, cartesian=True, **kwargs):
        self.nx = nx
        self.do_write = do_write
        self.setup_mesh()
        self.mesh.cartesian = cartesian

        # Set up function spaces - currently using the bilinear Q2Q1 element pair:
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)  # Velocity function space (vector)
        self.W = FunctionSpace(self.mesh, "CG", 1)  # Pressure function space (scalar)
        self.Q = FunctionSpace(self.mesh, "CG", 2)  # Temperature function space (scalar)
        self.setup_function_space()

        # Output function space information:
        log("Number of Velocity DOF:", self.V.dim())
        log("Number of Pressure DOF:", self.W.dim())
        log("Number of Velocity and Pressure DOF:", self.V.dim()+self.W.dim())

        self.setup_variables()

        self.initialise_wavenumber()

        self.X = SpatialCoordinate(self.mesh)

        self.initialise_temperature()

        self.initialise_approximation()
        self.rho0 = self.approximation.rho  # This defaults to rho0 = 1 (dimensionless)
        self.g = self.approximation.g  # This defaults to g = 1 (dimensionless)

        self.initialise_free_surfaces()

        # timestepping
        self.mu = Constant(1)  # Viscosity (dimensionless)
        self.tau0 = Constant(2 * self.kk * self.mu / (self.rho0 * self.g))  # Characteristic time scale (dimensionless)
        log("tau0", self.tau0)

        self.dt = Constant(dt_factor*self.tau0)  # timestep (dimensionless)
        log("dt (dimensionless)", self.dt)

        self.time = Constant(0.0)
        self.max_timesteps = round(10*self.tau0/self.dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
        log("max_timesteps", self.max_timesteps)

        self.setup_bcs()

        self.setup_nullspaces()

        self.setup_solver()

        self.error = 0

        # analytical function
        self.update_analytical_free_surfaces()
        # Create output file and select output_frequency:
        if do_write:
            # Write output files in VTK format:
            self.dump_period = 1
            log("dump_period ", self.dump_period)
            self.setup_output_file()
            self.write_file()

    def setup_mesh(self):
        # Set up geometry:
        self.D = 3e6  # Depth of domain in m
        self.L = self.D  # Length of the domain in m
        self.L0 = self.D  # characteristic length scale for scaling the equations
        ny = self.nx
        self.left_id, self.right_id, self.bottom_id, self.top_id = 1, 2, 3, 4  # Boundary IDs
        self.mesh = RectangleMesh(self.nx, ny, self.L/self.L0, self.D/self.L0)  # Rectangle mesh generated via firedrake
        self.ds = ds  # Need this so that cylindrical case can overload this later with CombinedSurfaceMeasure

    def setup_function_space(self):
        self.Z = MixedFunctionSpace([self.V, self.W])  # Mixed function space for velocity and pressure.

    def setup_variables(self):
        # Function to store the solutions:
        self.z = Function(self.Z)  # a field over the mixed function space Z.
        self.stokes_vars = self.z.subfunctions
        self.eta = Function(self.Z[1], name="eta")  # Define a free surface function

        # Next rename for output:
        self.stokes_vars[0].rename("Velocity")
        self.stokes_vars[1].rename("Pressure")

    def initialise_wavenumber(self):
        lam_dimensional = self.D/2  # wavelength of load in m
        self.lam = lam_dimensional/self.L0  # dimensionless lambda
        self.kk = Constant(2 * pi / self.lam)  # wavenumber (dimensionless)

    def initialise_temperature(self):
        self.T = Function(self.Q, name="Temperature").assign(0)  # Setup a dummy function for temperature

    def initialise_approximation(self):
        Ra = Constant(0)  # Rayleigh number, here we set this to zero as there are no bouyancy terms
        self.approximation = BoussinesqApproximation(Ra)

    def initialise_free_surfaces(self):
        self.F0 = Constant(1000 / self.L0)  # initial free surface amplitude (dimensionless)
        self.eta.interpolate(self.F0 * cos(self.kk * self.X[0]))  # Initial free surface condition
        self.eta_analytical = Function(self.eta, name="eta analytical")

    def setup_bcs(self):
        # No normal flow except on the free surface
        self.stokes_bcs = {
            self.top_id: {'normal_stress': self.rho0 * self.g * self.eta},  # Apply stress on free surface
            self.bottom_id: {'un': 0},
            self.left_id: {'un': 0},
            self.right_id: {'un': 0},
        }

    def setup_nullspaces(self):
        # Nullspaces and near-nullspaces:
        self.Z_nullspace = None  # Default: don't add nullspace for now
        self.Z_near_nullspace = None  # Default: don't add nullspace for now

    def setup_solver(self):
        # Set up the stokes solver
        self.stokes_solver = StokesSolver(self.z, self.T, self.approximation, bcs=self.stokes_bcs)

        eq_attrs = {
            "boundary_id": self.top_id,
            "buoyancy_scale": 1,
            "u": self.stokes_vars[0],
        }

        # Setup remaining free surface parameters needed for explicit coupling
        eta_eq = Equation(
            TestFunction(self.W),
            self.W,
            free_surface_term,
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )  # Initialise the separate free surface equation for explicit coupling
        # Apply strong homogenous boundary to interior DOFs to prevent a singular matrix when only integrating the free surface equation over the top surface.
        eta_strong_bcs = [InteriorBC(self.W, 0., self.top_id)]

        # Set up a timestepper for the free surface, here we use a first order backward Euler method following Kramer et al. 2012
        self.eta_timestepper = BackwardEuler(
            eta_eq, self.eta, self.dt, strong_bcs=eta_strong_bcs
        )

    def update_analytical_free_surfaces(self):
        # Equation A.4 from Kramer et al., 2012. Here we have a simplified form assuming that the relaxation time scale,
        # tau = tau0 (see Equation A.7) which is valid for wavelengths << depth (e.g. see Table 2 from Kramer et al 2012).
        self.eta_analytical.interpolate(exp(-self.time/self.tau0)*self.F0 * cos(self.kk * self.X[0]))

    def calculate_error(self):
        local_error = assemble(pow(self.eta-self.eta_analytical, 2)*self.ds(self.top_id))
        self.error += local_error*self.dt

    def calculate_final_error(self):
        self.final_error = pow(self.error, 0.5)/self.L0

    def setup_output_file(self):
        self.output_file = File(f"{self.name}_freesurface_D{float(self.D/self.L0)}_mu{float(self.mu)}_nx{self.nx}_dt{float(self.dt/self.tau0)}tau.pvd")

    def write_file(self):
        self.output_file.write(self.stokes_vars[0], self.stokes_vars[1], self.eta, self.eta_analytical)

    def advance_timestep(self):
        # Solve Stokes sytem:
        self.stokes_solver.solve()
        self.eta_timestepper.advance()

    def run_simulation(self):
        # Now perform the time loop:
        for timestep in range(1, self.max_timesteps+1):
            self.advance_timestep()

            self.time.assign(self.time + self.dt)

            # Calculate error
            self.update_analytical_free_surfaces()
            self.calculate_error()

            # Write output:
            if self.do_write:
                if timestep % self.dump_period == 0:
                    log("timestep", timestep)
                    log("time", float(self.time))
                    self.write_file()

        self.calculate_final_error()


if __name__ == "__main__":
    run_benchmark(ExplicitFreeSurfaceModel)
