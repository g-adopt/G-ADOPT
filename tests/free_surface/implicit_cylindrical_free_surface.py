from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
from implicit_free_surface import ImplicitFreeSurfaceModel
from test_free_surface import run_benchmark

# N.b. the analytical solution in this test is a guess that the initial free
# surface topography on the cylinder (a cosine) should decay exponentially in
# a similar way to the cartesian cases. This seems to hold true for the set of
# wavenumbers/resolution used here.


class CylindricalImplicitFreeSurfaceModel(ImplicitFreeSurfaceModel):
    # Free surface relaxation test in a cylindrical domain.

    name = "implicit-cylindrical"
    bottom_free_surface = False
    direct = False
    iterative = True

    def __init__(self, dt_factor, **kwargs):
        super().__init__(dt_factor, cartesian=False, **kwargs)

    def setup_mesh(self):
        # Set up geometry:
        self.rmin, self.rmax, self.ncells, self.nlayers = 1.22, 2.22, 512, 64
        # Construct a circle mesh and then extrude into a cylinder:
        mesh1d = CircleManifoldMesh(self.ncells, radius=self.rmin, degree=2)
        self.mesh = ExtrudedMesh(mesh1d, layers=self.nlayers, extrusion_type='radial')
        self.boundary = get_boundary_ids(self.mesh)
        self.ds = CombinedSurfaceMeasure(self.mesh, degree=6)

    def initialise_wavenumber(self):
        self.number_of_lam = 4*round(2 * pi * self.rmax / (self.rmax - self.rmin))  # number of waves
        lam = (2*pi*self.rmax) / self.number_of_lam  # wavelength (dimensionless)
        self.kk = 2*pi / lam  # wavenumber (dimensionless)

    def initialise_free_surfaces(self):
        self.L0 = 3e6
        self.F0 = Constant(1000 / self.L0)  # initial free surface amplitude (dimensionless)
        self.stokes_vars[2].interpolate(self.F0 * cos(self.number_of_lam * atan2(self.X[1], self.X[0])))  # Initial free surface condition
        self.eta_analytical = Function(self.stokes_vars[2], name="eta analytical")

    def setup_bcs(self):
        self.stokes_bcs = {
            self.boundary.top: {'free_surface': {}},  # Free surface boundary conditions are applied automatically in stokes_integrators and momentum_equation for implicit free surface coupling
            self.boundary.bottom: {'un': 0}
        }

    def setup_nullspaces(self):
        # Nullspaces and near-nullspaces:
        self.Z_nullspace = create_stokes_nullspace(self.Z, closed=False, rotational=True)
        self.Z_near_nullspace = create_stokes_nullspace(self.Z, closed=False, rotational=True, translations=[0, 1])

    def update_analytical_free_surfaces(self):
        # This is a guess of an analytical free surface solution based on exponential decay that seems to hold for
        # particular wavenumber / grid resolutions chosen here!
        self.eta_analytical.interpolate(exp(-self.time/self.tau0)*self.F0 * cos(self.number_of_lam * atan2(self.X[1], self.X[0])))

    def setup_output_file(self):
        self.output_file = File(f"{self.name}_rmax{self.rmax}_rmin{self.rmin}_mu{float(self.mu)}_ncells{self.ncells}_nlay{self.nlayers}_dt{float(self.dt/self.tau0)}tau.pvd")


if __name__ == "__main__":
    run_benchmark(CylindricalImplicitFreeSurfaceModel)
