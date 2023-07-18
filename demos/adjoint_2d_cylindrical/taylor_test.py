# from gadopt import *
import numpy as np
# from firedrake_adjoint import *

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    # "snes_monitor": ':./newton.txt',
    "ksp_type": "preonly",
    "pc_tyoe": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # "snes_view": none,
    "snes_converged_reason": None,
    "fiedsplit_0": {
        "ksp_converged_reason": None,
    },
    "fiedsplit_1": {
        "ksp_converged_reason": None,
    }}

# Set up geometry:
rmin, rmax, ncells, nlayers = 1.22, 2.22, 512, 128
rmax_earth = 6370  # Radius of Earth [km]
rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
r_410_earth = rmax_earth - 410  # 410 radius [km]
r_660_earth = rmax_earth - 660  # 660 raidus [km]
r_410 = rmax - (rmax_earth - r_410_earth)/(rmax_earth - rmin_earth)
r_660 = rmax - (rmax_earth - r_660_earth)/(rmax_earth - rmin_earth)

with CheckpointFile('Checkpoint_State.h5', mode='r') as chckpoint:
    mesh = chckpoint.load_mesh("firedrake_default_extruded")

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Test functions
q = TestFunction(Q)
w = TestFunction(W)

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
T = Function(Q, name='Temperature')

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)

Ra = Constant(1e7)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# Define time stepping parameters:
max_timesteps = 200
delta_t = Constant(5e-6)  # Initial time-step

# Without a restart to continue from, our initial guess is the final state of the forward run
# We need to project the state from Q2 into Q1
Tic = Function(W, name="Initial Temperature")
Taverage = Function(W, name="TemperatureAverage")
with CheckpointFile("Checkpoint_State.h5", "r") as f:
    Tic.project(
        f.load_function(
            mesh,
            "Temperature",
            idx=max_timesteps-1))
    Tic.project(
        f.load_function(
            mesh,
            "AverageTemperature",
            idx=0))

# building linear depth-dependent viscosity
mu_lin = 2.0


# A step function designed to design viscosity jumps
def step_func(center, mag, increasing=True, sharpness=50):
    """ r: Earth's radius, center:center of jump,
        increasing: if True jump happens with increasing r
        sharpness: how shar should the jump be
    """
    return mag * (0.5 * (1 + tanh((1 if increasing else -1)*(r-center)*sharpness)))


# assemble the depth dependence
for line, step in zip(
        [5.*(rmax-r), 1., 1.],
        [step_func(r_660, 30, False),
         step_func(r_410, 10, False),
         step_func(2.2, 10, True)]):
    mu_lin += line*step

# adding the temperature dependence of visc
mu_lin *= exp(- ln(Constant(80)) * (T))

# viscosity expression in terms of u and p
eps = sym(grad(u))
epsii = sqrt(0.5*inner(eps, eps))
sigma_y = 1e4 + 2.0e5*(rmax-r)
mu_plast = 0.1 + (sigma_y / epsii)
mu_eff = 2 * (mu_lin * mu_plast)/(mu_lin + mu_plast)
mu = conditional(mu_eff > 0.4, mu_eff, 0.4)

# viscosity function
mu_function = Function(W, name="Viscosity")

# radial average temperature function
Taverage = Function(W, name="AverageTemperature")

# Calculate the layer average of the initial state
averager = LayerAveraging(
    mesh,
    np.linspace(rmin, rmax, nlayers*2),
    cartesian=False,
    quad_degree=6)

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Create output file and select output_frequency:
output_file = File("vtk-files/output.pvd")
dump_period = 10
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:

temp_bcs = {
    "bottom": {'T': 1.0},
    "top": {'T': 0.0},
}
stokes_bcs = {
    "bottom": {'un': 0},
    "top": {'un': 0},
}
temp_bcs_q1 = [
    DirichletBC(W, 0.0, "top"),
    DirichletBC(W, 1.0, "bottom"),
]

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(
    z, T, approximation,
    mu=mu,
    bcs=stokes_bcs,
    cartesian=False,
    nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters=newton_stokes_solver_parameters)

# Define a simple problem to apply the imposed boundary condition to the IC
T_ = Function(Tic.function_space())
bc_problem = LinearVariationalProblem(
    w * TrialFunction(W) * dx,
    w * T_ * dx,
    Tic,
    bcs=temp_bcs_q1,
)
bc_solver = LinearVariationalSolver(bc_problem)

# Project the initial condition from Q1 to Q
ic_projection_problem = LinearVariationalProblem(
    q * TrialFunction(Q) * dx,
    q * Tic * dx,
    energy_solver.solution_old,
    bcs=energy_solver.strong_bcs,
)
ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

# Control variable for optimisation
control = Control(Tic)

# Apply the boundary condition to the control
# and obtain the initial condition
T_.assign(Tic)
bc_solver.solve()
ic_projection_solver.solve()
T.assign(energy_solver.solution_old)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(Taverage, name="AverageTemperature", idx=0)

# splitting and renaming the functions for visualisation
u_, p_ = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")


# Now perform the time loop:
for timestep in range(0, max_timesteps):
    # Solve Stokes sytem:
    stokes_solver.solve()

    # TODO: Load the velocity
    # checkpoint_file.save_function(u_, idx=timestep)

    # TOOD: Compute misfit for velocity

    # Temperature system:
    energy_solver.solve()

checkpoint_file.close()


# TODO Load average temperature Observed

# TODO Define the misfit

# TODO Calculate gradient (reduced_functional)

# TODO Taylor Test
