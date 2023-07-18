from firedrake import *
from firedrake.petsc import PETSc
from layer_average import LayerAveraging
from solver_prms import newton_stokes_solver_parameters
from mpi4py import MPI
import numpy as np
from cyl_parameters import (
    dx, ds_t, ds_b, ds_tb,
    rmin, rmax, rmax_earth,
    r_410_earth, r_660_earth,
    r_410, r_660)

# Define logging
log = PETSc.Sys.Print


with CheckpointFile(
        '../spin-up/states/Checkpoint230.h5',
        mode='r') as chckpoint:
    mesh = chckpoint.load_mesh("firedrake_default_extruded") 
    Told = chckpoint.load_function(mesh, 'Temperature')

bottom_id = 'bottom'
top_id = 'top'

# 1d profile for averaging temperature
r1p = np.linspace(rmin, rmax, 256) 
# utility for computing average profile 
ver_ave = LayerAveraging(
    mesh,
    r1p,
    cartesian=False, quad_degree=6)

# normal required for boundary condition
n = FacetNormal(mesh)

# Set up function spaces: Q2Q1:
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 2)
Q1 = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])

# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)

# Temperature Functions
Taverage = Function(Q1, name="Taverage")
Tnew = Function(Q, name="Temperature")
Tnew.assign(Told)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5*Tnew + (1 - 0.5)*Told

# Stokes related constants
# Rayleigh Number
Ra = Constant(1.0e+07)
# unit vector in r dir
k = as_vector((X[0], X[1])) / r
# Fudge factor for interior penalty term used in weak imposition of BCs
C_ip = Constant(100.0)
# Maximum polynomial degree of the _gradient_ of velocity
p_ip = 2
# Initial time-step
delta_t = Constant(5.0e-6)
# Thermal diffusivity
kappa = Constant(1.0)
# viscosity
mu_lin = 2.0


# A step function designed to
# design viscosity jumps
def step_func(r, center, mag, increasing=True, sharpness=50):
    """
        input:
            r: is the radius array
            center: radius of the jum
            increasing: if True, the jump happens towards lower r
                        otherwise jump happens at higher r
            sharpness: how shar should the jump be
    """
    if increasing:
        sign = 1
    else:
        sign = -1
    return mag * (0.5 * (1 + tanh(sign*(r-center)*sharpness)))


# assemble the depth dependence
# for the lower mantle increase we multiply the profile with a
# linear function
for line, step in zip([5.*(rmax-r), 1., 1.],
                      [step_func(r, r_660, 30, False),
                       step_func(r, r_410, 10, False),
                       step_func(r, 2.2, 10, True)]):
    mu_lin += line*step

# viscosity is a function of depth and temperature
# depth parameters
delta_mu_T = Constant(80)  # temperature dependence factor

# adding the temperature dependence of visc
mu_lin *= exp(- ln(delta_mu_T) * (Tnew))

# non-linear rheology
# Strain-rate:
epsilon = 0.5 * sym(grad(u))
# 2nd invariant
epsii = sqrt(inner(epsilon, epsilon)+1e-10)
sigma_y = 1e4 + 2.0e5*(rmax-r)
mu_plast = 0.1  + (sigma_y / epsii)
mu_eff = 2 * ( mu_lin * mu_plast )/(mu_lin + mu_plast)
mu = conditional(mu_eff > 0.4, mu_eff, 0.4)
mu_map = conditional(mu_plast < mu_lin, 1.0, 0.0)

# Define time stepping parameters:
max_timesteps = 200
time = 0.0


# UFL for the stokes system
# stress function
def stress(u):
    return 2 * mu * sym(grad(u))


# UFL for the stokes systems
F_stokes = (
    inner(grad(v), stress(u)) * dx
    - div(v) * p * dx + dot(n, v) * p * ds_tb
    - (dot(v, k) * Ra * Ttheta) * dx)

# Continuity equation
F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb

# nitsche free-slip BCs
F_stokes += -dot(v, n) * dot(dot(n, stress(u)), n) * ds_tb
F_stokes += -dot(u, n) * dot(dot(n, stress(v)), n) * ds_tb
F_stokes += C_ip * mu * (p_ip + 1)**2 * \
        FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx\
         + dot(grad(q), kappa * grad(Ttheta)) * dx

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0, bottom_id), DirichletBC(Q, 0.0, top_id)

# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((-X[1], X[0])))
V_nullspace = VectorSpaceBasis([x_rotV])
V_nullspace.orthonormalize()

# Constant nullspace for pressure n
p_nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
# Setting mixed nullspace
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])

# Mapping viscosity
mu_function = Function(Q, name='Viscosity')
mu_lin_function = Function(Q, name='ViscosityLinear')
mu_plastic_function = Function(Q, name='ViscosityPlastic')
mu_map_function = Function(Q, name='Plasticity')


# Write output files in VTK format:
u_, p_ = z.subfunctions
# Next rename for output:
u_.rename("Velocity")
p_.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("visualisation/output.pvd")
dump_period = 10

# Setup problem and solver objects so we can reuse (cache) solver setup
# velocity BC's handled through Nitsche
stokes_problem = NonlinearVariationalProblem(F_stokes, z)
stokes_solver = NonlinearVariationalSolver(
    stokes_problem,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    solver_parameters=newton_stokes_solver_parameters, 
    appctx={'mu': mu})

energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(
                    energy_problem)

# computing the average of temperature field
ver_ave.extrapolate_layer_average(
    Taverage, ver_ave.get_layer_average(Tnew))


# Write initial state:
init_state_checkpoint = CheckpointFile(
    "initial_state.h5",
    'w')
init_state_checkpoint.save_mesh(mesh)
init_state_checkpoint.save_function(Tnew)
init_state_checkpoint.save_function(Taverage)
# closing u_checkpoint
init_state_checkpoint.close()


u_checkpoint = CheckpointFile(
    "reference_velocity.h5",
    'w')
u_checkpoint.save_mesh(mesh)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Solve Stokes sytem:
    stokes_solver.solve()
    
    # saving velocity for inverse 
    u_checkpoint.save_function(u_, idx=timestep)

    # Write output:
    if timestep % dump_period == 0:
        mu_function.interpolate(mu)
        mu_lin_function.interpolate(mu_lin)
        mu_plastic_function.interpolate(mu_plast)
        mu_map_function.interpolate(mu_map)
        output_file.write(
            Tnew,
            u_,
            mu_function,
            mu_lin_function,
            mu_plastic_function,
            mu_map_function)

    # updating time
    time += float(delta_t)

    # Temperature system:
    energy_solver.solve()

    # minimum value of viscosity
    min_viscosity = mu_function.dat.data.min()
    min_viscosity = mu_function.comm.allreduce(min_viscosity, MPI.MIN)

    # Log diagnostics:
    log(f"{timestep}:, {time}, {min_viscosity}")

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)

# Close the file
u_checkpoint.close()

# Write out the final
mu_function.interpolate(mu)
mu_lin_function.interpolate(mu_lin)
mu_plastic_function.interpolate(mu_plast)
mu_map_function.interpolate(mu_map)
output_file.write(
    Tnew,
    u_,
    mu_function,
    mu_lin_function,
    mu_plastic_function,
    mu_map_function)



# Write final state:
state_checkpoint = CheckpointFile(
    "final_state.h5",
    'w')
state_checkpoint.save_mesh(mesh)
state_checkpoint.save_function(Tnew)
# closing u_checkpoint
state_checkpoint.close()

