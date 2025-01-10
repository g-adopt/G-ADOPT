# Generating reference fields for adjoint inversion
# =================================================
#
# This tutorial explains how to run the forward portion of the adjoint test case to generate the reference final
# condition and synthetic forcing (surface velocity observations).
#
# We will follow a similar structure to the base demo, focusing on generating the necessary fields for the adjoint
# inversion. Let's get started.

# Setting Up the Domain
# ---------------------
# First, we define the domain extents and discretisation.

from gadopt import *
x_max = 1.0
y_max = 1.0
disc_n = 150

# We create a 1D interval mesh and extrude it along the y-axis to form a 2D mesh. This mesh will serve as the basis for
# our simulation.

mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
mesh = ExtrudedMesh(
    mesh1d, layers=disc_n, layer_height=y_max / disc_n, extrusion_type="uniform"
)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

# Defining Function Spaces
# ------------------------
# We set up the function spaces for velocity (Q2), pressure (Q1), and temperature (Q2). These function spaces will be
# used to define the solution fields for our simulation.

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])  # Mixed function space for Stokes problem.

# We also specify functions to hold our solution: a mixed velocity-pressure function, and the initial temperature field
# which represents a synthetic mantle temperature distribution.

# +
z = Function(Z)
u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")

T = Function(Q, name="Temperature")
# -

# For the initial temperature field we use a mathematical expression:
#
# $$ T(x, y) = 0.5 \left( \text{erf}\left((1 - y) \cdot 3.0\right) + \text{erf}\left(-y \cdot 3.0\right) + 1 \right) + 0.1 \exp \left( -0.5 \left( \frac{(x - 0.5)^2 + (y - 0.2)^2}{0.1^2} \right) \right)$$
#
# The first two term of this equation represent the 1-D radial profile, with to error functions that represent the
# thermal boundary layers, and a Gaussian anomaly close to the mantle. This initial state is chosen, such that after 80
# time steps we would have a temperature field representing a plume-like structure in the domain.

# +
X = SpatialCoordinate(mesh)
T.interpolate(
    0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1) +
    0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
)
# -

# Configuring Layer Average Calculation
# -------------------------------------
# We calculate the depth average of model temperature at every timestep. This average
# temperature will serve as a regularisation constraint in the inversion process.

Taverage = Function(Q1, name="Average Temperature")
averager = LayerAveraging(
    mesh, np.linspace(0, 1.0, 150 * 2), quad_degree=6
)
averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

# Checkpointing of fields
# -----------------------
# We checkpoint the temperature at the start and end of our simulation to capture the essential states of our system.
# We checkpoint velocity and layer average temperature (in time loop below) at every timestep. This allows us to retrieve
# these states later using the indices and timestepping history, which are crucial for the adjoint inversion process.
# By saving the initial state, we can compare it against the final state to see how the system evolved, which is analogous
# to how seismic tomography uses initial models to interpret Earth's interior after simulation.

checkpoint_file = CheckpointFile("adjoint-demo-checkpoint-state.h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(T, name="Temperature", idx=0)

# Physical Setup
# --------------
# We define the Rayleigh number and physical approximation for the Boussinesq approximation. This sets up the basic
# physical parameters for our simulation.

# +
Ra = Constant(1e6)
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(4e-6)
timesteps = 80
# -

# Boundary Conditions
# -------------------
# We specify the boundary conditions for the Stokes and temperature problems. These conditions are crucial for ensuring
# the physical realism of our simulation. A higher temperature is imposed on the bottom boundary than the top, setting
# up a convective circulation. Because our simulation takes place in a closed box, we eliminate the constant pressure
# nullspace.

# +
stokes_bcs = {
    boundary.bottom: {"uy": 0},
    boundary.top: {"uy": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}
temp_bcs = {
    boundary.bottom: {"T": 1.0},
    boundary.top: {"T": 0.0},
}

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
# -

# Solvers
# -------
# We set up solvers for the energy and Stokes systems. These solvers will be used to advance the simulation in time and
# solve the governing equations.

# +
energy_solver = EnergySolver(
    T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    constant_jacobian=True,
)
# -

# Time Loop
# ---------
# We perform the time loop to solve the forward problem. During this loop, we solve the Stokes and energy systems, save
# the velocity field for later use, and periodically output the results.

# +
output_file = VTKFile("output.pvd")  # Create output file for visualisation.
for timestep in range(0, timesteps):
    output_file.write(*z.subfunctions, T, Taverage)
    stokes_solver.solve()
    energy_solver.solve()
    averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))  # Calculate layer average

    # Store velocity field and layer average temperature for subsequent use in the adjoint problem.
    checkpoint_file.save_function(u, name="Velocity", idx=timestep,
                                  timestepping_info={"index": float(timestep), "delta_t": float(delta_t)})
    checkpoint_file.save_function(Taverage, name="Average_Temperature", idx=timestep,
                                  timestepping_info={"index": float(timestep), "delta_t": float(delta_t)})

# Save final temperature field to checkpoint file.
checkpoint_file.save_function(T, name="Temperature", idx=timesteps - 1)
checkpoint_file.close()
# -

# This concludes the forward simulation to generate reference fields for the adjoint inversion. The final temperature
# field, after being convected forward for 80 timesteps, serves as the reference temperature field, analogous to a
# seismic tomography image, which allows us to study plume formation and other mantle dynamics features.
