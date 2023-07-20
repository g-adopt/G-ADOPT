from gadopt import *
import numpy as np

dx = dx(degree=6)

# Thermal boundary layer thickness
thickness_val = 3

x_max = 1.0
y_max = 1.0

# Number of intervals along x direction
disc_n = 150
depth_profile = np.linspace(0, y_max, disc_n*2)

# Interval mesh in x direction, to be extruded along y
mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
mesh = ExtrudedMesh(
    mesh1d,
    layers=disc_n,
    layer_height=y_max / disc_n,
    extrusion_type="uniform"
)

with CheckpointFile("mesh.h5", "w") as f:
    f.save_mesh(mesh)

bottom_id, top_id = "bottom", "top"
left_id, right_id = 1, 2

domain_volume = assemble(1*dx(domain=mesh))

# Set up function spaces for the Q2Q1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])

z = Function(Z)  # A field over the mixed function space Z
u, p = z.subfunctions  # Symbolic UFL expressions for u and p
u.rename("Velocity")
p.rename("Pressure")

T = Function(Q, name="Temperature")
X = SpatialCoordinate(mesh)
T.interpolate(
    0.5 * (erf((1 - X[1]) * thickness_val) + erf(-X[1] * thickness_val) + 1) +
    0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
)

T_average = Function(Q1, name="Average Temperature")

# Calculate the layer average of the initial state
averager = LayerAveraging(mesh, depth_profile, cartesian=True, quad_degree=6)
averager.extrapolate_layer_average(
    T_average,
    averager.get_layer_average(T)
)

with CheckpointFile("initial_state.h5", "w") as f:
    f.save_mesh(mesh)
    f.save_function(T_average)

Ra = Constant(1e6)
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(4e-6)  # Constant time step
max_timesteps = 80
time = 0.0

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Free-slip velocity boundary condition on all sides
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {
    top_id: {"T": 0.0},
    bottom_id: {"T": 1.0},
}

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    delta_t,
    ImplicitMidpoint,
    bcs=temp_bcs
)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace
)

output_file = File("visualisation/output_forward.pvd")
dump_period = 10

u_checkpoint = CheckpointFile("reference_velocity.h5", "w")
u_checkpoint.save_mesh(mesh)

for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    time += float(delta_t)

    average_temperature = assemble(T * dx) / domain_volume
    log(f"{timestep} {time:.02e} {average_temperature:.1e}")

    u_checkpoint.save_function(u, idx=timestep)

    if timestep % dump_period == 0 or timestep == max_timesteps-1:
        output_file.write(u, p, T)

u_checkpoint.close()

with CheckpointFile("final_state.h5", "w") as f:
    f.save_mesh(mesh)
    f.save_function(T)
