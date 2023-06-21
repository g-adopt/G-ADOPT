from gadopt import *

dx = dx(degree=6)

# Thermal boundary layer thickness
thickness_val = 3

x_max = 1.0
y_max = 1.0

# Number of intervals along x direction
disc_n = 150

# Interval mesh in x direction, to be extruded along y
mesh1d = IntervalMesh(disc_n, length_or_left=0.0, right=x_max)
mesh = ExtrudedMesh(
    mesh1d,
    layers=disc_n,
    layer_height=y_max / disc_n,
    extrusion_type="uniform"
)
bottom_id, top_id = "bottom", "top"
left_id, right_id = 1, 2

domain_volume = assemble(1*dx(domain=mesh))

# Set up function spaces for the P2P1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
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

Ra = Constant(1e6)
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(4e-6)  # Constant time step
max_timesteps = 80
time = 0.0

Z_nullspace = create_stokes_nullspace(Z)

# Free-slip velocity boundary condition on top and bottom
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

solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    delta_t,
    ImplicitMidpoint,
    bcs=temp_bcs,
    solver_parameters=solver_parameters
)
Told = energy_solver.T_old
Ttheta = 0.5*T + 0.5*Told
Told.assign(T)

stokes_solver = StokesSolver(
    z,
    Ttheta,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    solver_parameters=solver_parameters,
)

output_file = File("output.pvd")
dump_period = 10

for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    time += float(delta_t)

    average_temperature = assemble(T * dx) / domain_volume
    log(f"{timestep} {time:.02e} {average_temperature:.1e}")

    if timestep % dump_period == 0:
        output_file.write(u, p, T)