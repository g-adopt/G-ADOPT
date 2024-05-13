from gadopt import *

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ncells, nlayers = 1.22, 2.22, 128, 32

# Construct a circle mesh and then extrude into a cylinder:
mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')
bottom_id, top_id = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
T = Function(Q, name="Temperature")
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)

T.interpolate(rmax - r + 0.02*cos(4*atan2(X[1], X[0])) * sin((r - rmin) * pi))

dynamic_topography = Function(W, name="DynamicTopography")
interior_bc = InteriorBC(dynamic_topography.function_space(), 0.0, "top")

Ra = Constant(1e5)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

delta_t = Constant(1e-7)  # Initial time-step

# Define time stepping parameters:
steady_state_tolerance = 1e-7
max_timesteps = 20
time = 0.0

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = VTKFile("output.pvd")
dump_period = 1
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms nu_base nu_top energy avg_t")

temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             cartesian=False, constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)


t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        # calculation of dynamic topography
        dynamic_topography.project(stokes_solver.compute_force_on_surface(stokes_solver.k))
        ave = assemble(dynamic_topography * ds_t) / assemble(1 * ds_t(mesh))
        dynamic_topography.assign(dynamic_topography - Constant(ave))
        interior_bc.apply(dynamic_topography)
        output_file.write(u, p, T, dynamic_topography)

    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)
    f_ratio = rmin/rmax
    top_scaling = -1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = -0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = (assemble(dot(grad(T), n) * ds_t) / assemble(Constant(1.0) * ds_t(domain=mesh))) * top_scaling
    nusselt_number_base = (assemble(dot(grad(T), n) * ds_b) / assemble(Constant(1.0) * ds_b(domain=mesh))) * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(T * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {average_temperature} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        checkpoint_file.save_function(T, name="Temperature", idx=timestep)
        checkpoint_file.save_function(z, name="Stokes", idx=timestep)

plog.close()
checkpoint_file.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
