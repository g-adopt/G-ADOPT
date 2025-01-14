# Thermochemical convection
# ---

# Rationale
# -

# Our previous tutorial introduced multi-material simulations in G-ADOPT by
# investigating compositional effects on buoyancy. We extend that tutorial to include
# thermal effects, thereby simulating thermochemical convection, which is, for example,
# essential to modelling Earth's mantle evolution.

# This example
# -

# Here, we consider the entrainment of a thin, compositionally dense layer by thermal
# convection presented in [van Keken et al. (1997)](https://doi.org/10.1029/97JB01353).
# Inside a 2-D domain heated from below, a denser material sits at the bottom boundary
# beneath a lighter material. Whilst the compositional stratification is stable, heat
# transfer from the boundary generates positive buoyancy in the denser material,
# allowing thin tendrils to be entrained in the convective circulation. To resolve these
# tendrils using the level-set approach, significant mesh refinement is needed, making
# the simulation computationally expensive. This tutorial will be updated once the
# development of adaptive mesh refinement in Firedrake is complete. We describe below
# the current implementation of this problem in G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which also
# provides access to Firedrake and associated functionality.

from gadopt import *

# For this problem, it is useful to define a mesh with non-uniform refinement. To this
# end, we use the GMSH library to generate a mesh file in a format compatible with
# Firedrake. We specifically increase vertical resolution at the top and bottom
# boundaries of the domain.

# +
import gmsh
from mpi4py import MPI

lx, ly = 2, 1  # Domain dimensions in x and y directions
mesh_hor_res = lx / 50  # Uniform horizontal mesh resolution
mesh_file = "mesh.msh"  # Output mesh file

if MPI.COMM_WORLD.rank == 0:
    gmsh.initialize()
    gmsh.model.add("mesh")

    point_1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_hor_res)
    point_2 = gmsh.model.geo.addPoint(lx, 0, 0, mesh_hor_res)

    line_1 = gmsh.model.geo.addLine(point_1, point_2)

    gmsh.model.geo.extrude(
        [(1, line_1)], 0, ly / 10, 0, numElements=[10], recombine=True
    )  # Vertical resolution: 1e-2

    gmsh.model.geo.extrude(
        [(1, line_1 + 1)], 0, 4 * ly / 5, 0, numElements=[40], recombine=True
    )  # Vertical resolution: 2e-2

    gmsh.model.geo.extrude(
        [(1, line_1 + 5)], 0, ly / 10, 0, numElements=[10], recombine=True
    )  # Vertical resolution: 1e-2

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [line_1 + 2, line_1 + 6, line_1 + 10], tag=1)
    gmsh.model.addPhysicalGroup(1, [line_1 + 3, line_1 + 7, line_1 + 11], tag=2)
    gmsh.model.addPhysicalGroup(1, [line_1], tag=3)
    gmsh.model.addPhysicalGroup(1, [line_1 + 9], tag=4)

    gmsh.model.addPhysicalGroup(2, [line_1 + 4, line_1 + 8, line_1 + 12], tag=1)

    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_file)
    gmsh.finalize()
# -

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
mesh = Mesh(mesh_file)  # Load the GMSH mesh using Firedrake
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects.
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "Q", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])  # Stokes function space (mixed)
Q = FunctionSpace(mesh, "Q", 2)  # Temperature function space (scalar)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space for time step

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Indexed expressions for velocity and pressure
z.subfunctions[0].rename("Velocity")  # Associated Firedrake function for velocity
z.subfunctions[1].rename("Pressure")  # Associated Firedrake function for pressure
T = Function(Q, name="Temperature")  # Firedrake function for temperature
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now initialise the level-set field. All we have to provide to G-ADOPT is a
# mathematical description of the interface location and use the available API. In this
# case, the interface can be represented as a straight line, requiring a callable to be
# provided. Under the hood, G-ADOPT uses the `Shapely` library to determine the
# signed-distance function associated with the interface. We use G-ADOPT's default
# strategy to obtain a smooth step function profile from the signed-distance function.

# +
from numpy import array  # noqa: E402

# Initialise the level-set field. First, determine the signed-distance function at each
# level-set node using a mathematical description of the material-interface location.
# Then, define the thickness of the hyperbolic tangent profile used in the conservative
# level-set approach. Finally, overwrite level-set data array.
interface_coords_x = array([0.0, lx])
callable_args = (interface_slope := 0, interface_y := 0.025)
signed_distance_array = signed_distance(
    psi,
    interface_geometry="curve",
    interface_callable="line",
    interface_args=(interface_coords_x, *callable_args),
)
epsilon = interface_thickness(psi)
psi.dat.data[:] = conservative_level_set(signed_distance_array, epsilon)
# -

# Let us visualise the location of the material interface that we have just initialised.
# To this end, we use Firedrake's built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0, 1, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level-set")
# -

# We next define the material fields and instantiate the approximation. Here, the system
# of equations is non-dimensional and includes compositional and thermal buoyancy terms
# under the Boussinesq approximation. Moreover, physical parameters are constant through
# space apart from density. As a result, the system is fully defined by the values of
# the thermal and compositional Rayleigh numbers. We use the `material_field` function
# to define the compositional Rayleigh number throughout the domain (including the shape
# of the material interface transition). Both non-dimensional numbers are provided to
# our approximation.

# +
Ra = 3e5
Ra_c = material_field(
    psi, [Ra_c_dense := 4.5e5, Ra_c_reference := 0], interface="sharp"
)
approximation = BoussinesqApproximation(Ra, Ra_c=Ra_c)
# -

# Let us now verify that the material fields have been correctly initialised. We plot
# the compositional Rayleigh number across the domain.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(
#     Function(psi).interpolate(Ra_c), levels=linspace(0, 4.5e5, 11), axes=axes
# )
# fig.colorbar(contours, label="Compositional Rayleigh number")
# -

# We move on to initialising the temperature field.

# +
X = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates

# Calculate quantities linked to the temperature initial condition using UFL
u0 = lx ** (7 / 3) / (1 + lx**4) ** (2 / 3) * (Ra / 2 / sqrt(pi)) ** (2 / 3)
v0 = u0
Q_ic = 2 * sqrt(lx / pi / u0)
Tu = erf((1 - X[1]) / 2 * sqrt(u0 / X[0])) / 2
Tl = 1 - 1 / 2 * erf(X[1] / 2 * sqrt(u0 / (lx - X[0])))
Tr = 1 / 2 + Q_ic / 2 / sqrt(pi) * sqrt(v0 / (X[1] + 1)) * exp(
    -(X[0] ** 2) * v0 / (4 * X[1] + 4)
)
Ts = 1 / 2 - Q_ic / 2 / sqrt(pi) * sqrt(v0 / (2 - X[1])) * exp(
    -((lx - X[0]) ** 2) * v0 / (8 - 4 * X[1])
)

# Interpolate temperature initial condition and ensure boundary condition values
T.interpolate(max_value(min_value(Tu + Tl + Tr + Ts - 3 / 2, 1), 0))
DirichletBC(Q, 1, bottom_id).apply(T)
DirichletBC(Q, 0, top_id).apply(T)
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

time_now = 0  # Initial time
delta_t = Function(R).assign(1e-6)  # Initial time step
output_frequency = 1e-4  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# Here, we set up the variational problem for the energy, Stokes, and level-set
# systems. The Stokes and energy systems depend on the approximation defined above,
# and the level-set system includes both advection and reinitialisation components.

# +
# This problem setup has a constant pressure nullspace, which corresponds to the
# default case handled in G-ADOPT.
Z_nullspace = create_stokes_nullspace(Z)

# Boundary conditions are specified next: free slip on all sides, heating from below,
# and cooling from above. No boundary conditions are required for level set, as the
# numerical domain is closed.
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
temp_bcs = {bottom_id: {"T": 1}, top_id: {"T": 0}}
# Instantiate a solver object for the energy conservation system.
energy_solver = EnergySolver(
    T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs
)
# Instantiate a solver object for the Stokes system and perform a solve to obtain
# initial pressure and velocity.
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)
stokes_solver.solve()

# Instantiate a solver object for level-set advection and reinitialisation. G-ADOPT
# provides default values for most arguments; we only provide those that do not have
# one. No boundary conditions are required, as the numerical domain is closed.
adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
# -

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging, instantiate
# G-ADOPT geodynamical diagnostic utility, and define some parameters specific to this
# problem.

# +
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, T, psi, time=time_now)

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms entrainment")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id)

material_area = interface_y * lx  # Area of tracked material in the domain
entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
step = 0  # A counter to keep track of looping
output_counter = 1  # A counter to keep track of outputting
time_end = 0.03  # Will be changed to 0.05 once mesh adaptivity is available
while True:
    # Update timestep
    if time_end - time_now < output_frequency:
        t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    # Advect level set
    level_set_solver.solve()
    # Solve energy system
    energy_solver.solve()
    # Solve Stokes sytem
    stokes_solver.solve()

    # Increment iteration count and time
    step += 1
    time_now += float(delta_t)

    # Calculate proportion of material entrained above a given height
    buoy_entr = entrainment(psi, material_area, entrainment_height)
    # Log diagnostics
    plog.log_str(f"{step} {time_now} {float(delta_t)} {gd.u_rms()} {buoy_entr}")

    # Write output
    if time_now >= output_counter * output_frequency - 1e-16:
        output_file.write(*z.subfunctions, T, psi, time=time_now)
        output_counter += 1

    # Check if simulation has completed
    if time_now >= time_end:
        plog.close()  # Close logging file

        # Checkpoint solution fields to disk
        with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
            final_checkpoint.save_mesh(mesh)
            final_checkpoint.save_function(T, name="Temperature")
            final_checkpoint.save_function(z, name="Stokes")
            final_checkpoint.save_function(psi, name="Level set")

        log("Reached end of simulation -- exiting time-step loop")
        break
# -

# Let us finally examine the location of the material interface and the temperature
# field at the end of the simulation.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# collection = tripcolor(T, axes=axes, cmap="coolwarm")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(collection, label="Temperature")
# -
