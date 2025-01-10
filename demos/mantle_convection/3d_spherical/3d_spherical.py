# Idealised 3-D spherical mantle convection.
# ==========================================
#
# In this tutorial, we analyse a case in a 3-D spherical shell geometry.  We examine a well-known isoviscous community benchmark case,
# at a Rayleigh number of $Ra = 7 \times 10^{3}$, with free-slip velocity boundary conditions at both boundaries. Temperature boundary conditions are set to 1
# at the base of the domain ($r_{\text{min}} = 1.22$) and 0 at the surface ($r_{\text{max}}=2.22$), with the initial temperature
# distribution approximating a conductive profile with superimposed perturbations triggering tetrahedral symmetry at spherical harmonic
# degree $l=3$ and order $m=2$ (for further details, see Zhong et al. 2008, or Davies et al. 2022).
#
# This example focusses on differences between running simulations in a 2-D annulus and a 3-D sphere. These are
# 1. The geometry of the problem - i.e. the computational mesh.
# 2. Initialisation of the temperature field in a different domain.
#
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.
# We also import scipy.special and math, required to generate our initial condition for temperature.

from gadopt import *
import scipy.special
import math

# We next set up the mesh, function spaces, and specify functions to hold our solutions,
# as with our previous tutorials. For the mesh, we use Firedrake's built-in *CubedSphereMesh* and extrude it radially through
# 8 layers, forming hexahedral elements. As with our cylindrical shell example, we approximate the curved spherical domain quadratically,
# using the optional keyword argument *degree$=2$*.
# Because this problem is not formulated in a Cartesian geometry, we set the `mesh.cartesian`
# attribute to False. This ensures the correct configuration of a radially inward vertical direction.

# +
rmin, rmax, ref_level, nlayers = 1.208, 2.208, 4, 8

mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')
mesh.cartesian = False
boundary = get_boundary_ids(mesh)
domain_volume = assemble(1*dx(domain=mesh))  # Required for a diagnostic calculation.

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
# -

# We next specify the important constants for this problem, and set up the approximation.

Ra = Constant(7e3)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# As with the previous examples, we set up a *Timestep Adaptor*,
# for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time. For the latter,
# we specify the initial time, initial timestep $\Delta t$, and number of
# timesteps. Given the low Rayleigh number, a steady-state tolerance is also specified,
# allowing the simulation to exit when a steady-state has been achieved.

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 20  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)
steady_state_tolerance = 1e-6  # Used to determine if solution has reached a steady state.

# We next set up and initialise our Temperature field, and also specify two fields for computing
# lateral deviations from a radial layer average.

# +
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)

conductive_term = rmin*(rmax - r) / (r*(rmax - rmin))
l, m, eps_c, eps_s = 3, 2, 0.01, 0.01
Plm = Function(Q, name="P_lm")
cos_phi = Function(Q).interpolate(cos(phi))
Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)  # Evaluate P_lm node-wise using scipy lpmv
Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
if m == 0:
    Plm.assign(Plm/math.sqrt(2))

T = (
    Function(Q, name="Temperature")
    .interpolate(
        conductive_term +
        (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin))
    )
)

T_avg = Function(Q, name="Layer_Averaged_Temp")
T_dev = Function(Q, name="Temperature_Deviation")
# -

# Compute layer average for initial temperature field, using the LayerAveraging functionality provided by G-ADOPT.

averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# Nullspaces and near-nullspace objects are next set up,

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

# Followed by boundary conditions for velocity and temperature.

# +
stokes_bcs = {
    boundary.bottom: {'un': 0},
    boundary.top: {'un': 0},
}

temp_bcs = {
    boundary.bottom: {'T': 1.0},
    boundary.top: {'T': 0.0},
}
# -

# We next setup our output, in VTK format.
# We also open a file for logging and set up our diagnostic outputs.

# +
output_file = VTKFile("output.pvd")
output_frequency = 1

plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms nu_top nu_base energy avg_t t_dev_avg")

gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top, quad_degree=6)
# -

# We can now setup and solve the variational problem, for both the energy and Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.

# +
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)
# -

# We now initiate the time loop, which runs until a steady-state solution has been attained.

for timestep in range(0, timesteps):

    # Write output:
    if timestep % output_frequency == 0:
        # Compute radially averaged temperature profile as simulation evolves.
        averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))
        # Compute deviation from layer average
        T_dev.assign(T-T_avg)
        output_file.write(*z.subfunctions, T, T_dev)

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
    nusselt_number_top = gd.Nu_top() * (rmax*(rmin-rmax)/rmin) * -1.
    nusselt_number_base = gd.Nu_bottom() * (rmin*(rmax-rmin)/rmax)
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    T_dev_avg = assemble(T_dev * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} "
                 f"{nusselt_number_top} {nusselt_number_base} "
                 f"{energy_conservation} {gd.T_avg()} {T_dev_avg} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

# At the end of the simulation, once a steady-state has been achieved, we close our logging file
# and checkpoint steady state temperature and Stokes solution fields to disk. These can later be
# used to restart a simulation, if required.

# +
plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
# -
