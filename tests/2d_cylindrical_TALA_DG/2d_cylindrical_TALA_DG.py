# Idealised 2-D mantle convection problem inside an annulus
# =====================================================================
#
# In this tutorial, we analyse mantle flow in a 2-D annulus domain. We define our domain by the radii
# of the inner ($r_{\text{min}}$) and outer ($r_{\text{max}}$) boundaries. These are chosen such that
# the non-dimensional depth of the mantle, $z = r_{\text{max}} - r_{\text{min}} = 1$, and the ratio of
# the inner and outer radii, $f=r_{\text{min}} / r_{\text{max}} = 0.55$, thus approximating the ratio
# between the radii of Earth's surface and core-mantle-boundary (CMB). Specifically, we set
# $r_{\text{min}} = 1.208$ and $r_{\text{max}} = 2.208$.
#
# This example focusses on differences between incompressible isoviscous, and compressible (TALA)
# variable viscosity simulations in a 2-D annulus. It also incorporates a DG discretisation of temperature.
# Key differences can be summarised as follows:
# 1. Function space for temperature.
# 2. Specification of TALA approximation and associated reference fields.
# 3. Specification of 1-D viscosity profile from a file, and variations around this profile due to temperature.
#
# The example is configured at $Ra = 5e7$. Boundary conditions are free-slip at the surface and base of the domain.
#
#
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.
# We also import pyvista, which is used for plotting vtk output.

from gadopt import *

rmin, rmax, ncells, nlayers = 1.208, 2.208, 512, 128

# In this example, we load the mesh from a checkpoint, although the following code was used to generate
# the original mesh. It is included here for completeness.


def original_mesh():
    def gaussian(center, c, a):
        return a*np.exp(-(np.linspace(rmin, rmax, nlayers)-center)**2/(2*c**2))

    resolution_func = np.ones((nlayers))
    for idx, r_0 in enumerate([rmin, rmax, rmax - 660/6370]):
        c = 0.15
        res_amplifier = 5.
        resolution_func *= 1/(1+gaussian(center=r_0, c=c, a=res_amplifier))

    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
    # extrude circle into a cylinder
    mesh = ExtrudedMesh(
        mesh1d,
        layers=nlayers,
        layer_height=(rmax-rmin)*resolution_func/np.sum(resolution_func),
        extrusion_type='radial',
    )

    return mesh


with CheckpointFile("initial_condition_mat_prop/Final_State.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")  # Load temperature from checkpoint
    z = f.load_function(mesh, "Stokes")  # Load velocity and pressure from checkpoint
# We set the mesh `cartesian` attribute to False, which ensures that
# the unit vector points radially, in the direction opposite to gravity.
mesh.cartesian = False
boundary = get_boundary_ids(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
V_scalar = FunctionSpace(mesh, "CG", 2)  # CG2 Scalar function space
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "DQ", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
vr = Function(V_scalar, name="Radial_Velocity")  # For diagnostic output

# We next specify the important constants for this problem, and set up the approximation.
Ra = Constant(5.0e7)  # Rayleigh number
Di = Constant(0.9492824165791792)  # Dissipation number
H_int = Constant(9.93)  # Internal heating

X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
rhat = as_vector([X[0]/r, X[1]/r])  # Radial unit vector (in direction opposite to gravity)
thetahat = as_vector([-X[1]/r, X[0]/r])  # Tangential unit vector.

approximation_profiles = {}
approximation_sources = {
    "rho": {
        "name": "CompRefDensity",
        "filename": "initial_condition_mat_prop/rhobar.txt",
        "scaling": lambda x: x / 3200.,
    },
    "Tbar": {
        "name": "CompRefTemperature",
        "filename": "initial_condition_mat_prop/Tbar.txt",
        "scaling": lambda x: (x - 1600.) / 3700.,
    },
    "alpha": {
        "name": "IsobaricThermalExpansivity",
        "filename": "initial_condition_mat_prop/alphabar.txt",
        "scaling": lambda x: x / 4.1773e-05,
    },
    "cp": {
        "name": "IsobaricSpecificHeatCapacity",
        "filename": "initial_condition_mat_prop/CpSIbar.txt",
        "scaling": lambda x: x / 1249.7,
    },
    "g": {
        "name": "GravitationalAcceleration",
        "filename": "initial_condition_mat_prop/gbar.txt",
        "scaling": lambda x: x / 9.8267,
    },
}

for func, details in approximation_sources.items():
    f = Function(Q, name=details["name"])
    interpolate_1d_profile(function=f, one_d_filename=details["filename"])
    f.assign(details["scaling"](f))

    approximation_profiles[func] = f

Tbar = approximation_profiles["Tbar"]

# We next prepare our viscosity, starting with a radial profile.
mu_rad = Function(Q, name="Viscosity_Radial")  # Depth dependent component of viscosity
interpolate_1d_profile(function=mu_rad, one_d_filename="initial_condition_mat_prop/mu2_radial.txt")

# We next set up and initialise auxiliary fields associated with temperature:
FullT = Function(Q, name="FullTemperature").assign(T+Tbar)
T_avg = Function(Q, name='Layer_Averaged_Temp')
averager = LayerAveraging(mesh, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))
T_dev = Function(Q, name='Temperature_Deviation').assign(FullT-T_avg)

# Now that we have the average T profile, we add lateral viscosity variation due to temperature variations:
mu_field = Function(Q, name="Viscosity")
delta_mu_T = Constant(1000.)
mu = mu_rad * exp(-ln(delta_mu_T) * T_dev)

# These fields are used to set up our Truncated Anelastic Liquid Approximation.
approximation = TruncatedAnelasticLiquidApproximation(Ra, Di, H=H_int, mu=mu, **approximation_profiles)

# As with the previous examples, we set up a *Timestep Adaptor*,
# for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time. For the latter,
# we specify the initial time, initial timestep $\Delta t$, and number of
# timesteps.

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 21  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, target_cfl=0.8, maximum_timestep=0.1, increase_tolerance=1.5)

# With a free-slip boundary condition on both boundaries, one can add an arbitrary rotation
# of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity solution (i.e. this case incorporates a velocity nullspace,
# as well as a pressure nullspace). These lead to null-modes (eigenvectors) for the linear system, rendering the resulting matrix singular.
# In preconditioned Krylov methods these null-modes must be subtracted from the approximate solution at every iteration. We do that below,
# setting up a nullspace object as we did in the previous tutorial, albeit speciying the `rotational` keyword argument to be True.
# This removes the requirement for a user to configure these options, further simplifying the task of setting up a (valid) geodynamical simulation.

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

# Given the increased computational expense (typically requiring more degrees of freedom) in a 2-D annulus domain, G-ADOPT defaults to iterative
# solver parameters. As noted in our previous 3-D Cartesian tutorial, G-ADOPT's iterative solver setup is configured to use the GAMG preconditioner
# for the velocity block of the Stokes system, to which we must provide near-nullspace information, which, in 2-D, consists of two rotational and two
# translational modes.

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Boundary conditions are next specified. Boundary conditions for temperature are set to $T = 0$ at the surface ($r_{\text{max}}$) and $T = 1 - adiabatic contribution$
# at the base ($r_{\text{min}}$). For velocity, we specify free‐slip conditions on both boundaries. We incorporate these **weakly** through
# the _Nitsche_ approximation. This illustrates a key advantage of the G-ADOPT framework: the user only specifies that the normal component
# of velocity is zero and all required changes are handled under the hood.

# +
stokes_bcs = {
    boundary.bottom: {'un': 0},
    boundary.top: {'un': 0},
}

temp_bcs = {
    boundary.bottom: {'T': 1.0 - 930/3700.},  # Take out adiabat
    boundary.top: {'T': 0.0},
}
# -

# We next setup our output, in VTK format.
# We also open a file for logging and calculate our diagnostic outputs.

# +
output_file = VTKFile("output.pvd")
ref_file = VTKFile("reference_state.pvd")
output_frequency = 10

plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt u_rms nu_base nu_top energy avg_t FullT_min FullT_max")

gd = GeodynamicalDiagnostics(z, FullT, boundary.bottom, boundary.top, quad_degree=6)
# -

# We can now setup and solve the variational problem, for both the energy and Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)
stokes_solver.solver_parameters["fieldsplit_0"]["ksp_converged_reason"] = None
stokes_solver.solver_parameters["fieldsplit_1"]["ksp_converged_reason"] = None

# We now initiate the time loop, which runs for the number of timesteps specified above.

for timestep in range(0, timesteps):

    # Update varial velocity:
    vr.interpolate(inner(u, rhat))

    # Write output:
    if timestep % output_frequency == 0:
        # interpolate mu to field for visualisation
        mu_field.interpolate(mu)
        output_file.write(*z.subfunctions, vr, FullT, T, T_dev, mu_field)
        ref_file.write(*approximation_profiles.values(), mu_rad, T_avg)

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
    f_ratio = rmin/rmax
    top_scaling = 1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = 0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = gd.Nu_top() * top_scaling
    nusselt_number_base = gd.Nu_bottom() * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {gd.u_rms()} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {gd.T_avg()} {gd.T_min()} {gd.T_max()}")

    # Calculate Full T and update gradient fields:
    FullT.assign(T+Tbar)
    # Compute deviation from layer average
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(FullT))
    T_dev.assign(FullT-T_avg)

# +
plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
# -
