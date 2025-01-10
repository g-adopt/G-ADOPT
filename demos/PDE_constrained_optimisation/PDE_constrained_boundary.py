# PDE Constrained Optimisation with G-ADOPT - Boundary Values
# ===========================================================
#
# In this tutorial, we undertake an inversion for an (unknown) initial condition, to match given
# time-dependent boundary values. This differs to our previous tutorial, where our goal was to
# match a given final state.
#
# We start with our usual imports:

from gadopt import *
from gadopt.inverse import *

# Create synthetic twin experiment and record solution at all timesteps
# ---------------------------------------------------------------------
#
# Note that the setup is similar to our previous example, except that the velocity is now counter
# clockwise around the origin $(0,0)$ in the corner of the unit square domain. This implies an inflow
# at the bottom boundary and an outflow boundary on the left of the domain.

# +
mesh = UnitSquareMesh(40, 40)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')
T0 = Function(Q, name="Initial_Temperature")  # T Initial condition which we will invert for.
T0_ref = Function(Q, name="Reference_Initial_Temperature")

x, y = SpatialCoordinate(mesh)
u = Function(V, name="Velocity").interpolate(as_vector((-y, x)))

approximation = BoussinesqApproximation(Ra=1, kappa=5e-2)

delta_t = 0.1
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint)
# -

# The initial condition that we, again, later will invert for, is now centered in the domain.

x0, y0 = 0.5, 0.5
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T0_ref.interpolate(exp(-r2/w**2))

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T0_ref, axes=axes, cmap='magma', vmax=0.5)
# fig.colorbar(collection);
# -

# After setting the initial condition for T, we run this simulation for 20 timesteps to ensure
# the entire Gaussian has left the domain. For this example, we checkpoint the solution at every
# timestep, so that we can later use it as the target boundary values.

num_timesteps = 20
T.project(T0_ref)
with CheckpointFile("Model_State.h5", "w") as model_checkpoint:
    model_checkpoint.save_mesh(mesh)
    for timestep in range(num_timesteps):
        model_checkpoint.save_function(T, idx=timestep)
        energy_solver.solve()
    # After saving idx=0, 19 at beginning of each timestep, we include idx=20 for the solution at
    # the end of the final timestep:
    model_checkpoint.save_function(T, idx=timestep)

# The solution has almost completely disappeared (note the different scalebar):

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='magma', vmax=0.1)
# fig.colorbar(collection);
# -

# Advection diffusion model with unknown initial condition
# --------------------------------------------------------
#
# As with our previous example, we again set up the model with the same configuration, albeit where we
# do not know the initial condition. We will try to find the optimal initial condition such that we closely
# match the recorded outflow boundary values.

with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    mesh = model_checkpoint.load_mesh()
    mesh.cartesian = True

# We now set up the model exactly as before:

# +
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
T = Function(Q, name='Temperature')
T0 = Function(Q, name="Initial_Temperature")
T0_ref = Function(Q, name="Reference_Initial_Temperature")
T_wrong = Function(Q, name="Wrong_Initial_Temperature")

x, y = SpatialCoordinate(mesh)
u = Function(V, name="Velocity").interpolate(as_vector((-y, x)))

approximation = BoussinesqApproximation(Ra=1, kappa=5e-2)
delta_t = 0.1
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint)

# Make our solver output a little less verbose:
if "ksp_converged_reason" in energy_solver.solver_parameters:
    del energy_solver.solver_parameters["ksp_converged_reason"]
# -

# As a first guess we use a Gaussian that is in the wrong place: centred around $(0.7, 0.7)$
# instead of $(0.5, 0.5)$:

x0, y0 = 0.7, 0.7
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T_wrong.interpolate(exp(-r2/w**2))

# As in our first example, we make sure to clear the tape before our actual model starts and
# specify the control at the right stage. During the model we load back in the solutions from the synthetic twin,
# but only use its values at the boundary to compute a mismatch with the current model as an integral over the left
# boundary. Note that we start calculating the functional already in the first timestep, and we keep on adding terms to it,
# all of which will still be automatically recorded by the pyadjoint tape.

# +
tape = get_working_tape()
tape.clear_tape()

T0.project(T_wrong)

m = Control(T0)

J = AdjFloat(0.0)  # Initialise functional
factor = AdjFloat(0.5)  # First & final boundary integral weighted by 0.5 to implement mid-point rule time-integration.

T.project(T0)
with CheckpointFile("Model_State.h5", "r") as model_checkpoint:
    for timestep in range(num_timesteps):
        T_target = model_checkpoint.load_function(mesh, 'Temperature', idx=timestep)
        J = J + factor * assemble((T-T_target)**2*ds(boundary.left))
        factor = 1.0  # Remaining timesteps weighted by 1
        energy_solver.solve()

    T_target = model_checkpoint.load_function(mesh, 'Temperature', idx=timestep)
    # Add final contribution weighted again by 0.5
    J = J + factor * assemble((T-T_target)**2*ds(boundary.left))

print(J)
# -

# We define the reduced functional using the final value of `J` and the specified control. This allows us to rerun
# the model with an arbitrary initial condition. As with our previous example, we first try to simply rerun the
# model with the same "wrong" initial condition, and print the functional.

reduced_functional = ReducedFunctional(J, m)
print(reduced_functional(T_wrong))


# Now we re run the model with the "correct" initial condition from the twin experiment, ending up with
# a near-zero misfit.

x0, y0 = 0.5, 0.5
w = .2
r2 = (x-x0)**2 + (y-y0)**2
T0_ref.interpolate(exp(-r2/w**2))

print(reduced_functional(T0_ref))


# We can again look at the gradient. We evaluate the gradient
# around an initial guess of T=0 as the initial condition, noting
# that when a Function is created its associated data values are zero.

T_wrong.assign(0.0)
reduced_functional(T_wrong)

# In unstructured mesh optimisation problems, it is important to work in the L2 Riesz representation
# to ensure a grid-independent result:

gradJ = reduced_functional.derivative(options={"riesz_representation": "L2"})

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(gradJ, axes=axes, cmap='viridis', vmin=-5, vmax=5)
# fig.colorbar(collection);
# -

# Invert for optimal initial condition using gradient-based optimisation algorithm
# --------------------------------------------------------------------------------
#
# As in the previous example, we can now use ROL to invert for the inital condition.
# We last evaluated the reduced functional with a zero initial condition as the control value,
# so this will be our initial guess.

# We first set lower and upper bound values for the control, which we can
# provide as functions in the same function space as the control:

T_lb = Function(Q).assign(0.0)
T_ub = Function(Q).assign(1.0)

# We next specify our minimisation problem using the LinMore algorithm. As this case is a
# little more challenging than our previous tutorial, we specify 50 iterations as the limit.

# +
minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 50

# Define the LinMore Optimiser class:
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
# -

# And again use our callback function to record convergence:

# +
functional_values = []


def record_value(value, *args):
    if functional_values:
        functional_values.append(min(value, min(functional_values)))
    else:
        functional_values.append(value)


reduced_functional.eval_cb_post = record_value
# -

# We next run the optimisation:

optimiser.run()

# And we'll write the functional values to a file so that we can test them.

with open("functional_boundary.txt", "w") as f:
    f.write("\n".join(str(x) for x in functional_values))

# Let's see how well we have done. At this point a total number of 50 iterations
# have been performed so let's plot convergence:

# + tags=["active-ipynb"]
# plt.semilogy(functional_values)
# plt.xlabel("Iteration #")
# plt.ylabel("Functional value")
# plt.title("Convergence")
# -

# This demonstrates that the functional value decreases by roughly three orders of
# magnitude over the 50 iterations considered. As with the previous tutorial, the
# functional value can be reduced further if more iterations are specified, or if
# the optimisation procedure is configured to continue until a specified tolerance
# is achieved. We can also visualise the optimised initial condition and compare to
# the true initial condition:

# + tags=["active-ipynb"]
# fig, axes = plt.subplots(1,2,figsize=[8,4],subplot_kw={'aspect':1.0})
# ax1 = tripcolor(T0.block_variable.checkpoint, axes=axes[0], cmap='magma', vmax=0.5)
# ax2 = tripcolor(T0_ref, axes=axes[1], cmap='magma', vmax=0.5)
# fig.subplots_adjust(right=0.82)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.68])
# fig.colorbar(ax2,cax=cbar_ax);
# -


# Not bad. Not bad at all! Thank you for listening! Crowd. Goes. Wild.
