from gadopt import *
from firedrake_adjoint import *
import ROL

dx = dx(degree=6)
ds_t = ds_t(degree=6)

with CheckpointFile("mesh.h5", "r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")

bottom_id, top_id = "bottom", "top"
left_id, right_id = 1, 2

domain_volume = assemble(1*dx(domain=mesh))

# Set up function spaces for the Q2Q1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
Z = MixedFunctionSpace([V, W])

q = TestFunction(Q)
q1 = TestFunction(Q1)

z = Function(Z)  # A field over the mixed function space Z
u, p = z.subfunctions  # Symbolic UFL expressions for u and p
u.rename("Velocity")
p.rename("Pressure")

# Without a restart to continue from, our initial guess is the final state of the forward run
# We need to project the state from Q2 into Q1
T_ic = Function(Q1, name="Initial Temperature")
with CheckpointFile("state/final_state.h5", "r") as f:
    T_ic.project(f.load_function(mesh, "Temperature"))

T = Function(Q, name="Temperature")

u_reference = Function(V, name="Reference Velocity")

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
temp_bcs_q = [
    DirichletBC(Q, 0.0, top_id),
    DirichletBC(Q, 1.0, bottom_id),
]
temp_bcs_q1 = [
    DirichletBC(Q1, 0.0, top_id),
    DirichletBC(Q1, 1.0, bottom_id),
]

# Define a simple problem to apply the imposed boundary condition to the IC
T_ = Function(Q1)
bc_problem = LinearVariationalProblem(
    q1 * TrialFunction(Q1) * dx,
    q1 * T_ * dx,
    T_ic,
    bcs=temp_bcs_q1,
)
bc_solver = LinearVariationalSolver(bc_problem)

# Project the initial condition from Q1 to Q
ic_projection_problem = LinearVariationalProblem(
    q * TrialFunction(Q) * dx,
    q * T_ic * dx,
    T,
    bcs=temp_bcs_q,
)
ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

# Control variable for optimisation
control = Control(T_ic)

# Apply the boundary condition to the control
# and obtain the initial condition
T_.assign(T_ic)
bc_solver.solve()
ic_projection_solver.solve()

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    delta_t,
    ImplicitMidpoint,
    bcs=temp_bcs,
)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

u_checkpoint = CheckpointFile("state/reference_velocity.h5", "r")
u_misfit = 0

output_file = File("visualisation/output_inverse.pvd")
output_file.write(u, T)
dump_period = 10

# Populate the tape by running the forward simulation
for timestep in range(0, max_timesteps):
    stokes_solver.solve()
    energy_solver.solve()

    # Update the total surface velocity misfit
    u_reference.assign(u_checkpoint.load_function(mesh, name="Velocity", idx=timestep))
    u_misfit += assemble(dot(u - u_reference, u - u_reference) * ds_t)
    
    time += float(delta_t)

    average_temperature = assemble(T * dx) / domain_volume
    u_rms = sqrt(assemble(dot(u, u) * dx)) / sqrt(1 / domain_volume)
    log(f"{timestep} {time:.02e} {average_temperature:.1e} {u_rms:.4e}")

    if timestep % dump_period == 0 or timestep == max_timesteps - 1:
        output_file.write(u, T)

# try changing only initial state
with CheckpointFile("state/initial_state.h5", "r") as f:
    T_average = f.load_function(mesh, "Average Temperature")
    T_average.rename("Average Temperature")

with CheckpointFile("state/final_state.h5", "r") as f:
    T_reference = f.load_function(mesh, "Temperature")
    T_reference.rename("Reference Temperature")

alpha_u = 1e-1 # [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
alpha_smoothing = 1e-1  # [1e-1, 1e-2, 1e-3]
alpha_damping = 1e-2  # [1e-2, 1e-3, 1e-4]

X = SpatialCoordinate(mesh)

t_misfit = assemble((T - T_reference) ** 2 * dx)
norm_reference = assemble(T_reference ** 2 * dx)
# This will use the reference velocity from the final timestep
norm_reference_surface_velocity = assemble(dot(u_reference, u_reference) * ds_t)
norm_gradient = assemble(dot(grad(T_reference), grad(T_reference)) * dx)
smoothing = assemble(dot(grad(T_ic - T_average), grad(T_ic - T_average)) * dx)
damping_mask = gaussian(X[1], 1.0, 0.1) + gaussian(X[1], 0.0, 0.1)
damping = assemble(damping_mask * (T_ic - T_average) ** 2 * dx)

# Define the reduced functional for evaluating the optimisation
objective = (
    t_misfit +
    (alpha_u * norm_reference / norm_reference_surface_velocity / max_timesteps) * u_misfit +
    (alpha_smoothing * norm_reference / norm_gradient) * smoothing +
    alpha_damping * damping
)

reduced_functional = ReducedFunctional(objective, control)

# All done with the forward run, stop annotating anything else to the tape
pause_annotation()
log(f"reduced functional: {reduced_functional(T_average)}")

optimisation_output = File("solution/inverse_imposed.pvd")


class StatusTest(ROL.StatusTest):
    def check(self, status):
        # Write out solution at this point
        initial_state = Function(Q1, name="Optimised Initial State")
        initial_state.assign(T_ic.block_variable.checkpoint)  # .restore())

        final_state = Function(Q, name="Optimised Final State")
        final_state.assign(T.block_variable.checkpoint)  # .restore())
        optimisation_output.write(initial_state, final_state)

        initial_misfit = assemble((initial_state - T_average) ** 2 * dx)
        final_misfit = assemble((final_state - T_reference) ** 2 * dx)

        log(f"Initial misfit: {initial_misfit}; final misfit: {final_misfit}")

        # Pass through to the original status test
        return super().check(status)


T_lb = Function(Q1, name="Lower boundary temperature")
T_ub = Function(Q1, name="Upper boundary temperature")
T_lb.assign(0.0)
T_ub.assign(1.0)

minimisation_parameters = {
    "General": {
        "Print Verbosity": 1 if mesh.comm.rank == 0 else 0,
        "Output Level": 1 if mesh.comm.rank == 0 else 0,
        "Krylov": {
            "Iteration Limit": 10,
            "Absolute Tolerance": 1e-4,
            "Relative Tolerance": 1e-2,
        },
        "Secant": {
            "Type": "Limited-Memory BFGS",
            "Maximum Storage": 10,
            "Use as Hessian": True,
            "Barzilai-Borwein": 1,
        },
    },
    "Step": {
        "Type": "Trust Region",
        "Trust Region": {
            "Lin-More": {
                "Maximum Number of Minor Iterations": 10,
                "Sufficient Decrease Parameter": 1e-2,
                "Relative Tolerance Exponent": 1.0,
                "Cauchy Point": {
                    "Maximum Number of Reduction Steps": 10,
                    "Maximum Number of Expansion Steps": 10,
                    "Initial Step Size": 1.0,
                    "Normalize Initial Step Size": True,
                    "Reduction Rate": 0.1,
                    "Expansion Rate": 10.0,
                    "Decrease Tolerance": 1e-8,
                },
                "Projected Search": {
                    "Backtracking Rate": 0.5,
                    "Maximum Number of Steps": 20,
                },
            },
            "Subproblem Model": "Lin-More",
            "Initial Radius": 1.0,
            "Maximum Radius": 1e20,
            "Step Acceptance Threshold": 0.05,
            "Radius Shrinking Threshold": 0.05,
            "Radius Growing Threshold": 0.9,
            "Radius Shrinking Rate (Negative rho)": 0.0625,
            "Radius Shrinking Rate (Positive rho)": 0.25,
            "Radius Growing Rate": 10.0,
            "Sufficient Decrease Parameter": 1e-2,
            "Safeguard Size": 100,
        },
    },
    "Status Test": {
        "Gradient Tolerance": 0,
        "Iteration Limit": 100,
    },
}

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
# create a solver to set up the wrapped objective, vector, and bounds objects
rol_solver = ROLSolver(minimisation_problem, minimisation_parameters, inner_product="L2")

rol_parameters = ROL.ParameterList(minimisation_parameters, "Parameters")
rol_secant = ROL.InitBFGS(minimisation_parameters["General"]["Secant"]["Maximum Storage"])
rol_algorithm = ROL.LinMoreAlgorithm(rol_parameters, rol_secant)

rol_algorithm.setStatusTest(StatusTest(rol_parameters), False)
rol_algorithm.run(
    rol_solver.rolvector,
    rol_solver.rolobjective,
    rol_solver.bounds
)
