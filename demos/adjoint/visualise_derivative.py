from gadopt import *
from firedrake_adjoint import *


def main():
    for case in ["damping", "smoothing", "Tobs", "uobs"]:
        try:
            visualise_derivative(case)
        except Exception:
            raise Exception(f"derivative visualisation for {case} failed!")

def visualise_derivative(case):

    # Make sure we start from a clean tape
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    bottom_id, top_id = "bottom", "top"
    left_id, right_id = 1, 2

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Average temperature function space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    q = TestFunction(Q)
    q1 = TestFunction(Q1)

    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Symbolic UFL expressions for u and p

    # spliting z space to acces velocity and pressure
    u, p = z.split()
    u.rename("Velocity")
    p.rename("Pressure")

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")

    # Temperature function
    T = Function(Q, name="Temperature")

    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)  # Constant time step
    max_timesteps = 80
    init_timestep = 0 if case in ["Tobs", "uobs"] else max_timesteps

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # the initial guess for the control
    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        Tic.project(f.load_function(mesh, "Temperature", idx=max_timesteps - 1))

    # Imposed velocity boundary condition on top, free-slip on other sides
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
    temp_bcs_q1 = [
        DirichletBC(Q1, 0.0, top_id),
        DirichletBC(Q1, 1.0, bottom_id),
    ]

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

    # Define a simple problem to apply the imposed boundary condition to the IC
    T_ = Function(Tic.function_space())
    bc_problem = LinearVariationalProblem(
        q1 * TrialFunction(Tic.function_space()) * dx,
        q1 * T_ * dx,
        Tic,
        bcs=temp_bcs_q1,
    )
    bc_solver = LinearVariationalSolver(bc_problem)

    # Project the initial condition from Q1 to Q
    ic_projection_problem = LinearVariationalProblem(
        q * TrialFunction(Q) * dx,
        q * Tic * dx,
        T,
        bcs=energy_solver.strong_bcs,
    )
    ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

    # Control variable for optimisation
    control = Control(Tic)

    # Apply the boundary condition to the control
    # and obtain the initial condition
    T_.assign(Tic)
    bc_solver.solve()
    ic_projection_solver.solve()

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")

    u_misfit = 0.0

    # Populate the tape by running the forward simulation
    for timestep in range(init_timestep, max_timesteps):
        stokes_solver.solve()

        # load the reference velocity
        uobs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep)
        u_misfit += assemble(0.5 * (uobs - u)**2 * ds_t)
        energy_solver.solve()

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("ObservedTemperature")

    # Load the average temperature profile
    Taverage = checkpoint_file.load_function(mesh, "AverageTemperature", idx=0)
    Taverage.rename("AverageTemperature")

    checkpoint_file.close()
    if case == "smoothing":
        norm_grad_Taverage = assemble(
            0.5*dot(grad(Taverage), grad(Taverage)) * dx)
        objective = 0.5 * assemble(dot(grad(Tic-Taverage), grad(Tic-Taverage)) * dx) / norm_grad_Taverage
    elif case == "damping":
        norm_Tavereage = assemble(
            0.5*(Taverage)**2 * dx)
        objective = 0.5 * assemble((Tic - Taverage)**2 * dx) / norm_Tavereage
    elif case == "Tobs":
        norm_final_state = assemble(
            0.5*(Tobs)**2 * dx)
        objective = 0.5 * assemble((T - Tobs)**2 * dx) / norm_final_state
    else:
        norm_u_surface = assemble(
            0.5 * (uobs)**2 * ds_t)
        objective = u_misfit / (max_timesteps) / norm_u_surface

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    # derivative function
    der_function = reduced_functional.derivative(
        options={'riesz_representation': 'L2'})

    # derivative file
    der_file = File(f"derivative_{case}/derivative.pvd")
    der_file.write(
        der_function)

    # make sure we keep annotating after this
    continue_annotation()


if __name__ == "__main__":
    main()
