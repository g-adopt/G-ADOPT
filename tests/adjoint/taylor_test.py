"""
This standalone script tests the robustness of the derivatives
using the Taylor remainder convergence test.
"""

from gadopt import *
from gadopt.inverse import *
from mpi4py import MPI
import numpy as np
import sys
from pathlib import Path

from cases import cases

ds_t = ds_t(degree=6)
dx = dx(degree=6)


def rectangle_taylor_test(case):
    """
    Perform a second-order Taylor remainder convergence test
    for one term in the objective functional for the rectangular case
    and asserts if convergence is above 1.9

    Args:
        case (str): name of the objective functional term
            either of "damping", "smooothing", "Tobs", "uobs"
    """
    checkpoint_filename = Path(__file__).resolve().parent / "adjoint-demo-checkpoint-state.h5"

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile(str(checkpoint_filename), "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = True

    # Specify boundary markers, noting that for extruded meshes upper and lower boundaries are
    # tagged as "top" and "bottom" respectively.
    boundary = get_boundary_ids(mesh)

    # Retrieve timestepping information for the Velocity and Temperature functions from checkpoint file:
    with CheckpointFile(str(checkpoint_filename), "r") as f:
        temperature_timestepping_info = f.get_timestepping_history(mesh, "Temperature")
        Tobs = f.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
        Tobs.rename("Observed Temperature")
        Tic_ref = f.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
        Tic_ref.rename("Reference Initial Temperature")

    # Set up function spaces:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space

    # Specify test functions and functions to hold solutions:
    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(Q, name="Temperature")

    # Specify important constants for the problem, alongside the approximation:
    Ra = Constant(1e6)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    # Define time-stepping parameters:
    delta_t = Constant(4e-6)  # Constant time step
    timesteps = int(temperature_timestepping_info["index"][-1]) + 1  # number of timesteps from forward

    # Nullspaces for the problem are next defined:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Followed by boundary conditions, noting that all boundaries are free slip, whilst the domain is
    # heated from below (T = 1) and cooled from above (T = 0).
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

    # Setup Energy and Stokes solver
    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 constant_jacobian=True)

    initial_timestep = 0

    # Define control function space:
    Q1 = FunctionSpace(mesh, "CG", 1)

    # Create function for unknown initial temperature condition, which we will invert for. Our initial
    # guess is set to the 1-D average of the forward model. We first load that, at the relevant timestep.
    # Note that this layer average will later be used for the smoothing term in our objective functional.
    with CheckpointFile(str(checkpoint_filename), "r") as f:
        Taverage = f.load_function(mesh, "Average_Temperature", idx=initial_timestep)
    Tic = Function(Q1, name="Initial_Condition_Temperature").assign(Taverage)

    # Given Tic is updated during the optimisation, we also create a function to store our initial guess,
    # which we will later use for smoothing. Since smoothing is executed in the control space, we must
    # specify boundary conditions on this term in that same Q1 space.
    T0_bcs = [DirichletBC(Q1, 0., boundary.top), DirichletBC(Q1, 1., boundary.bottom)]
    T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)

    # We next make pyadjoint aware of our control problem:
    control = Control(Tic)

    # Take initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # We continue by integrating the solutions at each time-step.
    # Notice that we cumulatively compute the misfit term with respect to the
    # surface velocity observable.

    u_misfit = 0.0

    # Next populate the tape by running the forward simulation.
    for time_idx in range(initial_timestep, timesteps):
        stokes_solver.solve()
        energy_solver.solve()
        # Update the accumulated surface velocity misfit using the observed value.
        with CheckpointFile(str(checkpoint_filename), "r") as f:
            uobs = f.load_function(mesh, name="Velocity", idx=time_idx)
        u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Define component terms of overall objective functional and their normalisation terms:
    damping = assemble((T0 - Taverage) ** 2 * dx)
    norm_damping = assemble(Taverage**2 * dx)
    smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    # Define temperature misfit between final state solution and observation:
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    if case == "Tobs":
        objective = t_misfit
    elif case == "uobs":
        objective = norm_obs * u_misfit / timesteps / norm_u_surface
    elif case == "damping":
        objective = norm_obs * damping / norm_damping
    else:
        objective = norm_obs * smoothing / norm_smoothing

    pause_annotation()
    # To define the reduced functional, we provide the class with an objective (which is
    # an overloaded UFL object) and the control.
    reduced_functional = ReducedFunctional(objective, control)

    # Define the perturbation in the initial temperature field
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()

    return minconv


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for case_name in cases:
            minconv = rectangle_taylor_test(case_name)
            print(f"case: {case_name}, result: {minconv}")
    else:
        case_name = sys.argv[1]
        minconv = rectangle_taylor_test(case_name)

        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(f"{case_name}.conv", "w") as f:
                f.write(f"{minconv}")
