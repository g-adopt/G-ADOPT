"""
Test case for smoothing
"""
import pytest
import numpy as np
from pathlib import Path
from gadopt import *


# Fixture for loading cylindrical field data
@pytest.fixture
def load_field():
    checkpoint_base = Path(__file__).parent / "../demos/adjoint_2d_cylindrical"
    # Start with a previously-initialised temperature field
    with CheckpointFile(str(checkpoint_base.resolve() / "Checkpoint230.h5"), mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        T = f.load_function(mesh, "Temperature")

    temp_bcs = {
        "bottom": {'T': 1.0},
        "top": {'T': 0.0},
    }
    # Compute layer average for initial stage:
    T_avg = Function(T.function_space(), name='Layer_Averaged_Temp')
    averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
    averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

    return T, T_avg, temp_bcs


def test_isotropic_smoothing(load_field):
    # load field and its average for comparison
    T, T_avg, temp_bcs = load_field

    smooth_solution = Function(T.function_space(), name="smooth temperature")

    smoother = DiffusiveSmoothingSolver(
        function_space=T.function_space(),
        wavelength=.1,
        bcs=temp_bcs)

    smooth_solution.assign(smoother.action(T))
    expected_value = 0.06243541054145882
    result = assemble((smooth_solution - T) ** 2 * dx)
    assert np.isclose(result, expected_value, rtol=1e-5), f"Expected {expected_value}, got {result}"


def test_anisotropic_smoothing(load_field):
    # load field and its average for comparison
    T, T_avg, temp_bcs = load_field

    smooth_solution = Function(T.function_space(), name="smooth temperature")

    # Define the radial and tangential conductivity values
    kr = Constant(0.0)  # Radial conductivity
    kt = Constant(1.0)  # Tangential conductivity

    # Function to compute radial and tangential components of the conductivity tensor
    # Compute radial vector components
    X = SpatialCoordinate(T.function_space().mesh())
    r = sqrt(X[0]**2 + X[1]**2)
    # Unit radial and tangential vectors
    er = as_vector((X[0]/r, X[1]/r))
    et = as_vector((-X[1]/r, X[0]/r))
    # Construct the anisotropic conductivity tensor
    K = kr * outer(er, er) + kt * outer(et, et)

    smoother = DiffusiveSmoothingSolver(
        function_space=T.function_space(),
        wavelength=1e3,
        bcs=temp_bcs,
        K=K)

    smooth_solution.assign(smoother.action(T))
    assert (assemble((T_avg - smooth_solution) ** 2 * dx) < 1e-7)
