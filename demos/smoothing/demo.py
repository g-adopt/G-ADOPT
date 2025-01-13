"""
Isotropic and Anisotropic Smoothing Demonstration
=================================================

This script demonstrates the application of isotropic and anisotropic diffusive smoothing techniques on a cylindrical temperature field using the gadopt platform. The purpose is to illustrate how different diffusion properties can affect the smoothing behavior.

"""

from gadopt import *

# Load a cylindrical temperature field from a checkpoint file
with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

# Define boundary conditions for the temperature field
temp_bcs = {
    "bottom": {'T': 1.0},  # Fixed temperature at the bottom
    "top": {'T': 0.0},     # Fixed temperature at the top
}

# Compute layer average of the temperature for initial comparison
# This helps to visualize changes pre and post smoothing
T_avg = Function(T.function_space(), name='Layer_Averaged_Temp')
averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
averager.extrapolate_layer_average(T_avg, averager.get_layer_average(T))

# Isotropic Smoothing
# -------------------
# In isotropic smoothing, we assume that the diffusion coefficient is the same in all directions.
# This simplifies the diffusion tensor to a scalar value, promoting uniform smoothing across all spatial directions.
smooth_solution_isotropic = Function(T.function_space(), name="Smooth Temperature - Isotropic")
smoother_isotropic = DiffusiveSmoothingSolver(
    function_space=T.function_space(),
    wavelength=0.1,  # Smoothing duration
    bcs=temp_bcs)

smooth_solution_isotropic.assign(smoother_isotropic.action(T))
VTKFile("isotropic_smoothing.pvd").write(T.assign(T - T_avg), smooth_solution_isotropic.assign(smooth_solution_isotropic - T_avg))

# Anisotropic Smoothing
# ---------------------
# Anisotropic smoothing allows different diffusion rates in different directions. This is critical in materials or
# scenarios where properties vary spatially or by direction. Here, we model zero radial diffusion and full tangential
# diffusion.
# Define the radial and tangential conductivity values
kr = Constant(0.0)  # Radial conductivity is zero, preventing diffusion in the radial direction
kt = Constant(1.0)  # Tangential conductivity is one, allowing diffusion in the tangential direction

# Compute radial vector components
X = SpatialCoordinate(T.function_space().mesh())
r = sqrt(X[0]**2 + X[1]**2)
er = as_vector((X[0]/r, X[1]/r))  # Unit radial vector
et = as_vector((-X[1]/r, X[0]/r))  # Unit tangential vector

# Construct the anisotropic conductivity tensor
# K = kr * outer(er, er) + kt * outer(et, et) defines how the diffusion behaves differently in radial and tangential directions.
K = kr * outer(er, er) + kt * outer(et, et)

smoother_anisotropic = DiffusiveSmoothingSolver(
    function_space=T.function_space(),
    wavelength=0.1,
    bcs=temp_bcs,
    K=K)

smooth_solution_anisotropic.assign(smoother_anisotropic.action(T))
VTKFile("anisotropic_smoothing.pvd").write(T.assign(T - T_avg), smooth_solution_anisotropic.assign(smooth_solution_anisotropic - T_avg))

# Visualization is handled by external software that can read .pvd files, such as Paraview.
