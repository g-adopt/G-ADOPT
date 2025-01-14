"""Compositional benchmark.
Woidt, W. D. (1978).
Finite element calculations applied to salt dome analysis.
Tectonophysics, 50(2-3), 369-386.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from scipy.constants import g as g  # noqa: F401

import gadopt as ga

from .materials import overburden, salt


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    diag_fields["output_time"].append(simu_time / myr_to_sec)
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["entrainment"].append(
        ga.entrainment(
            diag_vars["level_set"][0],
            diag_params["domain_dim_x"] * diag_params["initial_interface_y"],
            diag_params["entrainment_height"],
        )
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

        ax[0].grid()
        ax[1].grid()

        ax[0].set_xlabel("Time (Myr)")
        ax[1].set_xlabel("Time (Myr)")
        ax[0].set_ylabel("Root-mean-square velocity (m/s)")
        ax[1].set_ylabel("Entrainment (non-dimensional)")

        ax[0].plot(
            diag_fields["output_time"],
            diag_fields["rms_velocity"],
            label="Conservative level set",
        )
        ax[1].plot(
            diag_fields["output_time"],
            diag_fields["entrainment"],
            label="Conservative level set",
        )

        ax[0].legend(fontsize=12, fancybox=True, shadow=True)
        ax[1].legend(fontsize=12, fancybox=True, shadow=True)

        fig.savefig(
            f"{output_path}/rms_velocity_and_entrainment_{tag}.pdf",
            dpi=300,
            bbox_inches="tight",
        )


def symmetric_cubic(x, centre, support, amplitude, vertical_shift):
    """Symmetric cubic with a support, an amplitude, and a vertical shift"""
    return (
        np.where(
            abs(x - centre) > support / 2,
            0,
            amplitude * (support / 2 - abs(x - centre)) ** 3 / (support / 2) ** 3,
        )
        + vertical_shift
    )


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (64e3, 4.8e3)
mesh_gen = "firedrake"
mesh_elements = (512, 32)

# Parameters to initialise level set
interface_coords_x = np.linspace(0.0, domain_dims[0], int(domain_dims[0] / 1e3) + 1)
callable_args = (
    perturbation_x := domain_dims[0] / 2,
    perturbation_support := 6e3,
    interface_deflection := 1,
    initial_interface_y := 2.5e3,
)
signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": symmetric_cubic,
    "interface_args": (interface_coords_x, *callable_args),
}
# The following list must be ordered such that, unpacking from the end, each dictionary
# contains the keyword arguments required to initialise the signed-distance array
# corresponding to the interface between a given material and the remainder of the
# numerical domain (all previous materials excluded). By convention, the material thus
# isolated occupies the positive side of the signed-distance array.
signed_distance_kwargs_list = [signed_distance_kwargs]

# Material ordering must follow the logic implemented in the above list. In other words,
# the last material in the below list must correspond to the portion of the numerical
# domain isolated by the signed-distance array calculated using the last dictionary in
# the above list. The first material in the below list will, therefore, occupy the
# negative side of the signed-distance array calculated from the first dictionary above.
materials = [salt, overburden]

# Approximation parameters
dimensional = True

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"u": 0}, 4: {"uy": 0}}

# Timestepping objects
myr_to_sec = 1e6 * 365.25 * 8.64e4
initial_timestep = 0.01 * myr_to_sec
dump_period = 0.1 * myr_to_sec
checkpoint_period = 5
time_end = 21.3 * myr_to_sec

# Diagnostic objects
diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
entrainment_height = initial_interface_y
diag_params = {
    "domain_dim_x": domain_dims[0],
    "initial_interface_y": initial_interface_y,
    "entrainment_height": entrainment_height,
}
