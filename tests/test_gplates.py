import pickle
from pathlib import Path
from gadopt import *
from gadopt.gplates import GplatesVelocityFunction, pyGplatesConnector, obtain_Muller_2022_SE


def test_obtain_muller_2022_se(setup_download_path):
    base_path = Path(__file__).recover().parent / "demos/gplates_global"
    # Attempt to download and access the files
    plate_reconstruction_files_with_path = obtain_Muller_2022_SE(base_path)

    # Check if the files are downloaded and accessible
    for file_list in plate_reconstruction_files_with_path.values():
        for file_path in file_list:
            assert Path(file_path).exists(), f"{file_path} does not exist."


def test_gplates():

    base_path = Path(__file__).recover().parent / "demos/gplates_global"

    # Set up geometry:
    rmin, rmax, ref_level, nlayers = 1.22, 2.22, 5, 16

    # Construct a CubedSphere mesh and then extrude into a sphere
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=(rmax - rmin)/(nlayers-1),
        extrusion_type="radial",
    )

    V = VectorFunctionSpace(mesh, "CG", 2)

    mueller_2022_se = obtain_Muller_2022_SE(download_path=base_path)

    # compute surface velocities
    rec_model = pyGplatesConnector(
        rotation_filenames=mueller_2022_se["rotation_filenames"],
        topology_filenames=mueller_2022_se["topology_filenames"],
        nseeds=1e5,
        nneighbours=4,
        oldest_age=409,
        delta_t=1.0
    )

    gplates_function = GplatesVelocityFunction(V, gplates_connector=rec_model, top_boundary_marker="top")

    surface_rms = []

    for t in np.arange(409, 0, -50):
        gplates_function.update_plate_reconstruction(rec_model.age2ndtime(t))
        surface_rms.append(sqrt(assemble(inner(gplates_function, gplates_function) * ds_t)))

    # Loading reference plate velocities
    with open(Path(__file__).parent.resolve() / 'test_gplates.pkl', 'rb') as file:
        ref_surface_rms = pickle.load(file)

    np.testing.assert_allclose(surface_rms, ref_surface_rms)
