import firedrake as fd
from gadopt.utility import get_boundary_ids
import pytest
import tempfile
from pathlib import Path
from subprocess import run

# List of tuples of firedrake utility meshes supported by get_boundary_ids, the
# number of dimensions we expect Firedrake to tag boundaries for and the args to the
# mesh creation function
fd_supported_meshes = [
    ("IntervalMesh", 1, (10, 10)),
    ("UnitIntervalMesh", 1, (10,)),
    ("RectangleMesh", 2, (10, 20, 10, 20)),
    ("SquareMesh", 2, (20, 20, 10)),
    ("UnitSquareMesh", 2, (20, 20)),
    ("BoxMesh", 3, (5, 10, 15, 5, 10, 15)),
    ("CubeMesh", 3, (10, 10, 10, 10)),
    ("UnitCubeMesh", 3, (10, 10, 10)),
    ("CircleManifoldMesh", 0, (50,)),
    ("CubedSphereMesh", 0, (20, 2, 2)),
    ("UnitCubedSphereMesh", 0, (2, 2)),
]


@pytest.mark.parametrize("mesh_name,ndim,args", fd_supported_meshes)
def test_utility_meshes(mesh_name, ndim, args):
    mesh_creator = getattr(fd, mesh_name)
    mesh = mesh_creator(*args)
    boundary = get_boundary_ids(mesh)

    if ndim == 0:
        assert boundary.bottom == "bottom"
        assert boundary.top == "top"
        assert not hasattr(boundary, "left")
        assert not hasattr(boundary, "right")
        assert not hasattr(boundary, "front")
        assert not hasattr(boundary, "back")
    else:
        assert boundary.left == 1
        assert boundary.right == 2
        if ndim >= 2:
            assert boundary.bottom == 3
            assert boundary.top == 4
        else:
            assert not hasattr(boundary, "bottom")
            assert not hasattr(boundary, "top")
        if ndim >= 3:
            assert boundary.front == 5
            assert boundary.back == 6
        else:
            assert not hasattr(boundary, "front")
            assert not hasattr(boundary, "back")


@pytest.mark.parametrize("mesh_name,ndim,args", fd_supported_meshes)
def test_recover_from_checkpoint(mesh_name, ndim, args):
    mesh_creator = getattr(fd, mesh_name)
    mesh_w = mesh_creator(*args)
    with tempfile.NamedTemporaryFile() as fp:
        cf = fd.CheckpointFile(fp.name, "w")
        cf.save_mesh(mesh_w)
        cf.close()

        cf = fd.CheckpointFile(fp.name, "r")
        mesh_r = cf.load_mesh()
        cf.close()

    boundary = get_boundary_ids(mesh_r)

    if ndim == 0:
        assert boundary.bottom == "bottom"
        assert boundary.top == "top"
        assert not hasattr(boundary, "left")
        assert not hasattr(boundary, "right")
        assert not hasattr(boundary, "front")
        assert not hasattr(boundary, "back")
    else:
        assert boundary.left == 1
        assert boundary.right == 2
        if ndim >= 2:
            assert boundary.bottom == 3
            assert boundary.top == 4
        else:
            assert not hasattr(boundary, "bottom")
            assert not hasattr(boundary, "top")
        if ndim >= 3:
            assert boundary.front == 5
            assert boundary.back == 6
        else:
            assert not hasattr(boundary, "front")
            assert not hasattr(boundary, "back")


# Make sure extruded meshes aren't adding subdomains we're not expecting
# Only handle cases currently seen in tests and demos
def test_extruded_mesh_interval_to_rectangle():
    mesh1d = fd.IntervalMesh(20, length_or_left=0.0, right=1.0)
    mesh = fd.ExtrudedMesh(mesh1d, layers=20, layer_height=0.05, extrusion_type="uniform")
    boundary = get_boundary_ids(mesh)
    assert boundary.left == 1
    assert boundary.right == 2
    assert boundary.bottom == "bottom"
    assert boundary.top == "top"
    assert not hasattr(boundary, "front")
    assert not hasattr(boundary, "back")


def test_extruded_mesh_square_to_cube():
    a, b, c = 1.0079, 0.6283, 1.0
    nx, ny, nz = 10, int(b / c * 10), 10
    mesh2d = fd.RectangleMesh(nx, ny, a, b, quadrilateral=True)  # Rectangular 2D mesh
    mesh = fd.ExtrudedMesh(mesh2d, nz)
    boundary = get_boundary_ids(mesh)
    assert boundary.left == 1
    assert boundary.right == 2
    assert boundary.front == 3
    assert boundary.back == 4
    assert boundary.bottom == "bottom"
    assert boundary.top == "top"


def test_extruded_mesh_circle_to_cylinder():
    rmin, ncells, nlayers = 1.22, 32, 8
    mesh1d = fd.CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
    mesh = fd.ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")  # extrude into a cylinder
    boundary = get_boundary_ids(mesh)
    assert boundary.bottom == "bottom"
    assert boundary.top == "top"
    assert not hasattr(boundary, "left")
    assert not hasattr(boundary, "right")
    assert not hasattr(boundary, "front")
    assert not hasattr(boundary, "back")


def test_extruded_mesh_cubed_sphere_to_sphere():
    rmin, ref_level, nlayers = 1.208, 4, 8
    mesh2d = fd.CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = fd.ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial")
    boundary = get_boundary_ids(mesh)
    assert boundary.bottom == "bottom"
    assert boundary.top == "top"
    assert not hasattr(boundary, "left")
    assert not hasattr(boundary, "right")
    assert not hasattr(boundary, "front")
    assert not hasattr(boundary, "back")


geo_files = (Path(__file__).parents[1] / "multi_material" / "benchmarks").glob("*.geo")


@pytest.mark.parametrize("geo_name", geo_files)
def test_gmsh_from_geo(geo_name):
    with tempfile.NamedTemporaryFile(suffix=".msh") as fp:
        run(["gmsh", "-2", str(geo_name), "-o", fp.name])
        mesh = fd.Mesh(fp.name)
    boundary = get_boundary_ids(mesh)
    assert boundary.left == 1
    assert boundary.right == 2
    assert boundary.bottom == 3
    assert boundary.top == 4
    assert not hasattr(boundary, "front")
    assert not hasattr(boundary, "back")


def test_gmsh_explicit_boundary():
    # Special test for a .geo file in which boundaries are explicitly labeled
    geo_name = Path(__file__).parents[1] / "base_gmsh" / "square.geo"
    with tempfile.NamedTemporaryFile(suffix=".msh") as fp:
        run(["gmsh", "-2", str(geo_name), "-o", fp.name])
        mesh = fd.Mesh(fp.name)
    boundary = get_boundary_ids(mesh)
    assert boundary.left == 14
    assert boundary.right == 12
    assert boundary.bottom == 11
    assert boundary.top == 13
    assert not hasattr(boundary, "front")
    assert not hasattr(boundary, "back")
