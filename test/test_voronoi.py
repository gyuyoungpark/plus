from mumaxplus import Grid, World
from mumaxplus.util import VoronoiTessellator

import pytest
import numpy as np

def test_generate_output():
    nx, ny, nz = np.random.randint(1, 50, size=3)
    cs = 1e-9
    world = World((cs, cs, cs))
    grid = Grid(size=(nx, ny, nz))
    tess = VoronoiTessellator(grainsize=5e-9)
    result = np.ravel(tess.generate(world, grid))
    assert len(result) == nx * ny * nz

def test_pbc():
    nx, ny, nz = np.random.randint(0, 50, size=3)
    cs = 1e-9
    world_pbc = World((cs, cs, cs), pbc_repetitions=(1, 1, 1), mastergrid=Grid((nx, ny, nz)))
    world_nopbc = World((cs, cs, cs))
    grid = Grid(size=(nx, ny, nz))
    tess = VoronoiTessellator(grainsize=5e-9)
    result_pbc = tess.generate(world_pbc, grid)
    result_nopbc = tess.generate(world_nopbc, grid)
    assert not np.array_equal(result_pbc, result_nopbc)

def test_region_of_center():
    nx, ny, nz = np.random.randint(0, 50, size=3)
    cs = 1e-9
    world = World((cs, cs, cs))
    grid = Grid(size=(nx, ny, nz))
    tess = VoronoiTessellator(grainsize=5e-9, region_of_center=lambda coo: int(coo[0]) % 5)
    result = np.ravel(tess.generate(world, grid))
    assert np.all((0 <= result) & (result < 5))

def test_max_index():
    nx, ny, nz = np.random.randint(0, 50, size=3)
    cs = 1e-9
    world = World((cs, cs, cs))
    grid = Grid(size=(nx, ny, nz))
    tess = VoronoiTessellator(grainsize=5e-9, max_idx=100)
    result = np.ravel(tess.generate(world, grid))
    assert np.all((0 <= result) & (result < 101))

def test_seed():
    nx, ny, nz = np.random.randint(0, 50, size=3)
    cs = 1e-9
    world = World((cs, cs, cs))
    grid = Grid(size=(nx, ny, nz))
    tess1 = VoronoiTessellator(grainsize=5e-9, seed=12345)
    tess2 = VoronoiTessellator(grainsize=5e-9, seed=12345)

    result1 = tess1.generate(world, grid)
    result2 = tess2.generate(world, grid)

    assert np.array_equal(result1, result2)

@pytest.mark.parametrize("size", [(5, 5, 1), (5, 5, 5)])
def test_grid_compatibility(size):
    cs = 1e-9
    world = World((cs, cs, cs))
    grid = Grid(size=size)
    tess = VoronoiTessellator(grainsize=5e-9)
    result = np.ravel(tess.generate(world, grid))
    assert len(result) == size[0] * size[1] * size[2]