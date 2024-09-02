import numpy as _np

import _mumaxpluscpp as _cpp
class VoronoiTesselator:

    def __init__(self, world, grid, grainsize):
        cellsize = world.cellsize
        self._impl = _cpp.VoronoiTesselator(grid._impl, grainsize, cellsize)

    def generate(self):
        return self._impl.generate()
    
    def centers(self):
        """lijst of tuples (idx, coo)"""
        return 0
    
    def indexdict(self):
        "dictionary with key: idx and values (all coo belonging to idx)"
        return 0
        