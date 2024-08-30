import numpy as _np

import _mumaxpluscpp as _cpp

from ..grid import Grid
from ..world import World

class VoronoiTesselator:

    def __init__(self, world, grid, grainsize):
        cellsize = world.cellsize
        self._impl = _cpp.VoronoiTesselator(grid._impl, grainsize, cellsize)

    def generate(self):
        a = self._impl.generate
        return self._impl.generate()
    
    def centers(self):
        """lijst van tuples (idx, coo)"""
        return 0
    
    def indexdict(self):
        "dictionary met key: idx en values (alle coo bij idx)"
        return 0
        