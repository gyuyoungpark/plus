import numpy as _np

import _mumaxpluscpp as _cpp
class VoronoiTesselator:

    def __init__(self, world, grid, grainsize):
        self._impl = _cpp.VoronoiTesselator(grid._impl, grainsize, world.cellsize)
        self.tesselation = self._impl.generate()
        
    def generate(self):
        """Returns a Voronoi tesselation.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices."""

        return self.tesselation[0]
    
    def indexDictionary(self):
        """Create a dictionary where each region (key) is linked
        to a list of grid coordinates (value)."""
        from collections import defaultdict
        _, nz, ny, nx = self.tesselation.shape
        
        idxs = self.tesselation[0].flatten()
        coords = _np.array(_np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
                          ).reshape(3, -1).T
        
        idxDict = defaultdict(list)
        
        for idx, coord in zip(idxs, coords):
            idxDict[idx].append(tuple(coord))
            
        idxDict = dict(idxDict)
        return idxDict

    def indices(self):
        """Returns list of unique region indices."""
        return _np.unique(_np.ravel(self.tesselation)).astype(int)

    def numRegions(self):
        """Returns number of unique region indices."""
        return _np.unique(_np.ravel(self.tesselation)).size
