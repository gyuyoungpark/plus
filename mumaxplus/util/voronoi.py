import numpy as _np

import _mumaxpluscpp as _cpp
class VoronoiTesselator:

    def __init__(self, world, grid, grainsize):
        self._impl = _cpp.VoronoiTesselator(grid._impl, grainsize, world.cellsize)
        
    def generate(self):
        """Generate a Voronoi tesselation."""
        self.tesselation = self._impl.generate()
        return self.tesselation[0]
    
    def indexDictionary(self):
        """Create a dictionary where each coordinate (value) is linked
        to a region index (key)."""
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

    def numRegions(self):
        """Returns number of unique region indices."""
        return _np.unique(_np.ravel(self.tesselation)).size
