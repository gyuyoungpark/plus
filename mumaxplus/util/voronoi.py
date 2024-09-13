import numpy as _np

import _mumaxpluscpp as _cpp
class VoronoiTessellator:

    def __init__(self, world, grid, grainsize):
        self._impl = _cpp.VoronoiTessellator(grid._impl, grainsize, world.cellsize)
        self.tessellation = self._impl.generate()
        
    def generate(self):
        """Returns a Voronoi tessellation.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices."""

        return self.tessellation
    
    @property
    def indexDictionary(self):
        """Create a dictionary where each region (key) is linked
        to a list of grid coordinates (value)."""
        from collections import defaultdict
        _, nz, ny, nx = self.tessellation.shape
        
        idxs = self.tessellation.flatten()
        coords = _np.array(_np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
                          ).reshape(3, -1).T
        
        idxDict = defaultdict(list)
        
        for idx, coord in zip(idxs, coords):
            idxDict[idx].append(tuple(coord))
            
        idxDict = dict(idxDict)
        return idxDict

    @property
    def indices(self):
        """Returns list of unique region indices."""
        return _np.unique(_np.ravel(self.tessellation)).astype(int)

    @property
    def number_of_regions(self):
        """Returns number of unique region indices."""
        return _np.unique(_np.ravel(self.tessellation)).size

    # TODO: implement (C++) function which returns neighbouring regions of a certain idx
