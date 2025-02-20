import numpy as _np

import _mumaxpluscpp as _cpp
class VoronoiTessellator:

    def __init__(self, world, grid, grainsize, max_idx=256, seed=1234567):
        self._impl = _cpp.VoronoiTessellator(grid._impl, grainsize, world.cellsize, max_idx, seed)
        
    @property
    def tessellation(self):
        """Returns a Voronoi tessellation.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices."""

        return self._impl.tessellation
    
    @property
    def indexDictionary(self):
        """Create a dictionary where each region (key) is linked
        to a list of grid coordinates (value)."""
        from collections import defaultdict
        tessellation = self.tessellation
        _, nz, ny, nx = tessellation.shape
        
        idxs = tessellation.flatten()
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
        return _np.unique(_np.ravel(self._impl.tessellation)).astype(int)

    @property
    def number_of_regions(self):
        """Returns number of unique region indices."""
        return _np.unique(_np.ravel(self._impl.tessellation)).size

    # TODO: implement (C++) function which returns neighbouring regions of a certain idx
