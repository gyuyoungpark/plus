"""Magnet implementation."""

import numpy as _np
from abc import ABC, abstractmethod

import _mumaxpluscpp as _cpp

from .grid import Grid
from .strayfield import StrayField


class Magnet(ABC):
    """A Magnet should never be initialized by the user. It contains no physics.
    Use ``Ferromagnet`` or ``Antiferromagnet`` instead.

    Parameters
    ----------
    _impl_function : callable
        The appropriate `world._impl` method of the child magnet, for example
        `world._impl.add_ferromagnet` or `world._impl.add_antiferromagnet`.
    world : mumaxplus.World
        World in which the magnet lives.
    grid : mumaxplus.Grid
        The number of cells in x, y, z the magnet should be divided into.
    geometry : None, ndarray, or callable (default=None)
        The geometry of the magnet can be set in three ways.

        1. If the geometry contains all cells in the grid, then use None (the default)
        2. Use an ndarray which specifies for each cell wheter or not it is in the
           geometry.
        3. Use a function which takes x, y, and z coordinates as arguments and returns
           true if this position is inside the geometry and false otherwise.
       
    regions : None, ndarray, or callable (default=None)
        The regional structure of a magnet can be set in the same three ways
        as the geometry. This parameter indexes each grid cell to a certain region.
    name : str (default="")
        The magnet's identifier. If the name is empty (the default), a name for the
        magnet will be created.
    """
    
    @abstractmethod  # TODO: does this work?
    def __init__(self, _impl_function, world, grid, name="", geometry=None, regions=None):
   
        geometry_array = self._get_mask_array(geometry, grid, world, "geometry")
        regions_array = self._get_mask_array(regions, grid, world, "regions")
        self._impl = _impl_function(grid._impl, geometry_array, regions_array, name)

    @staticmethod
    def _get_mask_array(input, grid, world, input_name):
        if input is None:
            return None
        
        T = bool if input_name == "geometry" else int
        if callable(input):
            # construct meshgrid of x, y, and z coordinates for the grid
            nx, ny, nz = grid.size
            cs = world.cellsize
            idxs = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)  # meshgrid of indices
            x, y, z = [(grid.origin[i] + idxs[i]) * cs[i] for i in [0, 1, 2]]

            # evaluate the input function for each position in this meshgrid
            return _np.vectorize(input, otypes=[T])(x, y, z)

        # When here, the input is not None, not callable, so it should be an
        # ndarray or at least should be convertable to ndarray
        input_array = _np.array(input, dtype=T)
        if input_array.shape != grid.shape:
            raise ValueError(
                "The dimensions of the {} do not match the dimensions "
                + "of the grid.".format(input_name)
                    )
        return input_array

    @abstractmethod
    def __repr__(self):
        """Return Magnet string representation."""
        return f"Magnet(grid={self.grid}, name='{self.name}')"

    @classmethod
    def _from_impl(cls, impl):
        magnet = cls.__new__(cls)
        magnet._impl = impl
        return magnet

    @property
    def name(self):
        """Name of the magnet."""
        return self._impl.name

    @property
    def grid(self):
        """Return the underlying grid of the magnet."""
        return Grid._from_impl(self._impl.system.grid)

    @property
    def cellsize(self):
        """Dimensions of the cell."""
        return self._impl.system.cellsize

    @property
    def geometry(self):
        """Geometry of the magnet."""
        return self._impl.system.geometry

    @property
    def origin(self):
        """Origin of the magnet.

        Returns
        -------
        origin: tuple[float] of size 3
            xyz coordinate of the origin of the ferromagnet.
        """
        return self._impl.system.origin

    @property
    def center(self):
        """Center of the magnet.

        Returns
        -------
        center: tuple[float] of size 3
            xyz coordinate of the center of the ferromagnet.
        """
        return self._impl.system.center

    @property
    def world(self):
        """Return the World of which the magnet is a part."""
        from .world import World  # imported here to avoid circular imports
        return World._from_impl(self._impl.world)

    @property
    def enable_as_stray_field_source(self):
        """Enable/disable this magnet (self) as the source of stray fields felt
        by other magnets. This does not influence demagnetization.
        
        Default = True.

        See Also
        --------
        enable_as_stray_field_destination
        """
        return self._impl.enable_as_stray_field_source

    @enable_as_stray_field_source.setter
    def enable_as_stray_field_source(self, value):
        self._impl.enable_as_stray_field_source = value

    @property
    def enable_as_stray_field_destination(self):
        """Enable/disable whether this magnet (self) is influenced by the stray
        fields of other magnets. This does not influence demagnetization.
        
        Default = True.

        See Also
        --------
        enable_as_stray_field_source
        """
        return self._impl.enable_as_stray_field_destination

    @enable_as_stray_field_destination.setter
    def enable_as_stray_field_destination(self, value):
        self._impl.enable_as_stray_field_destination = value


    def stray_field_from_magnet(self, source_magnet: "Magnet"):
        """Return the magnetic field created by the given input `source_magnet`,
        felt by this magnet (`self`). This raises an error if there exists no
        `StrayField` instance between these two magnets.

        Parameters
        ----------
        source_magnet : Magnet
            The magnet acting as the source of the requested stray field.
        
        Returns
        -------
        stray_field : StrayField
            StrayField with the given `source_magnet` as source and the Grid of
            this magnet (`self`) as destination.

        See Also
        --------
        StrayField
        """
        return StrayField._from_impl(
                        self._impl.stray_field_from_magnet(source_magnet._impl))
