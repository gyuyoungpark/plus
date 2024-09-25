"""InterParameter implementation."""

import _mumaxpluscpp as _cpp
import numpy as _np

class InterParameter():
    """Represent a physical material parameter which acts between
    different regions, i.e. `inter_exchange`.
    """

    def __init__(self, impl):
        """Initialize a python InterParameter from a c++ InterParameter instance.

        InterParameters should only have to be initialized within the mumaxplus
        module and not by the end user.
        """
        self._impl = impl

    def __repr__(self):
        """Return InterParameter string representation."""
        return (
            f"InterParameter(number_of_regions={self.number_of_regions}, "
            f"name='{self.name}', ncomp={self.ncomp}, unit={self.unit})"
        )

    def eval(self):
        """Evaluate the quantity.
        Return a numpy array containing an upper triangular matrix where each
        region index corresponds to a row/column index. The elements of this
        matrix corresponds to the values of self between the two regions.
        """
        values = self._impl.eval()
        if not values:
            raise ValueError(f"The InterParameter '{self.name}' is not defined.")

        N = int(_np.max(self.region_indices())) + 1

        value_matrix = _np.zeros((N, N))
        value_matrix[_np.triu_indices(N, k=1)] = values
        return value_matrix

    def __call__(self):
        """Evaluate the quantity.
        Return a numpy array containing an upper triangular matrix where each
        region index corresponds to a row/column index. The elements of this
        matrix corresponds to the values of self between the two regions."""
        return self.eval()

    @property
    def name(self):
        """Return instance's name."""
        return self._impl.name

    @property
    def unit(self):
        """Return unit of the quantity."""
        return self._impl.unit

    @property
    def ncomp(self):
        """Return the number of components of the quantity."""
        return self._impl.ncomp

    @property
    def number_of_regions(self):
        """Return the number of regions between which the quantity
        is active."""
        return self._impl.number_of_regions

    @property
    def region_indices(self):
        """Return list of unique region indices."""
        return self._impl.unique_regions

    def set(self, value):
        """Set the InterParameter value between every different region
        to the same value.
        """
        assert isinstance(value, (float, int)
                          ), "The value should be uniform and static."
        self._impl.set(value)

    def set_between(self, i, j, value):
        """Set InterParameter value between regions i and j."""
        self._impl.set_between(i, j, value)