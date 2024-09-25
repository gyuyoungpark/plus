"""InterParameter implementation."""

import _mumaxpluscpp as _cpp

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