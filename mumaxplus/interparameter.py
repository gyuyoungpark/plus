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
        return ""

    def set(self, value):
        """Set the InterParameter value between every region to
        the same value.
        """
        assert isinstance(value, (float, int)
                          ), "The value should be uniform and static."
        self._impl.set(value)

    def set_between(self, i, j, value):
        assert isinstance(self._impl, _cpp.InterParameter
                          ), ("Cannot set value of a regular Parameter" +
                              "between different regions")
        self._impl.set_between(i, j, value)