"""psytest.critval
==================

Module to handling pre-calculated critical values for the :class:`psytest.bubbles.PSYBubbles` class. The critical values are stored in the `data/` directory of the package and are used to determine the significance of the test statistics calculated by the `PSYBubbles` class.
"""

from .critval import critval_tabulated
from .critval_parameters import is_available_param

__all__: list[str] = ["critval_tabulated", "is_available_param"]
