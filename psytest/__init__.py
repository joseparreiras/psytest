"""psytest
=================

This package provides a replication of the :cite:`phillips2011testing` tests for the detection of multiple bubbles in time series data. The main class is :class:`psytest.bubbles.PSYBubbles` (See :mod:`psytest.bubbles` for details), which implements the tests and provides methods for their application and interpretation.

Usage
-----

Load the package and create an instance of the :class:`psytest.bubbles.PSYBubbles` class:

>>> from psytest import PSYBubbles

Load your data (formatted as a :class:`numpy.ndarray`) to an instance of the :class:`psytest.bubbles.PSYBubbles` class:

>>> from psytest import PSYBubbles
>>> psy = PSYBubbles(data = data)

If your data is a :class:`pandas.Series`, you can use the :func:`psytest.bubbles.from_pandas` function to initialize the class:

>>> psy = PSYBubbles.from_pandas(data)
"""

from .bubbles import PSYBubbles

__all__: list[str] = ["PSYBubbles"]
__name__: str = "psytest"
__version__: str = "0.2.8"
__author__: str = "José Antunes Neto"
__email__: str = "jose.neto@kellogg.northwestern.edu"
__description__: str = "Python module to test for the presence of bubbles."
__url__: str = "https://github.com/joseparreiras/psytest"
