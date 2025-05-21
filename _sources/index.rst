.. psytest documentation master file, created by
   sphinx-quickstart on Wed Apr  9 11:29:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:code:`psytest` documentation
========================

This is the documentation for the `psytest` package, a Python testing framework that applies the methodology of :cite:`phillips2015testing` (*PSY*) to test for the presence of multiple bubbles in a dataset. 

A bubble is defined as a period of explosive growth above a unit root in the time series for a consistent period of time.

The main test in the package is the Backward Sup Augmented Dickey Fuller (*BSADF*) test of *PSY*. This test consists of applying the Augmented Dickey Fuller (*ADF*) test to an backward expanding window of the time series at each point in time to identify the presence of bubbles. A bubble is detected once this test rises above the critical value and ends when it returns to the region of non-rejection of the null hypothesis.

The benefits of this test relative to other tests in the literature is its ability to identify multiple bubbles in a dataset.

The main class of the package is the :class:`psytest.PSYBubbles` class, which contains methods to calculate the BSADF test statistics and critical values as well as finding the start and end dates of the bubbles.


Installation
------------

You can install the package from my GitHub repository using :code:`pip`:


.. code-block:: bash

    pip install git+https://github.com/joseparreiras/psytest

or clone the repository and install it locally:

.. code-block:: bash

    git clone https://github.com/joseparreiras/psytest
    cd psytest
    pip install .

Usage Example
-------------

The package is designed to be easy to use. The main class is :class:`psytest.PSYBubbles`, which contains methods to calculate the BSADF test statistics and critical values as well as finding the start and end dates of the bubbles.

Assuming the time series data is stored in a :class:`numpy.ndarray`, you can innitiate the class with the data and the desired parameters:

.. code-block:: python

    import numpy as np
    from psytest import PSYBubbles

    # Create an instance of the PSYBubbles class
    psy = PSYBubbles(
       data,
       minwindow=10,
       minlength=90,
       maxlag=2,
    )

    # Find the start and end dates of the bubbles
    psy.find_bubbles(alpha=0.05)

See :doc:`notebooks/replication` for a replication example of the original paper as well as a walkthrough of the package. Also see the :doc:`modules` for a detailed description of the methods and attributes of the :class:`psytest.PSYBubbles` class.

.. toctree::
   :maxdepth: 4
   :caption: Table of Contents

   modules
   notebooks/replication.ipynb

References
------------

.. bibliography:: refs.bib
   :style: plain
   :all:
