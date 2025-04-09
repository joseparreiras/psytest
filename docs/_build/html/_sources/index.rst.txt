.. psytest documentation master file, created by
   sphinx-quickstart on Wed Apr  9 11:29:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

psytest documentation
=====================

This is the documentation for the psytest package, a Python testing framework that applies the methodology of :cite:`phillips2011testing` (*PSY*) to test for the presence of multiple bubbles in a dataset. 

A bubble is defined as a period of explosive growth above a unit root in the time series for a consistent period of time.

The main test in the package is the Backward Sup Augmented Dickey Fuller (*BSADF*) test of *PSY*. This test consists of applying the Augmented Dickey Fuller (*ADF*) test to an backward expanding window of the time series at each point in time to identify the presence of bubbles. A bubble is detected once this test rises above the critical value and ends when it returns to the region of non-rejection of the null hypothesis.

The benefits of this test relative to other tests in the literature is its ability to identify multiple bubbles in a dataset.

The main class of the package is the ``PSYBubbles`` class, which contains methods to calculate the BSADF test statistics and critical values as well as finding the start and end dates of the bubbles.

**Usage Example:**

.. code-block:: python

    from psytest.bubbles import PSYBubbles

    model = PSYBubbles(y, delta=1.5)
    model.teststat()


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules

.. bibliography:: refs.bib
   :style: plain