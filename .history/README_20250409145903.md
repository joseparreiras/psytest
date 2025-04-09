# `psytest` package documentation

[![PyPI](https://img.shields.io/pypi/v/psytest?color=blue)](https://pypi.org/project/psytest/)
[![Python Version](https://img.shields.io/pypi/pyversions/psytest)](https://www.python.org/)
[![License](https://img.shields.io/github/license/joseparreiras/psytest)](https://github.com/joseparreiras/psytest/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/joseparreiras/psytest/python-app.yml)](https://github.com/joseparreiras/psytest/actions)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://joseparreiras.github.io/docs/_build/html/index.html)

## Table of Contents

- [`psytest` package documentation](#psytest-package-documentation)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)

## About <a name = "about"></a>

This is the documentation for the `psytest` package, a Python testing framework that applies the methodology of :cite:`phillips2011testing` (_PSY_) to test for the presence of multiple bubbles in a dataset.

A bubble is defined as a period of explosive growth above a unit root in the time series for a consistent period of time.

The main test in the package is the Backward Sup Augmented Dickey Fuller (_BSADF_) test of _PSY_. This test consists of applying the Augmented Dickey Fuller (_ADF_) test to an backward expanding window of the time series at each point in time to identify the presence of bubbles. A bubble is detected once this test rises above the critical value and ends when it returns to the region of non-rejection of the null hypothesis.

The benefits of this test relative to other tests in the literature is its ability to identify multiple bubbles in a dataset.

The main class of the package is the `PSYBubbles` class, which contains methods to calculate the BSADF test statistics and critical values as well as finding the start and end dates of the bubbles.

## Getting Started <a name = "getting_started"></a>

Install the package from GitHub using pip:

```bash
pip install git+https://github.com/joseparreiras/psytest
```
