# `psytest` package documentation

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://joseparreiras.github.io/docs/_build/html/index.html)
[![Issues](https://img.shields.io/github/issues/joseparreiras/psytest)](https://github.com/joseparreiras/psytest/issues)
[![Last Commit](https://img.shields.io/github/last-commit/joseparreiras/psytest)](https://github.com/joseparreiras/psytest/commits/main)
[![Stars](https://img.shields.io/github/stars/joseparreiras/psytest?style=social)](https://github.com/joseparreiras/psytest/stargazers)
[![Forks](https://img.shields.io/github/forks/joseparreiras/psytest?style=social)](https://github.com/joseparreiras/psytest/network/members)

## Table of Contents

- [`psytest` package documentation](#psytest-package-documentation)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)

## About <a name = "about"></a>

This is the documentation for the `psytest` package, a Python testing framework that applies the methodology of Phillips, Shi & Yu (2011) to test for the presence of multiple bubbles in a dataset.

A bubble is defined as a period of explosive growth above a unit root in the time series for a consistent period of time.

The main test in the package is the Backward Sup Augmented Dickey Fuller (_BSADF_) test of _PSY_. This test consists of applying the Augmented Dickey Fuller (_ADF_) test to an backward expanding window of the time series at each point in time to identify the presence of bubbles. A bubble is detected once this test rises above the critical value and ends when it returns to the region of non-rejection of the null hypothesis.

The benefits of this test relative to other tests in the literature is its ability to identify multiple bubbles in a dataset.

The main class of the package is the `PSYBubbles` class, which contains methods to calculate the BSADF test statistics and critical values as well as finding the start and end dates of the bubbles.

Check the [documentation](docs/_build/html/index.html) for more details on how to use the package and its methods.

## Getting Started <a name = "getting_started"></a>

Install the package from GitHub using pip:

```bash
pip install git+https://github.com/joseparreiras/psytest
```
