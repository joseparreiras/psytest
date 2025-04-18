"""psytest.critval.critval_parameters
===================================================

This module contains the framework for checking the available parameters for the tabulated critical values. The tabulated critical values are stored in the `data` directory of the package and named according to the parameters used to calculate them. The parameters are extracted from the file names and stored in a dictionary for easy access. The module also provides functions to list all available tables and check if a given set of parameters is available.
"""

import os
from enum import Enum
from importlib.resources import files
from importlib.resources.abc import Traversable
from io import StringIO
from enum import Enum
from typing import NamedTuple


class CritValParameterName(Enum):
    """Enum for the critical value parameter names."""

    R0 = "r0"
    RSTEP = "rstep"
    NREPS = "nreps"
    NOBS = "nobs"
    KMAX = "kmax"


class CritValParameterInfo(NamedTuple):
    """Named tuple for the critical value parameters."""

    kmax: int
    r0: float
    rstep: float
    nreps: int
    nobs: int

    def __str__(self) -> str:
        return f"kmax={self.kmax}, r0={self.r0}, rstep={self.rstep}, nreps={self.nreps}, nobs={self.nobs}"


def traversable_to_string(traversable: Traversable) -> str:
    """Convert a Traversable object to a string representation.

    Parameters
    ----------
    traversable : Traversable
        The Traversable object to convert.

    Returns
    -------
    string: str
        The string representation of the Traversable object.
    """
    with StringIO() as buf:
        buf.write(str(traversable))
        return buf.getvalue()


def extract_parameters_from_tablenames(fname: str) -> dict[str, int | float]:
    """Retrieve the parameters from the file name of a critical value table.

    Parameters
    ----------
    fname : str
        Name of the table file

    Returns
    -------
    parameters : dict[str, int | float]
        a dictionary with the parameters and their values
    """
    fname_no_extension: str = os.path.splitext(fname)[0]
    fargs: list[str] = fname_no_extension.lstrip("critval_").split("_")
    param_dict: dict[str, int | float] = {}
    for arg in fargs:
        arg_case: str = arg.casefold()
        for param in CritValParameterName:
            pname: str = param.value.casefold()
            if arg_case.casefold().find(pname) == 0:
                value_str: str = arg_case.split(pname)[1]
                if "p" in value_str:
                    value_str = value_str.replace("p", ".")
                    value: float | int = float(value_str)
                else:
                    value: float | int = int(value_str)
                param_dict.update({pname: value})
    return param_dict


def list_critval_tables() -> list[str]:
    """Lists all critical value tables in the :literal:`data/` directory.

    Returns
    -------
    cval_list : list[str]
        A list of file names of the critical value tables.
    """
    data_path: Traversable = files("psytest.critval.data")
    return [
        traversable_to_string(f)
        for f in data_path.iterdir()
        if f.is_file() and f.name.endswith(".csv") and f.name.startswith("critval_")
    ]


def list_available_tables() -> list[str]:
    """Lists all available critical value tables in the :literal:`data/` directory.

    Returns
    -------
    table_list : list[str]
        A list of file names of the available critical value tables.
    """
    data_path: Traversable = files("psytest.critval.data")
    return [
        os.path.basename(traversable_to_string(f))
        for f in data_path.iterdir()
        if f.is_file() and f.name.endswith(".csv") and f.name.startswith("critval_")
    ]


def list_available_parameters() -> dict[str, CritValParameterInfo]:
    """Retrieves the parameters used to calculate the critical values from the available tables.

    Returns
    -------
    param_list : list[str, ParamInfo]
        A dictionary with the parameters and their values.
    """
    tables: list[str] = list_available_tables()
    enum_dict: dict[str, CritValParameterInfo] = {
        fname: CritValParameterInfo(**extract_parameters_from_tablenames(fname))  # type: ignore
        for fname in tables
    }
    return enum_dict


AVAILABLE_CRITICAL_VALUE_PARAMETERS = Enum(
    "AVAILABLE_CRITICAL_VALUE_PARAMETERS", list_available_parameters()
)


def is_available_param(kmax: int, r0: float) -> bool:
    """Checks if the given parameters are available in the critical value tables.

    Parameters
    ----------
    kmax : int
        Max lag
    r0 : float
        Minimum index

    Returns
    -------
    available: bool
        True if the parameters are available, False otherwise.
    """
    for param in AVAILABLE_CRITICAL_VALUE_PARAMETERS:
        if param.value.kmax == kmax:
            if param.value.r0 <= r0:
                return True
    return False
