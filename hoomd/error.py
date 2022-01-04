# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""HOOMD Errors."""


class MutabilityError(AttributeError):
    """Raised when setting an attribute after a simulation has been run."""

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name

    def __str__(self):
        """Returns the error message."""
        return (f'The attribute {self.attribute_name} is immutable after '
                'simulation has been run.')


class DataAccessError(RuntimeError):
    """Raised when data is inaccessible until the simulation is run."""

    def __init__(self, data_name):
        self.data_name = data_name

    def __str__(self):
        """Returns the error message."""
        return (f'The property {self.data_name} is unavailable until the '
                'simulation runs for 0 or more steps.')


class TypeConversionError(ValueError):
    """Error when validatimg TypeConverter subclasses fails."""
    pass


class IncompleteSpecificationError(ValueError):
    """Error when a value is missing."""
    pass


class SimulationDefinitionError(RuntimeError):
    """Error in definition of simulation internal state."""
    pass


class IsolationWarning(UserWarning):
    """Warn about data structure removal from original data source."""

    def __str__(self):
        """Returns the error message."""
        return ("The data structure is removed from its original data source, "
                "and updates will no longer modify the previously composing "
                "object. Call obj.to_base() to remove this warning.")
