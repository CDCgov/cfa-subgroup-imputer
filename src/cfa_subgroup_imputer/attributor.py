"""
Submodule for enumerating subgroup and supergroup maps.
"""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from cfa_subgroup_imputer.variables import Range


@runtime_checkable
class Attributor(Protocol):
    """
    A class that assists in making sub to supergroup maps for an underlying
    axis defined by a continuous variable, such as age.

    E.g., something that takes you from "my age subgroups are... and my age
    supergroups are..." to a sub : super group name/string dict.
    """

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str]
    ) -> dict[str, str]:
        """
        Takes flat supergroup and subgroup inputs, returns a sub : super
        group map which can be fed to a GroupMap.
        """
        ...


class AgeGroupAttributor(Attributor):
    """
    A class for enumerating age groups.
    """

    # In here, we'll put all the bespoke code for turning things
    # like "0-<1" years into a usable variable
    STR_TO_YEARS: dict[str, Range] = {}

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str]
    ) -> dict[str, str]:
        raise NotImplementedError()


class CartesianAttributor(Attributor):
    """
    A class for enumerating super and subgroup Cartesian combinations.

    For example, if we have age-based supergroups [0-17 years, 18-64 years, 65+ years]
    and want [low, moderate, high]-risk subgroups, this class makes and handles
    creating all "0-17 years, low risk", ..., "65+ years, high risk" subgroups
    and mapping them to the supergroups.
    """

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str]
    ) -> dict[str, str]:
        raise NotImplementedError()


class Aligner:
    """
    A class that takes in multiple sets of supergroups and defines the largest
    common denominator set of subgroups which allow things to be aligned among
    the groups.
    """

    pass
