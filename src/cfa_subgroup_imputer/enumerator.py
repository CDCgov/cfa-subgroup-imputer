"""
Submodule for enumerating subgroup and supergroup maps.
"""

from collections.abc import Collection, Iterable
from typing import Protocol, runtime_checkable

from cfa_subgroup_imputer.variables import Range


@runtime_checkable
class Enumerator(Protocol):
    """
    A class that assists in making sub to supergroup maps for an underlying
    axis defined by a continuous variable, such as age.

    E.g., something that takes you from "my age subgroups are... and my age
    supergroups are..." to a sub : super group name/string dict.
    """

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str], **kwargs
    ) -> dict[str, str]:
        """
        Takes flat supergroup and subgroup inputs, returns a sub : super
        group map which can be fed to a GroupMap.
        """
        ...


class AgeGroupEnumerator:
    """
    A class for enumerating age groups.
    """

    # In here, we'll put all the bespoke code for turning things
    # like "0-<1" years into a usable variable
    STR_TO_YEARS: dict[str, Range] = {}

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str], **kwargs
    ) -> dict[str, str]:
        missing_supergroups = self.find_missing(supergroups, subgroups)
        missing_option = kwargs.get("missing_option", "error")
        if len(missing_supergroups) > 0:
            if missing_option == "add":
                raise NotImplementedError()
            else:
                raise RuntimeError(
                    f"Subgroups {missing_supergroups} have no corresponding supergroups in {supergroups}"
                )

        raise NotImplementedError()

    def find_missing(
        self, supergroups: Iterable[str], subgroups: Iterable[str]
    ) -> Collection[str]:
        """
        Returns list of subgroups without supergroups. Empty if all are enumerated.
        """
        raise NotImplementedError()


class CartesianEnumerator:
    """
    A class for enumerating super and subgroup Cartesian combinations.

    For example, if we have age-based supergroups [0-17 years, 18-64 years, 65+ years]
    and want [low, moderate, high]-risk subgroups, this class makes and handles
    creating all "0-17 years, low risk", ..., "65+ years, high risk" subgroups
    and mapping them to the supergroups.
    """

    def attribute(
        self, supergroups: Iterable[str], subgroups: Iterable[str], **kwargs
    ) -> dict[str, str]:
        raise NotImplementedError()
