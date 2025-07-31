import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.variables import Attribute, ImputableAttribute


class TestGroup:
    def test_constructor(self):
        _ = Group(name="some group", attributes=[])
        _ = Group(
            name="some group",
            attributes=[
                Attribute(
                    name="a variable", value=set(), impute_action="ignore"
                ),
                Attribute(
                    name="another variable", value=dict(), impute_action="copy"
                ),
            ],
        )

        with pytest.raises(Exception):
            _ = Group(
                name="some group",
                attributes=None,
            )

        with pytest.raises(Exception):
            _ = Group(
                name="some group",
                attributes={
                    Attribute(
                        name="a variable", value=set(), impute_action="ignore"
                    ),
                    Attribute(
                        name="a variable", value=[], impute_action="copy"
                    ),
                },
            )

    def test_access(self):
        attr = Attribute(
            name="an attribute", value={"a": "value"}, impute_action="ignore"
        )
        grp = Group(name="a group", attributes=[attr])

        assert grp.get_attribute("an attribute") == attr

    def test_eq(self):
        assert Group(
            name="some group",
            attributes=[
                Attribute(
                    name="a variable", value=set(), impute_action="ignore"
                ),
                Attribute(
                    name="another variable", value=dict(), impute_action="copy"
                ),
            ],
        ) == Group(
            name="some group",
            attributes=[
                Attribute(
                    name="a variable", value=set(), impute_action="ignore"
                ),
                Attribute(
                    name="another variable", value=dict(), impute_action="copy"
                ),
            ],
        )

    def test_disagg_partial(self):
        parent = Group(
            name="parent",
            attributes=[
                Attribute(name="size", impute_action="ignore", value=100),
                Attribute(
                    name="foo",
                    impute_action="ignore",
                    value="this should not be copied",
                ),
                Attribute(name="bar", impute_action="copy", value=None),
                ImputableAttribute(
                    name="mcguffin",
                    impute_action="impute",
                    value=3.14159,
                    measurement_type="count",
                ),
                ImputableAttribute(
                    name="nee",
                    impute_action="impute",
                    value=2.718282,
                    measurement_type="rate",
                ),
            ],
        )
        child_precursor = Group(
            name="child",
            attributes=[
                Attribute(name="size", impute_action="ignore", value=42),
                Attribute(
                    name="spanish inquisition",
                    impute_action="ignore",
                    value="nobody expects",
                ),
            ],
        )

        child_expected = Group(
            name="child",
            attributes=[
                Attribute(name="size", impute_action="ignore", value=42),
                Attribute(name="bar", impute_action="copy", value=None),
                ImputableAttribute(
                    name="mcguffin",
                    impute_action="impute",
                    value=0.42 * 3.14159,
                    measurement_type="count",
                ),
                ImputableAttribute(
                    name="nee",
                    impute_action="impute",
                    value=2.718282,
                    measurement_type="rate_from_count",
                ),
                Attribute(
                    name="spanish inquisition",
                    impute_action="ignore",
                    value="nobody expects",
                ),
            ],
        )

        child = parent.disaggregate_one_subgroup(
            subgroup=child_precursor, prop=0.42
        )
        print(child)

        assert child == child_expected
