import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.variables import Attribute


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
