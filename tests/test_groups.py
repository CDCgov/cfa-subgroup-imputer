import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.variables import Attribute


def test_constructor():
    _ = Group(name="some group", attributes=[])
    _ = Group(
        name="some group",
        attributes=[
            Attribute(name="a variable", value=set(), impute_action="ignore"),
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
                Attribute(name="a variable", value=[], impute_action="copy"),
            },
        )
