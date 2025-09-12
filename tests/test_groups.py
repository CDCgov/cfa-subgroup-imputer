import pytest

from cfa_subgroup_imputer.groups import Group, GroupMap
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

    def test_copy_modify(self):
        original = Group(
            name="some group",
            attributes=[
                Attribute(
                    name="a variable", value=set(), impute_action="ignore"
                ),
            ],
            subgroup_filter_on=["foo"],
            supergroup_filter_on=["bar"],
        )

        renamed_expected = Group(
            name="otherwise the same group",
            attributes=[
                Attribute(
                    name="a variable", value=set(), impute_action="ignore"
                ),
            ],
            subgroup_filter_on=["foo"],
            supergroup_filter_on=["bar"],
        )

        new_attr = [
            Attribute(
                name="another variable",
                value=set(),
                impute_action="ignore",
            )
        ]
        reattributed_expected = Group(
            name="some group",
            attributes=new_attr,
            subgroup_filter_on=["foo"],
            supergroup_filter_on=["bar"],
        )

        renamed = original._copy_modify(**{"name": "otherwise the same group"})
        reattributed = original._copy_modify(**{"attributes": new_attr})
        print(reattributed)
        print(reattributed_expected)
        assert renamed == renamed_expected
        assert reattributed == reattributed_expected

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
            subgroup=child_precursor, prop=0.42, collision_option="error"
        )

        assert child == child_expected


class TestGroupMap:
    def test_add_attribute(self):
        group_map = GroupMap(
            sub_to_super={
                "subgroup1": "supergroup1",
                "subgroup2": "supergroup1",
            },
            groups=[
                Group(name="supergroup1", attributes=[]),
                Group(name="subgroup1", attributes=[]),
                Group(name="subgroup2", attributes=[]),
            ],
        )

        group_map.add_attribute(
            group_type="subgroup",
            attribute_name="new_attribute",
            attribute_values={"subgroup1": "value1", "subgroup2": "value2"},
            impute_action="ignore",
            attribute_class=Attribute,
        )

        group_map.add_attribute(
            group_type="supergroup",
            attribute_name="other_attribute",
            attribute_values={"supergroup1": "value0"},
            impute_action="ignore",
            attribute_class=Attribute,
        )

        groups_expected = {
            "supergroup1": Group(
                name="supergroup1",
                attributes=[
                    Attribute(
                        name="other_attribute",
                        value="value0",
                        impute_action="ignore",
                    )
                ],
            ),
            "subgroup1": Group(
                name="subgroup1",
                attributes=[
                    Attribute(
                        name="new_attribute",
                        value="value1",
                        impute_action="ignore",
                    )
                ],
            ),
            "subgroup2": Group(
                name="subgroup2",
                attributes=[
                    Attribute(
                        name="new_attribute",
                        value="value2",
                        impute_action="ignore",
                    )
                ],
            ),
        }

        assert group_map.groups == groups_expected
