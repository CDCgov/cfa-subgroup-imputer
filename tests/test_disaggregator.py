from cfa_subgroup_imputer.imputer import (
    Disaggregator,
    ProportionsFromCategories,
)
from cfa_subgroup_imputer.mapping import OuterProductSubgroupHandler
from cfa_subgroup_imputer.variables import (
    Attribute,
    ImputableAttribute,
)


def test_disaggregator_outer_product():
    supergroup_categories = ["Region1", "Region2"]
    subgroup_categories = [["Low", "High"]]
    supergroup_variable_name = "region"
    subgroup_variable_names = ["income"]

    handler = OuterProductSubgroupHandler()
    group_map = handler.construct_group_map(
        supergroup_categories=supergroup_categories,
        subgroup_categories=subgroup_categories,
        supergroup_variable_name=supergroup_variable_name,
        subgroup_variable_names=subgroup_variable_names,
    )

    supergroup_sizes = {"Region1": 100, "Region2": 200}
    group_map.add_attribute(
        group_type="supergroup",
        attribute_name="size",
        attribute_values=supergroup_sizes,
        impute_action="ignore",
        attribute_class=Attribute,
    )

    subgroup_sizes = {
        ("Low", "Region1"): 40,
        ("High", "Region1"): 60,
        ("Low", "Region2"): 80,
        ("High", "Region2"): 120,
    }
    group_map.add_attribute(
        group_type="subgroup",
        attribute_name="size",
        attribute_values=subgroup_sizes,
        impute_action="ignore",
        attribute_class=Attribute,
    )

    cases_super = {"Region1": 10, "Region2": 50}
    group_map.add_attribute(
        group_type="supergroup",
        attribute_name="cases",
        attribute_values=cases_super,
        impute_action="impute",
        attribute_class=ImputableAttribute,
        measurement_type="count",
    )

    vax_rate_super = {"Region1": 0.5, "Region2": 0.8}
    group_map.add_attribute(
        group_type="supergroup",
        attribute_name="vaccination_rate",
        attribute_values=vax_rate_super,
        impute_action="impute",
        attribute_class=ImputableAttribute,
        measurement_type="rate",
    )

    collection_date_super = {"Region1": "2024-01-01", "Region2": "2024-01-02"}
    group_map.add_attribute(
        group_type="supergroup",
        attribute_name="collection_date",
        attribute_values=collection_date_super,
        impute_action="copy",
        attribute_class=Attribute,
    )

    notes_super = {"Region1": "foo", "Region2": "bar"}
    group_map.add_attribute(
        group_type="supergroup",
        attribute_name="notes",
        attribute_values=notes_super,
        impute_action="ignore",
        attribute_class=Attribute,
    )

    disagg = Disaggregator(
        proportion_calculator=ProportionsFromCategories(size_from="size"),
    )
    result = disagg(group_map)

    # rate values match
    assert result.group(("Low", "Region1")).get_attribute("cases").value == 4.0
    assert (
        result.group(("High", "Region1")).get_attribute("cases").value == 6.0
    )
    assert (
        result.group(("Low", "Region2")).get_attribute("cases").value == 20.0
    )
    assert (
        result.group(("High", "Region2")).get_attribute("cases").value == 30.0
    )

    # rate values match
    for supergrp, val in vax_rate_super.items():
        assert all(
            result.group(grp_name).get_attribute("vaccination_rate").value
            == val
            for grp_name in result.subgroup_names(supergrp)
        )

    # copied values match
    for supergrp, val in collection_date_super.items():
        assert all(
            result.group(grp_name).get_attribute("collection_date").value
            == val
            for grp_name in result.subgroup_names(supergrp)
        )

    # ignored values aren't in subgroups
    assert all(
        result.group(grp_name)._get_attribute("notes") is None
        for grp_name in result.subgroup_names()
    )
