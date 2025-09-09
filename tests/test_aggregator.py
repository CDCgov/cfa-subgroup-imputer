from cfa_subgroup_imputer.imputer import Aggregator
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
)
from cfa_subgroup_imputer.variables import ImputableAttribute


def test_aggregator_age_continuous():
    supergroups = ["0-17 years", "18+ years"]
    subgroups = [
        "0-4 years",
        "5-17 years",
        "18-64 years",
        "65+ years",
    ]
    age_max = 100

    age_handler = AgeGroupHandler(age_max=age_max)
    group_map = age_handler.construct_group_map(
        supergroups=supergroups, subgroups=subgroups
    )

    subgroup_sizes = {
        "0-4 years": 500.0,
        "5-17 years": 1300.0,
        "18-64 years": 4700.0,
        "65+ years": 3500.0,
    }
    group_map.add_attribute(
        group_type="subgroup",
        attribute_name="size",
        attribute_values=subgroup_sizes,
        impute_action="impute",
        attribute_class=ImputableAttribute,
        measurement_type="count",
    )

    aggregator = Aggregator()
    result_map = aggregator(group_map)
    print(result_map.group("0-17 years"))

    expected_supergroup_sizes = {"0-17 years": 1800.0, "18+ years": 8200.0}

    for supergroup_name, expected_size in expected_supergroup_sizes.items():
        assert (
            result_map.group(supergroup_name).get_attribute("size").value
            == expected_size
        )


# def test_aggregator_outer_product():
#     """
#     Tests that the Aggregator correctly sums data from subgroups to supergroups.
#     This test is the inverse of test_disaggregator_outer_product.
#     """
#     supergroup_categories = ["Region1", "Region2"]
#     subgroup_categories = [["Low", "High"]]
#     supergroup_variable_name = "region"
#     subgroup_variable_names = ["income"]

#     handler = OuterProductSubgroupHandler()
#     group_map = handler.construct_group_map(
#         supergroup_categories=supergroup_categories,
#         subgroup_categories=subgroup_categories,
#         supergroup_variable_name=supergroup_variable_name,
#         subgroup_variable_names=subgroup_variable_names,
#     )

#     subgroup_sizes = {
#         ("Low", "Region1"): 40,
#         ("High", "Region1"): 60,
#         ("Low", "Region2"): 80,
#         ("High", "Region2"): 120,
#     }
#     group_map.add_attribute(
#         group_type="subgroup",
#         attribute_name="size",
#         attribute_values=subgroup_sizes,
#         impute_action="ignore",
#         attribute_class=Attribute,
#     )

#     subgroup_cases = {
#         ("Low", "Region1"): 4.0,
#         ("High", "Region1"): 6.0,
#         ("Low", "Region2"): 20.0,
#         ("High", "Region2"): 30.0,
#     }
#     group_map.add_attribute(
#         group_type="subgroup",
#         attribute_name="cases",
#         attribute_values=subgroup_cases,
#         impute_action="impute",
#         attribute_class=ImputableAttribute,
#         measurement_type="count",
#     )

#     vax_rate_sub = {
#         ("Low", "Region1"): 0.5,
#         ("High", "Region1"): 0.5,
#         ("Low", "Region2"): 0.8,
#         ("High", "Region2"): 0.8,
#     }
#     group_map.add_attribute(
#         group_type="subgroup",
#         attribute_name="vaccination_rate",
#         attribute_values=vax_rate_sub,
#         impute_action="impute",
#         attribute_class=ImputableAttribute,
#         measurement_type="rate",
#     )

#     collection_date_sub = {
#         ("Low", "Region1"): "2024-01-01",
#         ("High", "Region1"): "2024-01-01",
#         ("Low", "Region2"): "2024-01-02",
#         ("High", "Region2"): "2024-01-02",
#     }
#     group_map.add_attribute(
#         group_type="subgroup",
#         attribute_name="collection_date",
#         attribute_values=collection_date_sub,
#         impute_action="copy",
#         attribute_class=Attribute,
#     )

#     notes_sub = {
#         ("Low", "Region1"): "foo",
#         ("High", "Region1"): "bar",
#         ("Low", "Region2"): "baz",
#         ("High", "Region2"): "qux",
#     }
#     group_map.add_attribute(
#         group_type="subgroup",
#         attribute_name="notes",
#         attribute_values=notes_sub,
#         impute_action="ignore",
#         attribute_class=Attribute,
#     )

#     aggregator = Aggregator()
#     result = aggregator(group_map)

#     expected_cases = {"Region1": 10, "Region2": 50}
#     expected_vax_rates = {"Region1": 0.5, "Region2": 0.8}
#     expected_collection_dates = {
#         "Region1": "2024-01-01",
#         "Region2": "2024-01-02",
#     }

#     for region in ["Region1", "Region2"]:
#         assert (
#             result.group(region).get_attribute("cases").value
#             == expected_cases[region]
#         )
#         assert (
#             result.group(region).get_attribute("vaccination_rate").value
#             == expected_vax_rates[region]
#         )
#         assert (
#             result.group(region).get_attribute("collection_date").value
#             == expected_collection_dates[region]
#         )

#     assert all(
#         result.group(sg)._get_attribute("notes") is None
#         for sg in result.supergroup_names
#     )
