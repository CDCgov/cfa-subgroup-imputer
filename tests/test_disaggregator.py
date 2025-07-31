# import pytest
# from cfa_subgroup_imputer.groups import Group, GroupMap
# from cfa_subgroup_imputer.variables import Attribute, ImputableAttribute
# from cfa_subgroup_imputer.imputer import (
#     Disaggregator,
#     ProportionsFromCategories,
# )
# from cfa_subgroup_imputer.mapping import OuterProductSubgroupHandler


# def test_disaggregator_outer_product():
#     # Define two supergroups and three subgroups
#     supergroups = ["A", "B"]
#     subgroups = ["x", "y", "z"]
#     # Assign sizes to each subgroup within each supergroup
#     subgroup_sizes = {
#         ("A", "x"): 10,
#         ("A", "y"): 20,
#         ("A", "z"): 30,
#         ("B", "x"): 40,
#         ("B", "y"): 50,
#         ("B", "z"): 60,
#     }
#     supergroup_sizes = {
#         "A": sum(subgroup_sizes[("A", s)] for s in subgroups),
#         "B": sum(subgroup_sizes[("B", s)] for s in subgroups),
#     }

#     # Build group map using OuterProductSubgroupHandler
#     handler = OuterProductSubgroupHandler()
#     group_map = handler.construct_group_map(
#         supergroups=supergroups,
#         subgroups=subgroups,
#         supergroup_varname="region",
#         subgroup_varname="category",
#     )

#     # Add size attributes to groups
#     groups = []
#     for group in group_map.groups.values():
#         if group.name in supergroups:
#             # Supergroup
#             groups.append(
#                 Group(
#                     name=group.name,
#                     attributes=[
#                         Attribute(
#                             value=group.name,
#                             name="region",
#                             impute_action="ignore",
#                         ),
#                         Attribute(
#                             value=supergroup_sizes[group.name],
#                             name="size",
#                             impute_action="ignore",
#                         ),
#                     ],
#                 )
#             )
#         else:
#             # Subgroup
#             region = group.get_attribute("region").value
#             category = group.get_attribute("category").value
#             groups.append(
#                 Group(
#                     name=group.name,
#                     attributes=[
#                         Attribute(
#                             value=region, name="region", impute_action="ignore"
#                         ),
#                         Attribute(
#                             value=category,
#                             name="category",
#                             impute_action="ignore",
#                         ),
#                         Attribute(
#                             value=subgroup_sizes[(region, category)],
#                             name="size",
#                             impute_action="ignore",
#                         ),
#                     ],
#                 )
#             )
#     group_map = GroupMap(group_map.sub_to_super, groups)

#     # Mark group map as disaggregatable by monkeypatching if needed
#     # (Assume for this test that disaggregatable is not implemented)
#     group_map.disaggregatable = True

#     # Run Disaggregator
#     disagg = Disaggregator(ProportionsFromCategories(size_from="size"))
#     result = disagg(group_map)

#     # Check that each subgroup in result has the correct size
#     for supergroup in supergroups:
#         total = 0
#         for subgroup in subgroups:
#             name = f"{supergroup}_{subgroup}"
#             g = result.group(name)
#             size = g.get_attribute("size").value
#             assert size == pytest.approx(
#                 subgroup_sizes[(supergroup, subgroup)]
#             )
#             total += size
#         # Supergroup size should match sum of subgroups
#         super_size = result.group(supergroup).get_attribute("size").value
#         assert super_size == pytest.approx(total)
