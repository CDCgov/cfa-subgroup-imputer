# """
# Module for polars interface.
# """

# from collections.abc import Collection
# from typing import Literal, Protocol, runtime_checkable

# import polars as pl

# from cfa_subgroup_imputer.groups import Group, GroupMap
# from cfa_subgroup_imputer.imputer import (
#     Disaggregator,
#     ProportionsFromCategories,
#     ProportionsFromContinuous,
# )
# from cfa_subgroup_imputer.mapping import (
#     AgeGroupHandler,
#     OuterProductSubgroupHandler,
# )
# from cfa_subgroup_imputer.variables import (
#     GroupableTypes,
# )


# @runtime_checkable
# class FilterConstructor(Protocol):
#     """
#     A Protocol that constructs filters for populating sub and supergroup data
#     from polars dataframes.
#     """

#     def construct_filter(
#         self,
#         supergroup_var: str,
#         supergroup_name: str,
#         subgroup_var: str,
#         subgroup_name: str,
#         group_type: Literal["subgroup", "supergroup"],
#     ) -> pl.Expr: ...


# def create_group_map(
#     supergroup_df: pl.DataFrame,
#     subgroup_df: pl.DataFrame,
#     subgroup_to_supergroup: pl.DataFrame | None,
#     supergroups_from: str,
#     subgroups_from: str,
#     group_type: GroupableTypes | None,
#     **kwargs,
# ) -> GroupMap:
#     if subgroup_to_supergroup is not None:
#         # arbitrary groups
#         pass

#     supergroups = supergroup_df[supergroups_from].unique().to_list()
#     subgroups = subgroup_df[subgroups_from].unique().to_list()
#     if group_type == "categorical":
#         return OuterProductSubgroupHandler().construct_group_map(
#             supergroups=supergroups, subgroups=subgroups
#         )
#     elif group_type == "age":
#         return AgeGroupHandler().construct_group_map(
#             supergroups=supergroups, subgroups=subgroups
#         )
#     else:
#         raise RuntimeError(f"Unknown grouping variable type {group_type}")


# def disaggregate(
#     supergroup_df: pl.DataFrame,
#     subgroup_df: pl.DataFrame,
#     subgroup_to_supergroup: pl.DataFrame | None,
#     supergroups_from: str,
#     subgroups_from: str,
#     group_type: GroupableTypes | None,
#     loop_over: Collection[str] = [],
#     **kwargs,
# ) -> pl.DataFrame:
#     """
#     Takes in a dataframe `df` with measurements for the `supergroups`.
#     Imputes values for the subgroups and returns a dataframe with those.

#     Parameters
#     ----------

#     Returns
#     -------
#     pl.DataFrame
#         Dataframe with measurements imputed for the subgroups.
#     """

#     group_map = create_group_map(
#         supergroup_df=supergroup_df,
#         subgroup_df=subgroup_df,
#         subgroup_to_supergroup=subgroup_to_supergroup,
#         supergroups_from=supergroups_from,
#         subgroups_from=subgroups_from,
#         group_type=group_type,
#         **kwargs,
#     )

#     # if loop_over:
#     #     supergroup_df_looper = [
#     #         dfg[1] for dfg in supergroup_df.group_by(loop_over)
#     #     ]
#     #     # Broadcast static subgroup data up, or else make the list
#     #     if all(var in subgroup_df.columns for var in loop_over):
#     #         subgroup_df_list = [
#     #             dfg[1] for dfg in subgroup_df.group_by(loop_over)
#     #         ]
#     #     else:
#     #         subgroup_df_list = [subgroup_df] * len(supergroup_df_looper)
#     # else:
#     #     supergroup_df_list = [supergroup_df]
#     #     subgroup_df_list = [subgroup_df]

#     # assert len(supergroup_df_list) == len(subgroup_df_list)
#     # return pl.concat(
#     #     [
#     #         _disaggregate(
#     #             super_df, sub_df, sub_to_super, group_type, **kwargs
#     #         )
#     #         for super_df, sub_df in zip(supergroup_df_list, subgroup_df_list)
#     #     ]
#     # )


# # def _disaggregate(
# #     supergroup_df: pl.DataFrame,
# #     subgroup_df: pl.DataFrame,
# #     subgroup_to_supergroup: dict[str, str],
# #     group_vartype: GroupingVariable,
# #     **kwargs,
# # ) -> pl.DataFrame:
# #     """
# #     Internal disaggregation function for processed inputs.
# #     """
# #     all_groups = [
# #         Group(name=group_name)
# #         for group_name in list(set(subgroup_to_supergroup.values()))
# #         + list(subgroup_to_supergroup.keys())
# #     ]

# #     map = GroupMap(subgroup_to_supergroup, all_groups)
# #     map.add_data_from_polars(supergroup_df)
# #     map.add_data_from_polars(subgroup_df)

# #     weight_calculator = (
# #         ProportionsFromCategories(group_vartype.name)
# #         if group_vartype.type == "Categorical"
# #         else ProportionsFromContinuous()
# #     )
# #     return Disaggregator(weight_calculator)(map).data_to_polars("subgroup")
