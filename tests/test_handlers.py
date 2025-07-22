import math

from cfa_subgroup_imputer.mapping import AgeGroupHandler


class TestAgeGroups:
    def test_ranges(self):
        assert AgeGroupHandler().age_range_from_str(
            "6 months-4 years"
        ).to_tuple() == (
            0.5,
            5.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "6-23 months"
        ).to_tuple() == (
            0.5,
            2.0,
        )

        assert AgeGroupHandler().age_range_from_str("42 years").to_tuple() == (
            41.0,
            42.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "60+ years"
        ).to_tuple() == (
            60,
            math.inf,
        )

        assert AgeGroupHandler().age_range_from_str(
            "50-64 years"
        ).to_tuple() == (
            50.0,
            65.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "0-<1 year"
        ).to_tuple() == (
            0.0,
            1.0,
        )


# class TestCategoroical:
#     def test_constructor():
#         supergroups = ["festina", "festimus materia"]
#         subgroups = ["lente", "velociter"]

#         map_expected = {
#             "festina_lente": "festina",
#             "festina_velociter": "festina",
#             "festimus materia_lente": "festimus materia",
#             "festimus materia_velociter": "festimus materia",
#         }

#         groups_expected = {
#             "lente": Group(name="lente", attributes = []),
#             "velociter":,
#             "festina_lente": Group(name=),
#             "festina_velociter": ,
#             "festimus materia_lente": ,
#             "festimus materia_velociter": ,
#         }
