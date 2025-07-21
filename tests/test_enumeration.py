import math

from cfa_subgroup_imputer.enumerator import AgeGroupEnumerator


class TestAgeGroupEnumerator:
    def test_ranges(self):
        assert AgeGroupEnumerator().age_range_from_str(
            "6 months-4 years"
        ).to_tuple() == (
            0.5,
            5.0,
        )

        assert AgeGroupEnumerator().age_range_from_str(
            "6-23 months"
        ).to_tuple() == (
            0.5,
            2.0,
        )

        assert AgeGroupEnumerator().age_range_from_str(
            "42 years"
        ).to_tuple() == (
            41.0,
            42.0,
        )

        assert AgeGroupEnumerator().age_range_from_str(
            "60+ years"
        ).to_tuple() == (
            60,
            math.inf,
        )

        assert AgeGroupEnumerator().age_range_from_str(
            "50-64 years"
        ).to_tuple() == (
            50.0,
            65.0,
        )

        assert AgeGroupEnumerator().age_range_from_str(
            "0-<1 year"
        ).to_tuple() == (
            0.0,
            1.0,
        )
