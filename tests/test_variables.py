import pytest

from cfa_subgroup_imputer.variables import (
    Attribute,
    ImputableAttribute,
    Range,
    assert_range_spanned_exactly,
)


class TestAttribute:
    def test_constructor(self):
        _ = Attribute(
            name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
        )

        _ = Attribute(name="Outis", value=None, impute_action="ignore")

        with pytest.raises(Exception):
            _ = Attribute(
                name="Outis", value=[], impute_action="invalid option"
            )

    def test_eq(self):
        assert Attribute(
            name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
        ) == Attribute(
            name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
        )
        assert ImputableAttribute(
            name="attribute",
            value=42,
            impute_action="impute",
            measurement_type="rate",
        ) == ImputableAttribute(
            name="attribute",
            value=42,
            impute_action="impute",
            measurement_type="rate",
        )
        with pytest.raises(Exception):
            assert Attribute(
                name="Ourobouros",
                value=[dict(), tuple(), ""],
                impute_action="copy",
            ) == Attribute(
                name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
            )
        with pytest.raises(Exception):
            assert Attribute(
                name="Outis",
                value=[dict(), tuple()],
                impute_action="copy",
            ) == Attribute(
                name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
            )
        with pytest.raises(Exception):
            assert Attribute(
                name="Outis",
                value=[dict(), tuple(), ""],
                impute_action="ignore",
            ) == Attribute(
                name="Outis", value=[dict(), tuple(), ""], impute_action="copy"
            )


def test_range():
    one_ten = Range(1, 10)
    one_two = Range(1, 2)
    two_three = Range(2, 3)
    five_twelve = Range(5, 12)

    assert one_two != one_ten

    assert one_two in one_ten
    assert two_three in one_ten
    assert five_twelve not in one_ten

    assert two_three > one_two
    assert not (one_two < one_ten)
    assert not (one_two > one_ten)

    assert sorted([five_twelve, one_two, two_three]) == [
        one_two,
        two_three,
        five_twelve,
    ]

    assert Range.from_tuple(one_ten.to_tuple()) == one_ten


def test_range_span():
    assert_range_spanned_exactly(Range(1, 10), [Range(1, 10)])
    assert_range_spanned_exactly(
        Range(1, 10), [Range(1, 2), Range(2, 3), Range(3, 10)]
    )
    with pytest.raises(Exception):
        assert_range_spanned_exactly(
            Range(1.1, 10), [Range(1, 2), Range(2, 3), Range(3, 10)]
        )

    with pytest.raises(Exception):
        assert_range_spanned_exactly(
            Range(1, 10), [Range(1, 2), Range(2, 3), Range(3, 10.1)]
        )

    with pytest.raises(Exception):
        assert_range_spanned_exactly(Range(1, 10), [Range(1, 2), Range(3, 10)])
