from cfa_subgroup_imputer.variables import Range


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
