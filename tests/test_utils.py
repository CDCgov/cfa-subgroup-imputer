import pytest

import cfa_subgroup_imputer.utils as utils


def test_conversion():
    keys = ["code", 1, 2]
    xdict = {
        1: "A",
        2: "B",
        "code": 1,
    }
    xtup = (
        1,
        "A",
        "B",
    )

    assert utils._dict_to_tuple(xdict, keys) == xtup
    assert utils._tuple_to_dict(xtup, keys) == xdict


def test_key_getters():
    good = [{"foo": "bar", "alpha": "beta"}, {"alpha": "gamma", "foo": "baz"}]
    ragged = [
        {"foo": "bar", "alpha": "beta"},
        {"foo": "baz", "alpha": "gamma", "foxtrot": "echo"},
    ]
    nonstr = [{1: "bar", "alpha": "beta"}, {"alpha": "gamma", 1: "baz"}]

    assert utils.get_keys(good) == ["foo", "alpha"]

    with pytest.raises(Exception):
        _ = utils.get_keys(ragged)

    with pytest.raises(Exception):
        _ = utils.get_json_keys(nonstr)


def test_select():
    rows = [
        {"foo": "bar", "alpha": "beta", "foxtrot": "echo"},
    ]
    assert utils.select(rows, ["foo", "foxtrot"]) == [
        {"foo": "bar", "foxtrot": "echo"},
    ]


def test_unique():
    expected_rows = [
        {"foo": "bar", "alpha": "beta", "foxtrot": "echo"},
        {"foo": "bar", "alpha": "beta", "foxtrot": "echo2"},
    ]
    rows = expected_rows * 2
    unique_rows = utils.unique(rows)
    assert len(unique_rows) == 2
    for er in expected_rows:
        assert er in unique_rows
    assert utils.unique(rows, select=["foo", "alpha"]) == [
        {
            "foo": "bar",
            "alpha": "beta",
        }
    ]
