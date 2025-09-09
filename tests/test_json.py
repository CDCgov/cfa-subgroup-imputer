import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.json import aggregate, create_group_map, disaggregate
from cfa_subgroup_imputer.variables import Attribute


@pytest.fixture
def three_counties_sub_super_map():
    return [
        {"state": "California", "county": "Sutter"},
        {"state": "Washington", "county": "Skagit"},
        {"state": "Washington", "county": "San Juan"},
    ]


@pytest.fixture
def three_counties(three_counties_sub_super_map):
    return create_group_map(
        supergroup_data=None,
        subgroup_defs=None,
        subgroup_to_supergroup=three_counties_sub_super_map,
        supergroups_from="state",
        subgroups_from="county",
        group_type=None,
    )


@pytest.fixture
def state_data():
    return [
        {
            "state": "California",
            "size": 40,
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 10,
            "to_exclude": "wont",
            "to_ignore": "willbe",
        },
        {
            "state": "Washington",
            "size": 8,
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 20,
            "to_exclude": "see",
            "to_ignore": "ignored",
        },
    ]


@pytest.fixture
def age_group_data():
    return [
        {
            "age_group": "0-17 years",
            "size": 1800,
            "cases": 180,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
            "notes": "young",
            "to_exclude": "skip1",
        },
        {
            "age_group": "18+ years",
            "size": 8200,
            "cases": 820,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
            "notes": "adult",
            "to_exclude": "skip2",
        },
    ]


@pytest.fixture
def age_subgroups():
    return [
        {"age_group": "0-4 years"},
        {"age_group": "5-17 years"},
        {"age_group": "18-64 years"},
        {"age_group": "65+ years"},
    ]


def test_groups_from_dicts(three_counties):
    map_expected = {
        ("Sutter", "California"): "California",
        ("Skagit", "Washington"): "Washington",
        ("San Juan", "Washington"): "Washington",
    }

    groups_expected = {
        "California": Group(
            name="California",
            attributes=[
                Attribute(
                    value="California",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        "Washington": Group(
            name="Washington",
            attributes=[
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("Sutter", "California"): Group(
            name=("Sutter", "California"),
            attributes=[
                Attribute(
                    value="Sutter",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="California",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("Skagit", "Washington"): Group(
            name=("Skagit", "Washington"),
            attributes=[
                Attribute(
                    value="Skagit",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("San Juan", "Washington"): Group(
            name=("San Juan", "Washington"),
            attributes=[
                Attribute(
                    value="San Juan",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
    }

    assert three_counties.groups == groups_expected
    assert three_counties.sub_to_super == map_expected


def test_data_io_categorical(three_counties, state_data):
    three_counties.data_from_dicts(
        state_data,
        "supergroup",
        exclude=["to_exclude", "size"],
        count=["some_count"],
        rate=["some_rate"],
        copy=["flower"],
    )

    dicts = three_counties.to_dicts("supergroup")
    expected_dicts = [
        {
            "state": "California",
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 10,
            "to_ignore": "willbe",
        },
        {
            "state": "Washington",
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 20,
            "to_ignore": "ignored",
        },
    ]

    assert dicts == expected_dicts


def test_data_io_age_groups(age_subgroups, age_group_data):
    age_group_map = create_group_map(
        supergroup_data=age_group_data,
        subgroup_defs=age_subgroups,
        subgroup_to_supergroup=None,
        supergroups_from="age_group",
        subgroups_from="age_group",
        group_type="age",
    )

    age_group_map.data_from_dicts(
        age_group_data,
        "supergroup",
        exclude=["to_exclude", "notes"],
        count=["cases", "size"],
        rate=["vaccination_rate"],
        copy=["collection_date"],
    )

    dicts = age_group_map.to_dicts("supergroup")
    expected_dicts = [
        {
            "age_group": "0-17 years",
            "size": 1800,
            "cases": 180,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "18+ years",
            "size": 8200,
            "cases": 820,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
        },
    ]
    assert dicts == expected_dicts


def test_disagg_categorical(state_data):
    subgroup_defs = [
        {"state": "California", "splitvar": "cat1", "size": 20},
        {"state": "California", "splitvar": "cat2", "size": 20},
        {"state": "Washington", "splitvar": "cat1", "size": 2},
        {"state": "Washington", "splitvar": "cat2", "size": 6},
    ]

    disagg = disaggregate(
        supergroup_data=state_data,
        subgroup_defs=subgroup_defs,
        subgroup_to_supergroup=None,
        supergroups_from="state",
        subgroups_from="splitvar",
        group_type="categorical",
        loop_over=[],
        rate=["some_rate"],
        count=["some_count"],
        exclude=["to_exclude", "to_ignore"],
    )

    expected_disagg = [
        {
            "splitvar": "cat1",
            "state": "California",
            "size": 20,
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 5.0,
        },
        {
            "splitvar": "cat2",
            "state": "California",
            "size": 20,
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 5.0,
        },
        {
            "splitvar": "cat1",
            "state": "Washington",
            "size": 2,
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 5.0,
        },
        {
            "splitvar": "cat2",
            "state": "Washington",
            "size": 6,
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 15.0,
        },
    ]

    for od, ed in zip(disagg, expected_disagg):
        assert od == pytest.approx(ed)


def test_disagg_continuous_age(age_group_data, age_subgroups):
    disagg = disaggregate(
        supergroup_data=age_group_data,
        subgroup_defs=age_subgroups,
        subgroup_to_supergroup=None,
        supergroups_from="age_group",
        subgroups_from="age_group",
        group_type="age",
        loop_over=[],
        rate=["vaccination_rate"],
        count=["cases", "size"],
        copy=["collection_date"],
        exclude=["notes", "to_exclude"],
    )

    expected_disagg = [
        {
            "age_group": "0-4 years",
            "size": 500.0,
            "cases": 50.0,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "5-17 years",
            "size": 1300.0,
            "cases": 130.0,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "18-64 years",
            "size": 4700.0,
            "cases": 470.0,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "65+ years",
            "size": 3500.0,
            "cases": 350.0,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
        },
    ]

    for od, ed in zip(disagg, expected_disagg):
        assert od == pytest.approx(ed)


def test_agg_categorical(state_data):
    supergroup_data = [
        {
            "state": "California",
            "size": 40,
        },
        {
            "state": "Washington",
            "size": 8,
        },
    ]
    subgroup_data = [
        {
            "splitvar": "cat1",
            "state": "California",
            "size": 20,
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 5.0,
            "to_exclude": "foo",
            "to_ignore": "bar",
        },
        {
            "splitvar": "cat2",
            "state": "California",
            "size": 20,
            "flower": "Eschscholzia californica",
            "some_rate": 1.2,
            "some_count": 5.0,
            "to_exclude": "foz",
            "to_ignore": "baz",
        },
        {
            "splitvar": "cat1",
            "state": "Washington",
            "size": 2,
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 5.0,
            "to_exclude": "foo",
            "to_ignore": "bar",
        },
        {
            "splitvar": "cat2",
            "state": "Washington",
            "size": 6,
            "flower": "Rhododendron macrophyllum",
            "some_rate": 1.3,
            "some_count": 15.0,
            "to_exclude": "foz",
            "to_ignore": "baz",
        },
    ]

    agg = aggregate(
        supergroup_data=supergroup_data,
        subgroup_defs=subgroup_data,
        subgroup_to_supergroup=None,
        supergroups_from="state",
        subgroups_from="splitvar",
        group_type="categorical",
        loop_over=[],
        rate=["some_rate"],
        count=["some_count"],
        exclude=["to_exclude", "to_ignore"],
    )

    for od, ed in zip(agg, state_data):
        ed.pop("to_ignore")
        ed.pop("to_exclude")
        assert od == pytest.approx(ed)


def test_agg_continuous_age(age_group_data):
    supergroup_data = [
        {
            "age_group": "0-17 years",
        },
        {
            "age_group": "18+ years",
        },
    ]
    subgroup_data = [
        {
            "age_group": "0-4 years",
            "size": 500.0,
            "cases": 50.0,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "5-17 years",
            "size": 1300.0,
            "cases": 130.0,
            "vaccination_rate": 0.4,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "18-64 years",
            "size": 4700.0,
            "cases": 470.0,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
        },
        {
            "age_group": "65+ years",
            "size": 3500.0,
            "cases": 350.0,
            "vaccination_rate": 0.8,
            "collection_date": "2024-01-01",
        },
    ]

    agg = aggregate(
        supergroup_data=supergroup_data,
        subgroup_defs=subgroup_data,
        subgroup_to_supergroup=None,
        supergroups_from="age_group",
        subgroups_from="age_group",
        group_type="age",
        loop_over=[],
        rate=["vaccination_rate"],
        count=["cases", "size"],
        copy=["collection_date"],
    )

    for od, ed in zip(agg, age_group_data):
        ed.pop("notes")
        ed.pop("to_exclude")
        assert od == pytest.approx(ed)
