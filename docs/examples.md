# Usage examples

The following examples of python-based package use are meant to be worked through in order.

## ACS population size data

First, we will employ the following example data, which is a slightly reformatted version of ACS data from data.census.gov, in the `list`-of-`dict`s format used by the package.
The data contains population sizes (the `"value"` key in the dictionaries) for males and females broken down by age ranges in the United States.

```python
acs = [{"sex": "Male", "age": "0-4 years", "value": 9373156}, {"sex": "Male", "age": "5-9 years", "value": 10136158}, {"sex": "Male", "age": "10-14 years", "value": 10877744}, {"sex": "Male", "age": "15-17 years", "value": 6849276}, {"sex": "Male", "age": "18-19 years", "value": 4515196}, {"sex": "Male", "age": "20 years", "value": 2208029}, {"sex": "Male", "age": "21 years", "value": 2179369}, {"sex": "Male", "age": "22-24 years", "value": 6686561}, {"sex": "Male", "age": "25-29 years", "value": 11090992}, {"sex": "Male", "age": "30-34 years", "value": 11811352}, {"sex": "Male", "age": "35-39 years", "value": 11437479}, {"sex": "Male", "age": "40-44 years", "value": 11100697}, {"sex": "Male", "age": "45-49 years", "value": 9893835}, {"sex": "Male", "age": "50-54 years", "value": 10297208}, {"sex": "Male", "age": "55-59 years", "value": 9929586}, {"sex": "Male", "age": "60-61 years", "value": 4263864}, {"sex": "Male", "age": "62-64 years", "value": 6279212}, {"sex": "Male", "age": "65-66 years", "value": 3801601}, {"sex": "Male", "age": "67-69 years", "value": 5224212}, {"sex": "Male", "age": "70-74 years", "value": 7367430}, {"sex": "Male", "age": "75-79 years", "value": 5129494}, {"sex": "Male", "age": "80-84 years", "value": 3013517}, {"sex": "Male", "age": "85+ years", "value": 2263405}, {"sex": "Female", "age": "0-4 years", "value": 8960541}, {"sex": "Female", "age": "5-9 years", "value": 9663272}, {"sex": "Female", "age": "10-14 years", "value": 10326135}, {"sex": "Female", "age": "15-17 years", "value": 6462154}, {"sex": "Female", "age": "18-19 years", "value": 4341764}, {"sex": "Female", "age": "20 years", "value": 2077086}, {"sex": "Female", "age": "21 years", "value": 2078366}, {"sex": "Female", "age": "22-24 years", "value": 6388972}, {"sex": "Female", "age": "25-29 years", "value": 10815714}, {"sex": "Female", "age": "30-34 years", "value": 11593704}, {"sex": "Female", "age": "35-39 years", "value": 11212620}, {"sex": "Female", "age": "40-44 years", "value": 11025788}, {"sex": "Female", "age": "45-49 years", "value": 9965395}, {"sex": "Female", "age": "50-54 years", "value": 10364733}, {"sex": "Female", "age": "55-59 years", "value": 10268922}, {"sex": "Female", "age": "60-61 years", "value": 4464936}, {"sex": "Female", "age": "62-64 years", "value": 6668024}, {"sex": "Female", "age": "65-66 years", "value": 4176420}, {"sex": "Female", "age": "67-69 years", "value": 5824728}, {"sex": "Female", "age": "70-74 years", "value": 8430427}, {"sex": "Female", "age": "75-79 years", "value": 6189257}, {"sex": "Female", "age": "80-84 years", "value": 4027902}, {"sex": "Female", "age": "85+ years", "value": 3858663}]
```

### Aggregation

To get the total number of males and females across all age groups, we can aggregate as follows

```python
from cfa_subgroup_imputer.json import aggregate

pop_by_sex = aggregate(
    supergroup_data=[{"sex": "Male"}, {"sex": "Female"}],
    subgroup_data=acs,
    subgroup_to_supergroup=None,
    supergroups_from="sex",
    subgroups_from="age",
    group_type="categorical",
    count="value",
    size_from="value",
)
```

To get the total number of males and females combined in each age group, we can aggregate as follows

```python
ages = set([row["age"] for row in acs])

pop_by_age = aggregate(
    supergroup_data=[{"age": age} for age in ages],
    subgroup_data=acs,
    subgroup_to_supergroup=None,
    supergroups_from="age",
    subgroups_from="sex",
    group_type="categorical",
    count="value",
    size_from="value",
)
```

### Disaggregation

We can use [uniform density continuous disaggregation](index.md#uniform-density-continuous-disaggregation) to disaggregate the age-only data to yearly age groups.

```python
pop_by_year = [{"age": f"{i} year"} for i in range(100)]
disaggregate(
    supergroup_data=pop_by_age,
    subgroup_data=yearly_age_data,
    subgroup_to_supergroup=None,
    group_type="age",
    supergroups_from="age",
    subgroups_from="age",
    count="value",
    size_from="value",
)
```

## Vaccine administration

The following is a slightly reformatted excerpt of national vaccination uptake data from NIS (sw5n-wg2p), providing the estimated vaccine coverage for several weeks in October 2023, broken down by age groups.
```
[{'age': '50-64 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.21320050000000001}, {'age': '18+ years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.166}, {'age': '60+ years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.251}, {'age': '65+ years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.409}, {'age': '18-29 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.18600000000000003}, {'age': '30-39 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.11599999999999999}, {'age': '18-29 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.149}, {'age': '40-49 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.17500000000000002}, {'age': '65-74 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.368}, {'age': '65+ years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.449}, {'age': '18-49 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.156}, {'age': '30-39 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.138}, {'age': '50-64 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.255132}, {'age': '60+ years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.314}, {'age': '18-49 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.201}, {'age': '18-49 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.18300000000000002}, {'age': '18-29 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.166}, {'age': '40-49 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.142}, {'age': '65+ years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.35100000000000003}, {'age': '75+ years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.511}, {'age': '65-74 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.258}, {'age': '65-74 years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.326}, {'age': '18+ years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.247}, {'age': '18+ years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.21}, {'age': '75+ years', 'time_start': '2023-10-08', 'time_end': '2023-10-14', 'estimate': 0.391}, {'age': '40-49 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.20800000000000002}, {'age': '18-49 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.125}, {'age': '75+ years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.47200000000000003}, {'age': '75+ years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.312}, {'age': '65-74 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.408}, {'age': '50-64 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.162996}, {'age': '18-29 years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.114}, {'age': '50-64 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.28312550000000003}, {'age': '60+ years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.37}, {'age': '30-39 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.18300000000000002}, {'age': '65+ years', 'time_start': '2023-10-01', 'time_end': '2023-10-07', 'estimate': 0.28}, {'age': '60+ years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.408}, {'age': '40-49 years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.228}, {'age': '30-39 years', 'time_start': '2023-10-15', 'time_end': '2023-10-21', 'estimate': 0.16899999999999998}, {'age': '18+ years', 'time_start': '2023-10-22', 'time_end': '2023-10-28', 'estimate': 0.272}]
```
