from typing import Any, Sequence

import numpy as np
import polars as pl


def apportion(
    props: Sequence[float],
    values: Sequence[Any],
    n: int,
    rng: np.random.Generator,
) -> Sequence[Any]:
    assert sum(props) == 1.0
    ms = [round(p * n) for p in props]
    if sum(ms) != n:
        print(f"from {sum(ms)} to {n}")
        ms[-1] = n - sum(ms[:-1])

    assert sum(ms) == n

    out = [value for value, m in zip(values, ms) for _ in range(m)]
    rng.shuffle(out)

    assert len(out) == n

    return out


def make_pop(rng: np.random.Generator) -> pl.DataFrame:
    # initialize a population using the census data.
    # data: (lo age, hi age), # of people
    census_data = [
        ((0, 4), 1e4),
        ((5, 11), 1e4),
        ((12, 18), 1e4),
        ((19, 64), 6e4),
        ((65, 100), 1e4),
    ]

    # population is a list of individuals represented by dictionaries of attribute => value
    # I made the ages linearly spaced; could have drawn from a uniform too
    pop = [
        {"age": float(age)}
        for (lo, hi), size in census_data
        for age in np.linspace(lo, hi, num=int(size))
    ]

    # assign risk groups
    # data: (lo age, hi age), (% of age group in risk group, ), (risk group name, )
    p_high_risk = 0.25
    risk_data = [
        ((0, 50), (1.0,), ("baseline",)),
        ((50, 75), (1.0 - p_high_risk, p_high_risk), ("baseline", "high")),
        ((75, 100), (1.0,), ("baseline",)),
    ]

    for (lo, hi), props, risk_levels in risk_data:
        # find the indices of the people who match
        idx = [i for i, person in enumerate(pop) if lo <= person["age"] <= hi]

        # select the risk groups
        risks = apportion(props, risk_levels, len(idx), rng=rng)

        for i, risk in zip(idx, risks):
            pop[i]["risk"] = risk

    # assign vax status
    # data: (lo, hi age), risk, %vax
    vax_data = [
        ((0, 50), "baseline", 0.0),
        ((50, 75), "baseline", 0.0),
        ((50, 75), "high", 0.50),
        ((75, 100), "baseline", 0.25),
    ]

    for (lo, hi), risk, p_vax in vax_data:
        idx = [
            i
            for i, person in enumerate(pop)
            if lo <= person["age"] <= hi and person["risk"] == risk
        ]
        is_vaxs = apportion(
            (p_vax, 1.0 - p_vax), [True, False], n=len(idx), rng=rng
        )

        for i, is_vax in zip(idx, is_vaxs):
            pop[i]["is_vax"] = bool(is_vax)

    # I omit the hospitalization data, since judging how vaccination affects the
    # probability of hospitalization is out of scope for this imputer

    # put into a polars dataframe. many of these operations would have been
    # faster in polarverse.
    return pl.from_records(pop)


rng = np.random.default_rng()
pop = make_pop(rng)
print(pop)
