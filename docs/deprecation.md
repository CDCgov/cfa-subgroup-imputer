# Rationale for deprecation

We identified a ["simple" problem space](#simple-problem-space) for which the methods in this codebase was well-suited, and at least one [more difficult problem](#difficult-problems) that had many components that this codebase would not simplify.

We concluded that a rigorous treatment of age groups and a robust age group parser would be helpful, leading to [nascor](https://github.com/CDCgov/nascor). However, the "simple" problems were sufficiently simple that a codebase was of unclear utility, while the "difficult" problem had subject matter-specific problems that would be out of scope for a subgroup imputer _per se_.

Should this project be revisited, we have attempted to distill the lessons learned in this implementation which will be relevant.

## Simple problem space

"Subgroup imputation" can refer to many different problems. Our team identified problems we faced in practice:

- Given hospitalization rates for large age groups (e.g., 0-4 years), impute hospitalization rates for smaller age groups (e.g., 0-<1 years), assuming that hospitalization rates are:
  - identical within each large age group (and therefore also within small age groups), or
  - identical within each small age group (but vary linearly across those small age groups).
- Ditto, but for vaccination coverage rates.
- Ditto, but for _counts_ of hospitalizations.
- Ditto, but for geographies (e.g., counties inside states).
- Given population sizes of individuals in large age ranges (e.g., 18-29 year olds), compute the population sizes of individuals in age ranges which subdivide these (e.g. 18, 19, 20, ... 29 year olds) assuming that the "population pyramid" is flat/uniform/identical within each large age group.

These are all problems where we can straightforwardly and immediately think of allocating the individuals in the large group proportional to the sizes of the small subgroups, and as such can be addressed by the framework discussed in [this package](index.md).

## Difficult problems

Our most difficult problem was the "RSV adults problem" of estimating vaccine-prevented burden of RSV among adults. In practice, this problem required:

1. Starting from "large" Census age groups, infer the size of "small" age group cohorts.
2. Divide "small" age groups into risk levels. For some age groups, the recommendation for RSV vaccination was universal; for other age groups, the recommendation was only for certain "high risk" individuals.
3. Divide hospitalization data, reported for "large" age groups (that differ from those in the Census), and split those into hospitalization rates for smaller age groups, accounting for the risk levels.
4. Divide vaccination data, reported for (yet another set of) "large" age groups, into vaccination rates for the risk-stratified, "small" age groups.
5. Infer relative risks of hospitalization in the absence of vaccination. (This requires information about vaccine efficacy.)
6. Compute relative risks of hospitalization in the presence of vaccination.

This problem involves multiple, interacting problems that are outside the scope of subgroup imputation _per se_ and require further elements of theory.

## Implementation lessons

We have attempted to identify some of the key computational lessons learned, both things done right and wrong in this code base.

The general spec here is that the input to the imputer is a dataframe (in some form) on supergroups, the output is a dataframe on subgroups.

- **Separate _mapping_ from _imputing_.** There is always a 1:1 mapping of subgroup to supergroup, which is based on some number of variables (e.g., age). This should be done _first_ and imputation should be done _conditional_ on that mapping. This allows for the same downstream code to handle a wide range of cases, including splitting population sizes based on the continuous variable age, and splitting vaccination rates up based on population sizes.
- **Separate the computation of imputation _weights_ from both _mapping_ and _imputing_**. Regardless of whether all subgroups are assumed to, e.g. have the same rate of vaccination or that it's a linear function of age, once we know the mapping, we have a problem which we can frame as proportional allocation of supergroup counts to subgroup counts. If we separate out the computation of what those proportions _are_ from actually doing the distribution, age groups and vaccination rates and such can be handled with the same downstream code.
- **Separate group _access_ from group _properties_.**
  - Short version: _all_ sub and super groups should have _unique identifiers_, even if this leads to duplication of groups.
  - Long version. Each subgroup and supergroup needs a unique identifier, independent of its defining properties, to use as an accesser/filterer. You will always have the mapping to fall back to, so you can always find the properties of a subgroup as needed to work with _from the unique identifier_. Consider the case where a subgroup comprises its entire supergroup, for example imputing populations for age ranges and you have the group "21 year olds" in both sub and super groups. If you treat the supergroup 21 year olds as distinct from the subgroup 21 year olds, each with their own identifier, you can simply copy all relevant information from the supergroup to the subgroup. If you try to have only one group for 21 year olds, it will be a tangle of exceptions and sharp edges.
- **A group is defined by a tuple of variable values.** When performing county-level imputation, for example, the supergroups are defined by the single variable `(state=some_state)` while the subgroups are defined by the _pair_ of variables `(state=some_state, county=some_county)`. Age ranges have the same single variable `(low <= age < high)` for both subgroups and supergroups.
- **Much of the value of a good UI is being able to automatically construct subgroups.** When all subgroups are in all supergroups, and we are willing to assume that the relative proportions are the same, the user should be able to pass in less information than when this is not true. For example, if we assume that all states have the same age pyramid, we should be able to feed in state-level data, compact information about the age subgroups we want, and there should be code that automatically expands this to all `(state, age_subgroup)` and `(state, age_supergroup)` combinations. This is different from, say, county-level imputation, where the user will have to feed in data on the population sizes of _all counties_ in _all states_, or any other case where we are unwilling to assume homogeneity/independence of the variable(s) we're using to define subgroups from the variable(s) we're using to define supergroups.
- **Complexity quickly compounds.** As implemented, the codebase is designed to split one "level" of subgroups. Going from state to county level, or from large age groups to small age groups, on the assumption of uniformity/homogeneity of rates in the underlying subgroups. Much more complex imputation is possible, but the statespace of the problem rapidly explodes, and even carefully delimited scope will lead to increasingly complex code for the UI and increasing numbers of intermediate functions. Generalization should only be undertaken when a very specific goal and direction proves necessary.
