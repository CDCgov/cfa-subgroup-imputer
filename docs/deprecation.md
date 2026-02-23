# Rationale for deprecation

We identified a ["simple" problem space](#simple-problem-space) for which the methods in this codebase was well-suited, and at least one [more difficult problem](#difficult-problems) that had many components that this codebase would not simplify.

We concluded that a rigorous treatment of age groups and a robust age group parser would be helpful, leading to [nascor](https://github.com/CDCgov/nascor). However, the "simple" problems were sufficiently simple that a codebase was of unclear utility, while the "difficult" problem had subject matter-specific problems that would be out of scope for a subgroup imputer _per se_.

## Simple problem space

"Subgroup imputation" can refer to many different problems. Our team identified problems we faced in practice:

- Given hospitalization rates for large age groups (e.g., 0-4 years), impute hospitalization rates for smaller age groups (e.g., 0-<1 years), assuming that hospitalization rates are:
  - identical within each large age group (and therefore also within small age groups), or
  - identical within each small age group (but vary linearly across those small age groups).
- Ditto, but for vaccination coverage rates.
- Ditto, but for _counts_ of hospitalizations.
- Ditto, but sometimes the large age group and the small age group are the same (e.g., 0-<1 years is both a "large" and a "small" age group).
- Ditto, but for geographies (e.g., counties inside states).

These problems can be addressed by the framework discussed in [this package](index.md).

## Difficult problems

Our most difficult problem was the "RSV adults problem" of estimating vaccine-prevented burden of RSV among adults. In practice, this problem required:

1. Starting from "large" Census age groups, infer the size of "small" age group cohorts.
2. Divide "small" age groups into risk levels. For some age groups, the recommendation for RSV vaccination was universal; for other age groups, the recommendation was only for certain "high risk" individuals.
3. Divide hospitalization data, reported for "large" age groups (that differ from those in the Census), and split those into hospitalization rates for smaller age groups, accounting for the risk levels.
4. Divide vaccination data, reported for (yet another set of) "large" age groups, into vaccination rates for the risk-stratified, "small" age groups.
5. Infer relative risks of hospitalization in the absence of vaccination. (This requires information about vaccine efficacy.)
6. Compute relative risks of hospitalization in the presence of vaccination.

This problem involves multiple, interacting problems that are outside the scope of subgroup imputation _per se_ and require further elements of theory.
