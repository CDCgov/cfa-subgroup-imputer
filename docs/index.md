# cfa-subgroup-imputer

This is a package for working with [groups](#groups), specifically for disaggregating values from supergroups to subgroups, and, eventually, for the reverse operation, aggregating values from subgroups to supergroups.
The scope of disaggregatable variables is discussed [below](#imputable-values), but, roughly, is any variable which can be expressed as either a count or a ratio of counts per something.
It is focused on disaggregating subgroups of homogeneous supergroups, though extensions are possible to non-homogenous cases when the source of heterogeneity, and its effect, are known and available in the data.
This package does not infer models for disaggregating.


## Preliminary notes on terminology and assumptions

⚠️ This notation should not be considered finalized.
If you see something you think is wrong, bad, or unwise, open a PR with an explanation of the problem and a proposed solution.

### Groups

For our purposes, a group is an arbitrary subpopulation (possibly the entire population).
In particular, we are thinking about groups of people, so while the mathematical presentation is general, the scope may in practice be somewhat more restricted.
A group becomes a subgroup or supergroup only in relation to other groups.
That is, 10 year olds is both a subgroup of children under 18, and a supergroup of children aged 10 years and 0 months, 10 years and 1 month, ans so on.

We will assume that subgroups provided comprise the entire supergroup.
That is, that there won't be a supergroup of children under 18 with subgroups 1-3 year olds, 4-11 year olds, and 12-17 year olds, as this is missing infants less than one year old.
🚧 We will provide some functionality for filling these groups in, under simple assumptions, and regardless of such padding, validating completeness.

### Aggregating and disaggregating

The primary use of `cfa-subgroup-imputer` is _disaggregation_.
We have values for some supergroup, such as counts of vaccine doses in children under age 18, and we want to have values for subgroups thereof.
Generally we will want to disaggregate multiple supergroups simultaneously, such as moving from one set of groups to another, which is supported by the package.
The package supports disaggregating many variables simultaneously.

The package will eventually also support _aggregation_.
In this case, we combine measurements for subgroups into a supergroup.

### Data

Groups may have arbitrary forms of data associated with them,
However, as stated, the focus of this package is on disaggregating values which reflect in one some sense, in form or another, actual counts in groups.
Handling of other values is done much more simply.

We formalize this with a class hierarchy.
- An `Attribute` is essentially a tuple of:
  - A `name` specifying what this is (e.g. corresponding to the column name in a spreadsheet).
  - A `value` which can be anything.
  - A choice of `impute_action`, either:
    - `"copy"`, specifying that the variable should be copied directly from supergroups to their subgroups.
    - `"ignore"`, specifying that the variable should not be propagated to subgroups.
- An `ImputableAttribute` is a special case where:
  - The `value` is numeric and nonnegative.
  - The `impute_action` can additionally be `"impute"` specifying that this value should be disaggregated.
  - A `MeasurementType` is specified, tracking whether this is a rate-like or count-like quantity. This is discussed more [below](#imputable-values).

#### Imputable values

Values which can be imputed are one of two types.

Count-like attributes are distributed proportionately to subgroups.
For example, if we had the count of vaccinated individuals in the supergroup as the attribute, then each subgroup gets assigned a proportion of this total, according to [some model](#what-is-subgroup-imputation-anyways).
Quantities that fall into this category are:
- The size of the group itself, that is, the number of people (which [can be imputed, if needed](#a-special-case-when)).
- Hospitalization, infection, or case counts.
- Counts of vaccinated individuals.

A rate-like attribute can be disaggregated if the size of the group is available.
Then, rate-like attributes are first transformed into count-like measurements by scaling by the appropriate variable in the supergroup (usually, supergroup size), splitting that quantity proportionately, and finally re-scaling by the variable's value in the subgroup.
Quantities that fall into this category are:
- Per-capita hospitalization, infection, or case rates.
- Proportions of a population vaccinated.
- The proportion of a population successfully protected via immunization.
- $R$, as it is the number of secondary infections per primary infection. The same disclaimer as with wastewater concentrations applies.

Examples of things this package is unsuitable for disaggregating:
- Concentration parameters (e.g., for negative binomial models), standard deviations, and most other dispersion parameters. (Variances are additive, so variances of something summed over subgroups could be split if strong assumptions about covariances are made.)
- (Contact) networks, DAGS, or other graphs. These aren't things to which a notion of apportioning applies.

### Dis/aggregation versus enumeration

There are two related problems when handling subgroups and supergroups.
The first of these is _enumeration_.
Only after subgroups have been enumerated can supergroups be disaggregated, or aggregated.


To take age groups as an example, consider that we have measurements for supergroups "0-3 years", "4-11 years", and "12-17 years", and that we want to impute measurements on yearly age subgroups.
Enumeration is the process of specifying that the subgroup to supergroup map is:
```python
sub_to_super = {
    "0 years" : "0-3 years",
    "1 years" : "0-3 years",
    "2 years" : "0-3 years",
    "3 years" : "0-3 years",
    "4 years" : "4-11 years",
    "5 years" : "4-11 years",
    "6 years" : "4-11 years",
    "7 years" : "4-11 years",
    "8 years" : "4-11 years",
    "9 years" : "4-11 years",
    "10 years" : "4-11 years",
    "11 years" : "4-11 years",
    "12 years" : "12-17 years",
    "13 years" : "12-17 years",
    "14 years" : "12-17 years",
    "15 years" : "12-17 years",
    "16 years" : "12-17 years",
    "17 years" : "12-17 years",
}
```

The package offers built-in enumeration support to handle:
- Supergroups and subgroups defined by age (`AgeGroupEnumerator`)
- Arbitrary subgroups present in all supergroups (`CartesianEnumerator`)


## What is subgroup disaggregation anyways?

Let us consider a single variable $y$ which we have value for in supergroups $1, \dots I$ as $\hat{\mathbf{y}} = \hat{y}_1, \dots \hat{y}_I$.
That we want to impute values in subgroups of these supergroups implies the existence of at least one other variable which defines these subgroups in some way, $\mathbf{x}$.
From this (these) other variable(s), we can in some way obtain a set of proportions $\boldsymbol{\pi}$ which we use to apportion the values in supergroups to their constituent subgroups.

Supergroup $i$ has subgroups $j \in 1, \dots J_i$, and proportion vector $\boldsymbol{\pi}_i = \pi_{i1}, \dots \pi_{iJ_i}$, with $1 = \sum_j \pi_{ij}$.
We will impute
```math
\hat{y}_{ij} = \pi_{ij} \hat{y}_i
```
We convert densities to masses before disaggregation so that we can retain this mass-splitting paradigm for all subgroup disaggregation.

Subgroup disaggregation is thus the problem of defining and computing $\boldsymbol{\pi}(\mathbf{x})$.
This is a very broad class of problems, and we restrict ourselves to a relatively small set of them in what we provide with the package.
However, architecturally, this functionality can be readily extended by implementing a new object satisfying the `ProportionCalculator` [Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol).
Essentially, this is just a class that provides a function compute $\boldsymbol{\pi}_i(\mathbf{x})$.

At present, we focus exclusively on _uniform density disaggregation_ where it is assumed that all subgroups within a supergroup have the same density for the value in question.
We arrive at this shared approximation for two different kinds of grouping variables, categorical and one-dimensional continuous.

NOTE: The package allows for multivariate disaggregation of more than one measurement simultaneously.
In this case, it is assumed that the same weight model $\boldsymbol{\pi}(\mathbf{x})$ applies to all of the outcome measurements independently.

### Uniform density categorical disaggregation

This is perhaps the simplest of all disaggregation cases.
There is a single categorical subgrouping variable, and we have either proportion or count measurements for each subgroup.
Here we have
```math
\pi_{ij} = \frac{x_{ij}}{\sum_j x_{ij}} = \frac{x_{ij}}{x_{i}}
```

In some cases, it will be that $x_{ij} = x_{kj}$ for all $i,k$, and all subgroups have the same compositions.

When is this approach useful?
Consider that we have state-level vaccination data, but we want county-level data as model input.
If we assume that vaccination is homogenous within each state, that is, that vaccination rates are the same in all counties in a state, we can use this disaggregation approach.
Here the supergroups are states, and the subgroups counties within each state.

This assumption about proportionality among subgroups is baked into this disaggregation approach.
Supergroup $i$ has $\hat{y}_i$ vaccinated individuals, to each subgroup of which we distribute a fraction $\pi_{ij} = x_{ij} / x_i$.
It is instructive to look at the implied vaccination _rates_.
For supergroup $i$ that is $\hat{y}_i / x_i$, and for subgroup $ij$ it is
```math
\frac{\hat{y}_{ij}}{x_{ij}} = \frac{(x_{ij} / x_i) \hat{y}_i}{x_{ij}} = \frac{\hat{y}_{i}}{x_{i}}
```
Thus, we have assumed that the vaccination rate is uniform across the age supergroups.

What if our data weren't numbers of vaccinated individuals but vaccination rates?
As stated elsewhwere, `cfa-subgroup-imputer` transforms density measurements to mass measurements prior to disaggregation, and transforms them back afterwards.
So, first we use the supergroup sizes to transform the vaccination rates to counts.
Then we do the same distribution step as above.
Finally, we explicitly transform back to vaccination rates, which, per above, we have assumed uniform over the age supergroups.

As the implied incidence rates will always be uniform over subgroups, we call this a uniform density approach.

### Uniform density continuous disaggregation

Continuous disaggregation is the process of disaggregating where supergroups and subgroups are defined by (nonoverlapping, except between subgroups and their supergroups) ranges of a continuous variable.
We say that there is some underlying variable, $z$, that all attributes to be proportionally disaggregated are functions of it, and that both supergroups and subgroups are defined by breakpoints therein.
In the uniform density case, there is a single variable $x(z)$ which is used to define the weights.

The supergroups are defined by ranges of $z$ specified by breakpoints $z_0, \dots, z_I$, with supergroup $i$ spanning $z_{i - 1}$ to $z_i$.
Our model here is
```math
\hat{y}_i = \int_{z_{i - 1}}^{z_{i}} y(z) x(z) \mathrm{d}z
```

Each of these ranges is further subdivided by breakpoints $ z_{ij} \in z_{i0}, \dots, z_{iJ_i}$.
Analogously to above we have
```math
\hat{y}_{ij} = \int_{z_{(i)(j-1)}}^{z_{ij}} y(z) x(z) \mathrm{d}z
```

Making this equation useable in practice requires imposing more structure on the integral.
The package offers one option for this (though more may eventually be added), in which we assume that $y(z)$ is piecewise constant functions, uniform on the intervals $z_{i - 1}$ to $z_i$.
In this case,
```math
\hat{y}_i = \int_{z_{i - 1}}^{z_{i}} y_i x(z) \mathrm{d}z = y_i \int_{z_{i - 1}}^{z_{i}} x(z) \mathrm{d}z = y_i x_i
```
where assume that we know the integrated value $x_i$ measured for each group.
Thus
```math
y_i = \frac{\hat{y_i}}{x_i}
```

Applying the piecewise constant definition to the subgroup equation, we obtain
```math
\hat{y}_{ij} = \frac{\hat{y}_i}{x_i} \int_{z_{(i)(j-1)}}^{z_{ij}} x(z) \mathrm{d}z = \frac{\hat{y}_i}{x_i} x_{ij}
```
where again we assume we know the integrated value $x_{ij}$ for each subgroup.

Rearranging, we obtain
```math
\hat{y}_{ij} = \frac{x_{ij}}{x_i} \hat{y}_i
```
which fits into the stated weight-based framework with $\pi_{ij} = x_{ij} / x_i$.
It also fits into the categorical approach above if we define $w_ij$ to be 0 for all subgroups not contained within a supergroup, as $\sum_j x_{ij} = x_i$.
Thus, this framework is also a uniform density approach.

When is this approach useful?
While it ends up in the same place as the categorical case, it is useful for understanding how to tackle subgroups based on, for example, age.
In such a case, we have vaccination data on relatively large age subgroups, e.g. from [nis-py-api](https://github.com/CDCgov/nis-py-api).
But if our model needs values on smaller groups, such as yearly age groups, we need to disaggregate it.
The continuous variable component is probably mostly useful for hammering this problem into the correct shape for passing to a categorical-style imputer.
But if we ever want to try non-uniform interpolation, either because another continuous variable $x_2(z)$ is in play or because we don't want to assume $y(z)$ is piecewise constant, we will start from here.

#### A special case: when $x(z) = z$

There will be times we have the special case that $x(z) = z$.
For example, this can arise during the initial construction of age groups.
If our population data arises from a data source that does not have yearly age data, such as ACS data, before we use yearly age groups in downstream disaggregation, we must first disaggregate their sizes.
Thus we have $x(z) = z$ as age itself, while $y(z)$ remains the population density function.

The math works out identically as above.

🚧 But do we need a special flag in the code? How do we want to handle this?

## Disaggregating based on multiple variables

The package is focused on disaggregation based on a single variable.
By stringing together a series of such steps, disaggregation can be accomplished across multiple variables, for example, age group and age-associated risk factors.
While this approach appears marginal, which would suggest assumptions that the joint distribution factorizes, this need not be true as different supergroups can be disaggregated according to different proportions, thus achieving a more complex joint distribution from conditional distributions.
However, care must be taken that the desired joint distribution is sensible over the underlying variables, especially if they are continuous.
