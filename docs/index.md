# cfa-subgroup-imputer

Imputing values in subgroups from supergroups, and vice-versa.
While a wide variety of values can be imputed, the focus of this package is on physical measurements of groups of people, such as the size of the group.

## Preliminary notes on terminology and assumptions

⚠️ This notation should not be considered finalized.
If you see something you think is wrong, bad, or unwise, open a PR.

### Groups

For our purposes, a group is an arbitrary subpopulation (possibly the entire population).
A group becomes a subgroup or supergroup only in relation to other groups.
That is, 10 year olds is both a subgroup of children under 18, and a supergroup of children aged 10 years and 0 months, 10 years and 1 month, ans so on.

We will assume that subgroups provided comprise the entire supergroup.
That is, that there won't be a supergroup of children under 18 with subgroups 1-3 year olds, 4-11 year olds, and 12-17 year olds, as this is missing infants less than one year old.

### Aggregating and disaggregating

The primary use of `cfa-subgroup-imputer` is _disaggregation_.
We have values for some supergroup, such as counts of vaccine doses in children under age 18, and we want to impute values for subgroups thereof.
Frequently we will want to disaggregate multiple supergroups simultaneously, such as moving from one set of groups to another, which is supported by the package.

The package will eventually also support _aggregation_.
In this case, we combine measurements for subgroups into a supergroup.

### Data, measurements, and properties, oh my

As stated, the focus of this package is on imputing values which reflect in one form or another actual counts in groups.
These are likely estimates, though they may come from a census, and we refer to them as _measurements_, making a distinction between two types.
We allow other variables of interest to be associated with groups, but not imputed, and refer to these as _properties_.

#### Measurements

Measurements are real-valued quantities, such as counts of hospitalizations or infection-hospitalization rates.
We distinguish between two broad classes of measurements:
- _Mass_(-like) measurements are things like counts of vaccinated individuals or group sizes. Mass-like measurements can readily be aggregated from subgroups to supergroups by summation, though disaggregation requires a model.
- _Density_(-like) measurements are things like rates of vaccination. Density-like measurements can readily be aggregated from subgroups to supergroups as long as the sizes of all subgroups are known. Disaggregation of densities also requires a model.

NOTE: Internally, when imputing, we move all densities to the mass scale (transforming rates into counts, possibly conditional counts in the case of IHRs), handle the imputation there, and then move back to the density scale.

#### Properties

Properties can be anything, relevant to the imputation process or not.
For example, we might have measurements for supergroups in a variety of locations.
In this case, we might want to track the location as a `property` which is shared between supergroups and subgroups but which does not further affect imputation, but which is useful to keep around to enable downstream work.

🚧 What is the right paradigm for things that _do_ matter?
Or can they matter?
Say we want to split out high and low risk groups, how does that work?

### Enumeration versus imputation

There are two related problems when handling subgroups and supergroups, which we term enumeration and imputation.
To take age groups as an example, consider that we have measurements for supergroups "0-3 years", "4-11 years", and "12-17 years", and that we want to impute measurements on yearly age subgroups.

_Enumeration_ is the process of specifying that the subgroup to supergroup map is:
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

_Imputation_ is the process which uses the enumerated subgroup to supergroup map and populates measurements from the supergroups into the subgroups.
(Or the other way around, if aggregating instead of disaggregating.)


## What is subgroup imputation anyways?

Let us consider a single variable $y$ which we have measured for supergroups $1, \dots I$ as $\hat{\mathbf{y}} = \hat{y}_1, \dots \hat{y}_I$.
That we want to impute values in subgroups of these supergroups implies the existence of at least one other variable which defines these subgroups in some way, call this $\mathbf{X}$.
From this (these) other variable(s), we can in some way obtain a set of normalized _weights_ $\mathbf{W}$ which we use to apportion the values in supergroups to their constituent subgroups.

Supergroup $i$ has subgroups $j \in 1, \dots J_i$, and weight vector $\mathbf{w}_i = w_{i1}, \dots w_{iJ_i}$, with $1 = \sum_j w_{ij}$.
We will impute
```math
\hat{y}_{ij} = w_{ij} \hat{y}_i
```
We convert densities to masses before imputation so that we can retain this mass-splitting paradigm for all subgroup imputation.

Subgroup imputation is thus the problem of defining and computing $\mathbf{w}(\mathbf{X})$.
This is a very broad class of problems, and we restrict ourselves to a relatively small set of them.

At present, we focus exclusively on _uniform density imputation_ where it is assumed that all subgroups within a supergroup have the same density for the imputed measurements.
We arrive at this shared approximation for two different kinds of grouping variables, categorical and one-dimensional continuous.

NOTE: The package allows for multivariate imputation of more than one measurement simultaneously.
In this case, it is assumed that the same weight model $\mathbf{w}(\mathbf{X})$ applies to all of the outcome measurements independently.

### Uniform density categorical imputation

This is perhaps the simplest of all imputation cases.
There is a single categorical subgrouping variable, and we have either proportion or count measurements for each subgroup.
Here we have
```math
w_{ij} = \frac{x_{ij}}{\sum_j x_{ij}} = \frac{x_{ij}}{x_{i}}
```

In some cases, it will be that $x_{ij} = x_{kj}$ for all $i,k$, and all subgroups have the same compositions.

When is this approach useful?
Consider that we have state-level vaccination data, but we want county-level data as model input.
If we assume that vaccination is homogenous across each state, that is, that vaccination rates are the same in all counties in a state, we can use this imputation approach.
Here the supergroups are states, and the subgroups counties within each state.

This assumption about proportionality among subgroups is baked into this imputation approach.
Supergroup $i$ has $\hat{y}_i$ vaccinated individuals, to each subgroup of which we distribute a fraction $w_{ij} = x_{ij} / x_i$.
It is instructive to look at the implied vaccination _rates_.
For supergroup $i$ that is $\hat{y}_i / x_i$, and for subgroup $ij$ it is
```math
\frac{\hat{y}_{ij}}{x_{ij}} = \frac{(x_{ij} / x_i) \hat{y}_i}{x_{ij}} = \frac{\hat{y}_{i}}{x_{i}}
```
Thus, we have assumed that the vaccination rate is uniform across the age supergroups.

What if our data weren't numbers of vaccinated individuals but vaccination rates?
As stated elsewhwere, `cfa-subgroup-imputer` transforms density measurements to mass measurements prior to imputation, and transforms them back afterwards.
So, first we use the supergroup sizes to transform the vaccination rates to counts.
Then we do the same distribution step as above.
Finally, we explicitly transform back to vaccination rates, which, per above, we have assumed uniform over the age supergroups.

As the implied incidence rates will always be uniform over subgroups, we call this the uniform density interpolation approach.

### Uniform density continuous imputation

When performing continuous interpolation, we assume that there is some underlying variable, $z$, that all measurements are functions of it, and that both supergroups and subgroups are defined by breakpoints therein.
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
which fits into the stated weight-based framework with $w_{ij} = x_{ij} / x_i$.
It also fits into the categorical approach above if we define $w_ij$ to be 0 for all subgroups not contained within a supergroup, as $\sum_j x_{ij} = x_i$.
Thus, this framework can be also be considered uniform density interpolation.

When is this approach useful?
While it ends up in the same place as the categorical case, it is useful for understanding how to tackle subgroups based on, for example, age.
In such a case, we have vaccination data on relatively large age subgroups, e.g. from [nis-py-api](https://github.com/CDCgov/nis-py-api).
But if our model needs values on smaller groups, such as yearly age groups, we need to impute it.
The continuous variable component is probably mostly useful for hammering this problem into the correct shape for passing to a categorical-style imputer.
But if we ever want to try non-uniform interpolation, either because another continuous variable $x_2(z)$ is in play or because we don't want to assume $y(z)$ is piecewise constant, we will start from here.

#### A special case: when $x(z) = z$

There will be times we have the special case that $x(z) = z$.
For example, this can arise during the initial construction of age groups.
If our population data arises from a data source that does not have yearly age data, such as ACS data, before we use yearly age groups in downstream imputation, we must first impute their sizes.
Thus we have $x(z) = z$ as age itself, while $y(z)$ remains the population density function.

The math works out identically as above.

🚧 But do we need a special flag in the code? How do we want to handle this?


## Spec/UI thinking

It would be nice if we could provide file to file interface, since users might like to use this in a workflow in other languages, e.g. python, or as a step in a pipeline.
A command line interface therefore seems called for.
This means passing in or constructing the `sub : super` group map.
- The easy way for developing would seem to be to require a dataframe passed in that does this.
- The easy way for users to interact would seem to be to allow either
  - Passing in a tabular data file, to capture irregular sub/supergroup cases like states and counties OR
  - Construct the map on the fly based on
    - One file of supergroups AND
    - One file of subgroups AND
    - An argument specifying the type of variable used for group definition, then
        - If `group_vartype` is "continuous" we use continuous variable logic and assume some subgroups go in some supergroups, but not all
        - If `group_vartype` is "categorical" assume that each supergroup contains one of each subgroup, and construct a `f"{super}_{sub}" : super` map.
