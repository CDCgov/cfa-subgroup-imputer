"""
A command line interface for using cfa-subgroup-imputer as a utility, providing the ability
to say, more or less, "take this file, with measurements on these supergroups, and write
me a new one, with measurements on those subgroups."

Possible syntax:
python -m cfa_subgroup_imputer
    --[dis]aggregate
    --supergroup_df [path to dataframe with supergroups]
    --subgroup_df [path to dataframe with subgroups]
    --split_on [the variable that defines sub/supergroups]
    [--group_map_df [path to dataframe with arbitrary subgroups]]
    [--loop_over []]

We will eventually have to figure out how to handle all the columns and how they map to `Group.attributes`.
Default assumptions might be, other than columns called out for splitting on and looping over:
- Any column found in both sub and supergroup data is `ignore`d
- Any non-numeric column found only in the supergroup data should be copied or imputed to subgroups, where
    - Non-numeric columns are copied
    - Numeric columns are imputed
"""

if __name__ == "__main__":
    raise NotImplementedError("Command-line usage is not currently supported.")
