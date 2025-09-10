"""
A command line interface for using cfa-subgroup-imputer as a utility. For more, see the json submodule thereof.
"""

import argparse
import json

from cfa_subgroup_imputer.json import aggregate, disaggregate


def main():
    parser = argparse.ArgumentParser(
        description="A command line interface for supergroup disaggregation or subgroup aggregation."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    # TODO: do we just want to call this --action, let it be "[dis]aggregate", and call `impute` instead below?
    action.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate from subgroups to supergroups.",
    )
    action.add_argument(
        "--disaggregate",
        action="store_true",
        help="Disaggregate from supergroups to subgroups.",
    )

    parser.add_argument(
        "--supergroup-data",
        type=argparse.FileType("r"),
        required=True,
        help="Path to JSON file with supergroup definitions and data for disaggregation.",
        optional=False,
    )
    parser.add_argument(
        "--subgroup-defs",
        type=argparse.FileType("r"),
        required=True,
        help="Path to JSON file with subgroup definitions and data for aggregation.",
        optional=False,
    )
    parser.add_argument(
        "--subgroup-to-supergroup",
        type=argparse.FileType("r"),
        help="Path to JSON file with subgroup to supergroup mapping.",
    )
    parser.add_argument(
        "--supergroups-from",
        type=str,
        required=True,
        help="Name of key in supergroup_data defining supergroups.",
        optional=False,
    )
    parser.add_argument(
        "--subgroups-from",
        type=str,
        required=True,
        help="Name of key in subgroup_data defining subgroups.",
        optional=False,
    )
    parser.add_argument(
        "--group-type",
        type=str,
        choices=["categorical", "age"],
        help="What kind of groups are these, categorical or age?",
        optional=False,
    )
    parser.add_argument(
        "--loop-over",
        nargs="*",
        default=[],
        help="A collection of covariates to loop over. For example, if disaggregating age groups separately by location, list the name of the location variable.",
    )
    parser.add_argument(
        "--rate",
        nargs="*",
        default=[],
        help="Keys defining rate measurements in JSON file from which imputation is performed.",
    )
    parser.add_argument(
        "--count",
        nargs="*",
        default=[],
        help="Keys defining count measurements in JSON file from which imputation is performed.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Keys to exclude from imputation in JSON file from which imputation is performed.",
    )
    parser.add_argument(
        "--size-from",
        type=str,
        default="size",
        help="Key defining the size measurement (assumed the same in both supergroup and subgroup JSON files).",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("r"),
        help="Path to output JSON file.",
        optional=False,
    )

    args = parser.parse_args()

    with open(args.supergroup_data) as f:
        supergroup_data = json.load(f)
    with open(args.subgroup_data) as f:
        subgroup_data = json.load(f)

    subgroup_to_supergroup = None
    if args.subgroup_to_supergroup:
        with open(args.subgroup_to_supergroup) as f:
            subgroup_to_supergroup = json.load(f)

    kwargs = {
        "supergroup_data": supergroup_data,
        "subgroup_data": subgroup_data,
        "subgroup_to_supergroup": subgroup_to_supergroup,
        "supergroups_from": args.supergroups_from,
        "subgroups_from": args.subgroups_from,
        "group_type": args.group_type,
        "loop_over": args.loop_over,
        "rate": args.rate,
        "count": args.count,
        "exclude": args.exclude,
        "size_from": args.size_from,
    }

    if args.aggregate:
        result = aggregate(**kwargs)
    else:
        result = disaggregate(**kwargs)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
