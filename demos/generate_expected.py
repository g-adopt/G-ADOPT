#!/usr/bin/env python3
import argparse
from pathlib import Path

from test_all import cases, get_convergence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Expected demo results",
        description="Generates a pickle file with the expected demo results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("cases", nargs="*", help="Cases to process")
    args = parser.parse_args()

    requested_cases = {case.rstrip("/") for case in args.cases}

    for case in requested_cases:
        try:
            extra_checks = cases[case].get("extra_checks", [])
        except KeyError:
            print(f"Skipping unknown case: {case}")
            continue

        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[["u_rms"] + extra_checks]

        df.to_pickle(b / "expected.pkl")
