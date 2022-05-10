import sys
from pathlib import Path

from test_all import cases, get_convergence

if __name__ == "__main__":
    if sys.argv[1:]:
        cases = set(cases).intersection(sys.argv[1:])

    for case in cases:
        b = Path(__file__).parent.resolve() / case
        df = get_convergence(b)[["u_rms", "nu_top"]]

        df.to_pickle(b / "expected.pkl")
