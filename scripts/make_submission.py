"""Build the current best model and write it as a validated submission.

    python scripts/make_submission.py                    # evaluation/results/final.csv (the submitted model)
    python scripts/make_submission.py --name my_team     # evaluation/results/my_team.csv
    python scripts/make_submission.py --model v8_whistle # the previous best, for comparison

The file lands in evaluation/results/ (gitignored) and is checked with exactly the official
validator's rules. The script prints only whether it is valid -- never a score.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm   # noqa: E402
from analysis import best                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="final", choices=["final", "v8_whistle"])
    ap.add_argument("--name", default=None, help="file name without .csv (default: the model's name)")
    args = ap.parse_args()
    args.name = args.name or args.model
    b = best.Bundle.load()
    labels = getattr(best, args.model)(b)
    path = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
    reason = pm.validate_submission(pm.read_label_csv(path), b.ctx.N)
    print(("INVALID: " + reason) if reason else "valid", "->", os.path.relpath(path, ROOT))
    sys.exit(1 if reason else 0)


if __name__ == "__main__":
    main()
