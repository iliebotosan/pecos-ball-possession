"""One-call accuracy checker for every idea we try.

    from analysis.check import check, history
    check(labels, "idea_name")               # score + log it
    check(labels, "idea_name", save=True)    # also write evaluation/results/idea_name.csv and validate it
    history()                                # every check so far, newest last

`labels` is anything 199319 long: letters 'H'/'A'/'D'/'X' or the scorer's codes 0..3.

What it reports
---------------
On the public 5 minutes (the only truth we have):
  spellF1 (+ 5-95% bootstrap range over the 17 possessions), macroF1, possessions found vs 17,
  dead-ball precision/recall, team accuracy where both agree the ball is alive.
On the full match (no truth needed -- a sanity check):
  number of possessions, per minute, median duration, share of dead ball.

The bootstrap range is wide (~0.3) because 17 possessions is a tiny sample: two ideas whose ranges
overlap are not distinguishable on the public window. Watch the full-match columns too -- an idea
that looks good on 5 minutes but invents 8 possessions a minute over the match is not good.

The log lives in evaluation/results/check_log.csv, which is gitignored like every results CSV.
"""

import os
import time

import numpy as np
import pandas as pd

from games import GAME, half_video_blocks, source_frames
from evaluation import possession_metrics as pm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")
LOG = os.path.join(RESULTS, "check_log.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25
N_FRAMES = source_frames("tactical")
BOOTSTRAP_N = 1000

_gt = None


def truth():
    """Public ground truth as codes (cached)."""
    global _gt
    if _gt is None:
        _gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return _gt


def to_codes(labels):
    """Letters or codes -> int8 codes 0..3 (H, A, D, X)."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels.astype(np.int8)
    return pm.label_codes(labels)


def full_match_stats(codes):
    """Possession counts over both halves -- no truth involved."""
    halves = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    spells = pm.possession_spells(codes, halves)
    graded = np.concatenate([codes[a:b] for a, b in halves])
    minutes = len(graded) / FPS / 60.0
    lengths = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    return {
        "full_poss": len(spells),
        "full_poss_per_min": round(len(spells) / minutes, 2),
        "full_median_dur_s": round(float(np.median(lengths)), 1) if len(lengths) else 0.0,
        "full_share_D": round(float(np.mean(graded == pm.D)), 3),
    }


def score(codes):
    """The checker's numbers for one prediction, as a flat dict."""
    r = pm.evaluate(truth(), codes, bootstrap_n=BOOTSTRAP_N)
    d = r["per_class"]["D"]
    return {
        "spellF1": round(r["spell_f1_mean"], 4),
        "spell_lo": round(r["spell_bootstrap_low"], 3),
        "spell_hi": round(r["spell_bootstrap_high"], 3),
        "macroF1": round(r["macro_f1"], 4),
        "poss": r["n_pred_spells"],
        "poss_truth": r["n_gt_spells"],
        "mean_dur_s": round(r["mean_spell_duration_s"], 1),
        "D_precision": round(d["precision"], 3),
        "D_recall": round(d["recall"], 3),
        "team_acc": round(r["team_acc_both_alive"], 3),
        **full_match_stats(codes),
    }


def check(labels, name, save=False, note="", log=True):
    """Score a prediction, append it to the log, return a one-row DataFrame."""
    codes = to_codes(labels)
    if len(codes) != N_FRAMES:
        raise ValueError("{} labels, expected {}".format(len(codes), N_FRAMES))

    # every row has the same columns, so the log CSV stays aligned
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "name": name, **score(codes), "note": note,
           "saved": ""}

    if save:
        path = os.path.join(RESULTS, "{}.csv".format(name))
        table = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": np.array(pm.LABELS)[codes]})
        table.to_csv(path, index=False)
        # the same check the official validator and scorer run
        reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
        row["saved"] = "INVALID: " + reason if reason else os.path.relpath(path, ROOT)

    if log:
        os.makedirs(RESULTS, exist_ok=True)
        pd.DataFrame([row]).to_csv(LOG, mode="a", index=False, header=not os.path.isfile(LOG))
    return pd.DataFrame([row]).set_index("name")


def history(last=None):
    """Every logged check, newest last."""
    if not os.path.isfile(LOG):
        return pd.DataFrame()
    table = pd.read_csv(LOG)
    return table.tail(last) if last else table


def check_file(path, name=None):
    """Score an existing submission CSV the same way."""
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, N_FRAMES)
    if reason:
        raise ValueError("{} is not a valid submission: {}".format(path, reason))
    return check(pm.label_codes(table), name or os.path.splitext(os.path.basename(path))[0])


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200)
    for p in sys.argv[1:]:
        print(check_file(p).T)
