"""Team shape: how each team moves with and without the ball, and a ball-free possession model.

    from analysis import shape as sh
    F = sh.team_features(ctx)                  # DataFrame, one row per frame, per-team shape columns
    y, w = sh.pseudo_labels(ctx, S)            # labels from strong ball evidence only (dribble / clear touch)
    m = sh.fit(F, y, mask)                     # logistic regression (numpy), standardised features
    p = sh.predict(m, F)                       # P(home has the ball) per frame -- never looks at the ball

Every team is described in its OWN attacking direction (x' = metres from its own goal line towards the
opponent's, y' across), outfield players only (the keeper's dedicated slot is dropped), with robust
statistics so a missing or duplicated player moves them little:
  cx, cy        centroid
  length, width p90 - p10 of x', y'
  spread        mean distance to the centroid
  front, back   p90 / p10 of x' (attacking line, defensive line)
  vx            mean velocity along the attacking direction (m/s): + = moving towards the opponent goal
  speed         mean speed
  n_opp_half    players in the opponent half
  n             tracked outfield players
Features are smoothed over SMOOTH_S (centred), and the model sees home, away and home-minus-away.
"""

import numpy as np
import pandas as pd

from analysis import features as fx
from analysis import identity as idn
from analysis.atd import HOME, AWAY

FPS = fx.FPS
LENGTH, WIDTH = fx.LENGTH, fx.WIDTH
SMOOTH_S = 1.0
BASE = ["cx", "cy", "length", "width", "spread", "front", "back", "vx", "speed", "n_opp_half", "n"]


def team_features(ctx, smooth_s=SMOOTH_S):
    players, team, half = ctx.players, ctx.team, ctx.half
    pxy = fx.to_metres(players[:, :, :2])
    pvel = np.full_like(pxy, np.nan)
    pvel[1:-1] = (pxy[2:] - pxy[:-2]) * FPS / 2
    gk = idn.goalkeeper_slots(players, team, ctx.graded)
    sides = fx.defending_side(players, team)
    cols = {}
    for side, tag in ((HOME, "h"), (AWAY, "a")):
        idx = np.flatnonzero(team == side)
        idx = idx[idx != gk[side]]
        x, y = pxy[:, idx, 0].copy(), pxy[:, idx, 1].copy()
        vx = pvel[:, idx, 0].copy()
        for h in (1, 2):
            if sides[h][side] != "left":                       # attacks towards x = 0: mirror
                m = half == h
                x[m], y[m], vx[m] = LENGTH - x[m], WIDTH - y[m], -vx[m]
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
            q = lambda a, p: np.nanpercentile(a, p, axis=1)
            cols[f"{tag}_cx"], cols[f"{tag}_cy"] = cx, cy
            cols[f"{tag}_length"] = q(x, 90) - q(x, 10)
            cols[f"{tag}_width"] = q(y, 90) - q(y, 10)
            cols[f"{tag}_spread"] = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)
            cols[f"{tag}_front"], cols[f"{tag}_back"] = q(x, 90), q(x, 10)
            cols[f"{tag}_vx"] = np.nanmean(vx, 1)
            cols[f"{tag}_speed"] = np.nanmean(np.hypot(pvel[:, idx, 0], pvel[:, idx, 1]), 1)
            cols[f"{tag}_n_opp_half"] = (x > LENGTH / 2).sum(1).astype(float)
            cols[f"{tag}_n"] = np.isfinite(x).sum(1).astype(float)
    F = pd.DataFrame(cols)
    k = max(1, int(smooth_s * FPS))
    F = F.rolling(k, center=True, min_periods=1).mean()
    for b in BASE:
        F[f"d_{b}"] = F[f"h_{b}"] - F[f"a_{b}"]
    return F


def pseudo_labels(ctx, S, levels=(0, 1)):
    """y = 1 home / 0 away / -1 unknown, from the cascade sources (analysis.cascade.sources) at the given
    levels only (default: dribble, clear touch), on live graded frames."""
    lab = np.full(ctx.N, -1, np.int8)
    for k in levels[::-1]:                     # higher-priority level written last
        has = S[:, k] >= 0
        lab[has] = np.where(S[has, k] == 0, 1, 0)
    lab[~ctx.graded | ctx.dead_any] = -1
    return lab


def auc(score, y):
    "area under the ROC curve (rank formulation), NaN-safe"
    m = np.isfinite(score) & (y >= 0)
    s, t = score[m], y[m]
    if t.min() == t.max():
        return np.nan
    r = pd.Series(s).rank().values
    n1 = t.sum()
    return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(t) - n1))


def fit(F, y, mask, l2=1.0, iters=25, every=5):
    """L2 logistic regression by Newton's method on standardised features; trains on mask & (y >= 0),
    one frame in `every` (neighbouring frames are near-duplicates)."""
    cols = list(F.columns)
    X = F.values
    sel = np.flatnonzero(mask & (y >= 0) & np.isfinite(X).all(1))[::every]
    mu, sd = X[sel].mean(0), X[sel].std(0) + 1e-9
    Z = np.c_[np.ones(len(sel)), (X[sel] - mu) / sd]
    t = y[sel].astype(float)
    w = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - t) + R @ w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"cols": cols, "mu": mu, "sd": sd, "w": w, "n_train": len(sel)}


def predict(model, F):
    X = F[model["cols"]].values
    Z = (X - model["mu"]) / model["sd"]
    Z = np.where(np.isfinite(Z), Z, 0.0)                     # missing feature = average
    return 1 / (1 + np.exp(-(model["w"][0] + Z @ model["w"][1:])))


def blackout_addon(letters, p_home, ok, runs, min_gap_s=2.0, thr=0.8, min_s=3.0, smooth_s=3.0):
    """Inside every ball-missing gap of at least min_gap_s, if the shape model says the OTHER team has
    the ball (smoothed P >= thr) for at least min_s in a row, switch to that team from where it starts
    until the gap ends. Dead-ball frames are never touched; X is read as the previous label."""
    out = np.asarray(letters).copy()
    k = max(1, int(smooth_s * FPS))
    ps_ = pd.Series(p_home).rolling(k, center=True, min_periods=1).mean().values
    need = int(min_s * FPS)
    for a, b in runs:
        gaps = zip(*fx.runs(~ok[a:b]))
        for g0, g1 in gaps:
            g0, g1 = a + g0, a + g1
            if g1 - g0 < min_gap_s * FPS:
                continue
            seg = out[g0:g1].copy()
            # current team inside the gap = last H/A label at or before each frame
            cur, team_now = None, np.empty(len(seg), dtype="<U1")
            prev = out[g0 - 1] if g0 > a else "X"
            cur = prev if prev in ("H", "A") else None
            for i, c in enumerate(seg):
                if c in ("H", "A"):
                    cur = c
                team_now[i] = cur or ""
            other_strong = np.where(team_now == "H", ps_[g0:g1] <= 1 - thr,
                                    np.where(team_now == "A", ps_[g0:g1] >= thr, False)) & (seg != "D")
            for r0, r1 in zip(*fx.runs(other_strong)):
                if r1 - r0 >= need:
                    new = "A" if team_now[r0] == "H" else "H"
                    stop = r0
                    while stop < len(seg) and seg[stop] != "D":
                        stop += 1
                    seg[r0:stop] = new
                    break                                     # one switch per gap
            out[g0:g1] = seg
    return out
