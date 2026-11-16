"""A reliability cascade for the team in possession.

Instead of a majority vote, every frame asks its evidence sources in order of reliability and takes
the first one that has an opinion:

| level | source        | says the team when ...                                                    |
|-------|---------------|---------------------------------------------------------------------------|
| 1     | dribble       | a tracked player moves with the ball (analysis.contest)                   |
| 2     | clear touch   | nearest player within 1 m, the other team >= 1.5 m further, not contested |
| 3     | flight        | the ball is in flight: the kicker's team keeps it until someone receives  |
| 4     | restart       | the 2 s after a decided restart (analysis.restarts)                       |
| 5     | touch         | nearest player within 2 m, not contested, not in flight                   |
| 6     | tag phase     | a Smart Tagging phase with a team (attacking codes: that team;            |
|       |               | defending codes: the other one)                                           |
| -     | carry         | nobody has an opinion: keep the last team                                 |

Three ways to turn that into labels:
- `first_available`  : the top source wins every frame, carry where there is none.
- `confirmed`        : a strong source (levels 1-4) switches the team at once; a weak one (5-6) only
                       switches it when the challenger is confirmed -- CONFIRM_N of its frames, at
                       least one from a different source, within CONFIRM_S -- otherwise keep the team.
- `to_viterbi`       : each frame's top source becomes Viterbi evidence weighted by its reliability,
                       w = SCALE * log(r / (1 - r)); levels with no opinion give 0.
The reliabilities r are set by hand (not fitted); the notebook prints how often each level agrees with
the public truth so they can be revisited.
"""

import numpy as np

from games import GAME, half_video_blocks
from analysis import features as fx
from analysis import passes as ps
from analysis import viterbi as vt

FPS = fx.FPS
LEVELS = ["dribble", "clear touch", "flight", "restart", "touch", "tag phase"]
RELIABILITY = {"dribble": 0.95, "clear touch": 0.92, "flight": 0.85, "restart": 0.92, "touch": 0.75, "tag phase": 0.65}
SCALE = 0.5
STRONG = 4                  # levels 1..STRONG switch at once in `confirmed`
CONFIRM_S, CONFIRM_N = 0.6, 4
CLEAR_M, CLEAR_MARGIN_M = 1.0, 1.5
RESTART_S = 2.0

ATTACKING = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "CREATING CHANCES", "BALL IN FINAL THIRD",
             "BALL IN THE BOX", "LONG BALL"}
DEFENDING = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
             "DEFENSIVE TRANSITION"}


def tag_team(ctx):
    "(n,) 0 home / 1 away / -1 from the Smart Tagging phases (shortest phase wins where they overlap)"
    out = np.full(ctx.N, -1, np.int8)
    span = np.full(ctx.N, np.inf)
    for e in ctx.tags:
        code = e["code"].upper()
        t = (e["tags"].get("Team") or "").lower()
        side = 0 if t.startswith("barcelona") else (1 if t.startswith("at") else -1)
        if side < 0 or code not in ATTACKING | DEFENDING:
            continue
        if code in DEFENDING:
            side = 1 - side
        a, b = max(0, e["start_frame"]), min(ctx.N, e["end_frame"])
        L = b - a
        m = span[a:b] > L
        out[a:b][m] = side
        span[a:b][m] = L
    return out


def sources(ctx, kin, C, R):
    """(n, len(LEVELS)) team per source: 0 home, 1 away, -1 no opinion."""
    n = ctx.N
    ok, dn, tn, near = kin["ok"], kin["d_near"], kin["team_near"], kin["near"]
    S = np.full((n, len(LEVELS)), -1, np.int8)
    S[:, 0] = np.where(C["dribble"], C["dribbler"], -1)
    with np.errstate(invalid="ignore"):
        margin = np.abs(near[:, 0] - near[:, 1])
        margin[np.isnan(margin)] = np.inf
        free = ok & ~C["contested"] & ~ctx.in_flight
        S[:, 1] = np.where(free & (dn <= CLEAR_M) & (margin >= CLEAR_MARGIN_M), tn, -1)
        S[:, 4] = np.where(free & (dn <= ps.TOUCH_M), tn, -1)
    flight = np.full(n, -1, np.int8)
    kick = dict(zip(ctx.flights.start, ctx.flights.kicker))
    for a, b in zip(*fx.runs(ctx.in_flight)):
        k = kick.get(a, "")
        if k:
            flight[a:b] = 0 if k == "H" else 1
    S[:, 2] = flight
    for r in R[(R.team != "") & (R.restart >= 0)].itertuples():
        S[r.restart:r.restart + int(RESTART_S * FPS), 3] = 0 if r.team == "H" else 1
    S[:, 5] = tag_team(ctx)
    S[~ctx.graded] = -1
    return S


def top(S):
    "(team, level) of the first source with an opinion; level -1 where none"
    has = S >= 0
    lvl = np.where(has.any(1), has.argmax(1), -1)
    team = np.where(lvl >= 0, S[np.arange(len(S)), np.maximum(lvl, 0)], -1)
    return team.astype(np.int8), lvl


def _carry(team, ctx):
    out = np.full(ctx.N, "X", dtype="<U1")
    for _, a, z in half_video_blocks(GAME):
        seg = team[a:z + 1].astype(float)
        seg[seg < 0] = np.nan
        filled = np.nan_to_num(_ffill_bfill(seg), nan=0).astype(int)
        out[a:z + 1] = np.where(filled == 0, "H", "A")
    return out


def _ffill_bfill(x):
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    if np.isnan(y).any():
        first = np.flatnonzero(np.isfinite(y))
        if len(first):
            y[:first[0]] = y[first[0]]
    return y


def first_available(ctx, S):
    team, _ = top(S)
    return _carry(team, ctx)


def confirmed(ctx, S):
    team, lvl = top(S)
    out = np.full(ctx.N, "X", dtype="<U1")
    window = int(CONFIRM_S * FPS)
    for _, a, z in half_video_blocks(GAME):
        state = -1
        seg = np.empty(z + 1 - a, np.int8)
        for i, t in enumerate(range(a, z + 1)):
            c = team[t]
            if c >= 0 and c != state:
                if state < 0 or lvl[t] < STRONG:
                    state = c
                else:
                    w = slice(t, min(z + 1, t + window))
                    mine = team[w] == c
                    if mine.sum() >= CONFIRM_N and len(set(lvl[w][mine].tolist())) >= 2:
                        state = c
            seg[i] = state
        seg[seg < 0] = seg[seg >= 0][0] if (seg >= 0).any() else 0
        out[a:z + 1] = np.where(seg == 0, "H", "A")
    return out


def with_dead(letters, dead):
    out = np.asarray(letters).copy()
    out[dead & (out != "X")] = "D"
    return out


def to_viterbi(ctx, S, EK, p, extra_x=None, extra_y=None):
    """Viterbi with the cascade's top source as team evidence (reliability-weighted)."""
    team, lvl = top(S)
    w = np.array([SCALE * np.log(RELIABILITY[k] / (1 - RELIABILITY[k])) for k in LEVELS])
    x = np.where(lvl >= 0, np.where(team == 0, 1.0, -1.0) * w[np.maximum(lvl, 0)], 0.0)
    _, y = vt.evidence(ctx, EK, p)
    ev = EK[(EK.player_m <= ps.TOUCH_M) & EK.kind.isin(["kick (ball speeds up)", "reception (ball slows down)"])]
    np.add.at(x, ev.frame.values, np.where(ev.team.values == "H", p.w_event, -p.w_event))
    if extra_x is not None:
        x = x + extra_x
    if extra_y is not None:
        y = y + extra_y
    cost = vt.costs(x, y)
    trans = vt.transitions(p)
    out = np.full(ctx.N, "X", dtype="<U1")
    for _, a, z in half_video_blocks(GAME):
        out[a:z + 1] = vt.LETTERS[vt.decode(cost[a:z + 1], trans, p.late)]
    return out
