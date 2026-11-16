"""Untracked ("ghost") ball carriers, as a module (notebook_ball §7).

    from analysis import ghost as gh
    GC = gh.ghost_carries(ctx, kin, EK)            # one row per ghost carry; `team` = decided team or ''

A ghost carry: the ball moves at dribbling speed (CARRY_MIN..CARRY_MAX), keeps its pace, and no
tracked player moves WITH it (within CARRY_RADIUS and CO_MOVE_MS of its velocity) -- the carrier is
not in the tracking, and "nearest player" can only point at whoever chases him.
Three clues name the carrier's team; a carry is decided when at least MIN_AGREE clues name the same
team and no other team is named as often:
  1. missing player count: the team further below 11 tracked players (if they differ by >= 0.5);
  2. missing formation role near the ball (analysis.identity roles): the team whose missing role is
     clearly closer to the ball (<= ROLE_RATIO x the other team's, and <= ROLE_MAX_M);
  3. who receives the ghost's pass: first reception by a tracked player after the ball is next
     played away fast (> PASS_MS) within NEXT_S.
The carry is back-dated to the last ball event up to BACKDATE_S before it (where it was won).
"""

import numpy as np
import pandas as pd

from analysis import features as fx
from analysis import identity as idn
from analysis.atd import HOME, AWAY
from analysis import passes as ps

FPS = fx.FPS
LENGTH, WIDTH = fx.LENGTH, fx.WIDTH
CARRY_MIN, CARRY_MAX = 1.0, 7.0
CARRY_RADIUS = 3.0
CO_MOVE_MS = 2.0
KEEP_PACE = 0.6
MIN_CARRY_S = 1.0
JOIN_S = 0.3
NEXT_S = 8.0
PASS_MS = 8.0
PASS_LOOK_S = 4.0
ROLE_RATIO = 0.6
ROLE_MAX_M = 30.0
MIN_AGREE = 2
BACKDATE_S = 1.0


def ghost_carries(ctx, kin, EK):
    ok, graded, N = ctx.ok, ctx.graded, ctx.N
    bxy, pxy, vel, speed = kin["bxy"], kin["pxy"], kin["vel"], kin["speed"]
    team, players = ctx.team, ctx.players
    counts = fx.team_counts(players, team)

    pvel = np.full_like(pxy, np.nan)
    pvel[1:-1] = (pxy[2:] - pxy[:-2]) * FPS / 2
    with np.errstate(invalid="ignore"):
        dist_all = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
        vdiff = np.linalg.norm(pvel - vel[:, None, :], axis=2)
        carrier_seen = np.any((dist_all <= CARRY_RADIUS) & (vdiff <= CO_MOVE_MS), axis=1)
        cand = ok & graded & ~ctx.dead_any & (speed >= CARRY_MIN) & (speed <= CARRY_MAX) & ~carrier_seen
    del dist_all, vdiff
    joined = cand.copy()
    for a, b in zip(*fx.runs(~cand)):
        if 0 < a and b < N and (b - a) <= JOIN_S * FPS and ok[a:b].all():
            joined[a:b] = True

    gk_slot = idn.goalkeeper_slots(players, team, graded)
    roles, templates, _ = idn.assign_roles(players, team, gk_slot)
    sides = fx.defending_side(players, team)
    half_idx = ctx.half

    def missing_role_distance(f):
        out = []
        h = half_idx[f]
        for side in (HOME, AWAY):
            tmpl = templates[(side, h)]
            idx = np.flatnonzero((team == side) & (roles[f] >= 0))
            missing = [r for r in range(len(tmpl)) if not (roles[f, idx] == r).any()]
            if not missing or not len(idx):
                out.append(np.inf)
                continue
            flip = sides[h][side] != "left"
            p = pxy[f, idx]
            pr = np.column_stack([LENGTH - p[:, 0], WIDTH - p[:, 1]]) if flip else p
            centre = (pr - tmpl[roles[f, idx]]).mean(0)
            spots = tmpl[missing] + centre
            if flip:
                spots = np.column_stack([LENGTH - spots[:, 0], WIDTH - spots[:, 1]])
            out.append(float(np.min(np.linalg.norm(spots - bxy[f], axis=1))))
        return out

    all_events = np.sort(EK.frame.values)
    receptions = EK[(EK.player_m <= ps.TOUCH_M) & (EK.kind == "reception (ball slows down)")].sort_values("frame")
    rows = []
    for a, b in zip(*fx.runs(joined)):
        if (b - a) < MIN_CARRY_S * FPS:
            continue
        third = max(1, (b - a) // 3)
        if np.nanmedian(speed[b - third:b]) < KEEP_PACE * np.nanmedian(speed[a:a + third]):
            continue
        deficit = 11 - counts[a:b].mean(0)
        clue_count = ("H" if deficit[0] > deficit[1] else "A") if abs(deficit[0] - deficit[1]) >= 0.5 else ""
        d_role = missing_role_distance((a + b) // 2)
        clue_role = ""
        for k, (mine, theirs) in enumerate(((d_role[0], d_role[1]), (d_role[1], d_role[0]))):
            if mine <= ROLE_MAX_M and mine <= ROLE_RATIO * theirs:
                clue_role = "HA"[k]
        fast = np.flatnonzero(speed[b - 2:b + int(NEXT_S * FPS)] > PASS_MS)
        clue_pass = ""
        if len(fast):
            k0 = b - 2 + fast[0]
            rec = receptions[(receptions.frame > k0) & (receptions.frame <= k0 + PASS_LOOK_S * FPS)]
            clue_pass = rec.team.iloc[0] if len(rec) else ""
        clues = [c for c in (clue_count, clue_role, clue_pass) if c]
        votes = pd.Series(clues, dtype=object).value_counts()
        top = votes.index[0] if len(votes) else ""
        decided = top if (len(votes) and votes.iloc[0] >= MIN_AGREE and (len(votes) == 1 or votes.iloc[0] > votes.iloc[1])) else ""
        prior = all_events[(all_events >= a - BACKDATE_S * FPS) & (all_events <= a)]
        rows.append({"start": int(prior[-1]) if len(prior) else int(a), "carry_start": int(a), "stop": int(b),
                     "dur_s": (b - a) / FPS, "clue_count": clue_count, "clue_role": clue_role,
                     "clue_pass": clue_pass, "n_clues": len(clues), "team": decided})
    return pd.DataFrame(rows, columns=["start", "carry_start", "stop", "dur_s", "clue_count", "clue_role",
                                       "clue_pass", "n_clues", "team"])


def ghost_addon(letters, GC):
    "notebook_ball's rule: every decided carry belongs to its team (never over a dead ball)"
    out = np.asarray(letters).copy()
    for r in GC[GC.team != ""].itertuples():
        seg = out[r.start:r.stop]
        out[r.start:r.stop] = np.where(seg == "D", "D", r.team)
    return out


def ghost_evidence(GC, n, w=1.0):
    """(x, mask): team evidence +-w per frame over every decided carry, and the frames it covers
    (where nearest-player evidence points at the chaser and should be dropped)."""
    x = np.zeros(n)
    mask = np.zeros(n, bool)
    for r in GC[GC.team != ""].itertuples():
        x[r.start:r.stop] += w if r.team == "H" else -w
        mask[r.start:r.stop] = True
    return x, mask
