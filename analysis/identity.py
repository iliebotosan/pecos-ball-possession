"""Who is who: track slots, tracklets, duplicates, stitching and formation roles.

The ATD has 46 track slots and no player identities. A slot is reused for many different people
over the match, except the goalkeepers, who each get a dedicated slot. This module turns the raw
slots into things we can reason about:

    tracklets(...)        one row per continuous run of a slot (start, stop, where it began/ended)
    goalkeeper_slots(...) the dedicated keeper slot of each team, found from the data
    teleports(...)        a slot jumping faster than a human can run = it switched person
    duplicate_episodes()  two same-team slots on top of each other for a while
    stitch(...)           link a tracklet that ends to one that starts nearby soon after
    assign_roles(...)     map every outfield player, every frame, to a formation role
                          (e.g. "DEF-L"), so a missing player becomes "the left back is missing"

Import from the project root, like the rest of analysis/.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analysis.atd import HOME, AWAY
from analysis import features as fx

FPS = fx.FPS
TEAM_NAME = {HOME: "H", AWAY: "A"}


# -- slots and tracklets -----------------------------------------------------------------------

def tracklets(players, team, mask=None):
    """One row per continuous run of a slot: slot (1-based, as in the ATD), team, start, stop
    (exclusive), duration, and the position in metres where it started and ended.

    mask (bool, n) restricts to e.g. graded frames; a run is cut where the mask is False.
    """
    present = np.isfinite(players[:, :, 0])
    if mask is not None:
        present = present & mask[:, None]
    xy = fx.to_metres(players[:, :, :2])
    rows = []
    for j in range(players.shape[1]):
        starts, stops = fx.runs(present[:, j])
        for a, b in zip(starts, stops):
            rows.append((j + 1, TEAM_NAME[team[j]], a, b, (b - a) / FPS,
                         xy[a, j, 0], xy[a, j, 1], xy[b - 1, j, 0], xy[b - 1, j, 1]))
    return pd.DataFrame(rows, columns=["slot", "team", "start", "stop", "dur_s",
                                       "start_x", "start_y", "end_x", "end_y"])


def slot_table(players, team, mask):
    """Per-slot summary: how much it is used, in which half, how fragmented, how much it moves."""
    halves = fx.half_index(len(players))
    present = np.isfinite(players[:, :, 0])
    xy = fx.to_metres(players[:, :, :2])
    tl = tracklets(players, team, mask)
    rows = []
    for j in range(players.shape[1]):
        t = tl[tl.slot == j + 1]
        rows.append({
            "slot": j + 1, "team": TEAM_NAME[team[j]],
            "active_graded": present[mask, j].mean(),
            "active_h1": present[halves == 1, j].mean(), "active_h2": present[halves == 2, j].mean(),
            "tracklets": len(t),
            "median_tracklet_s": t.dur_s.median() if len(t) else 0.0,
            "longest_tracklet_s": t.dur_s.max() if len(t) else 0.0,
            "x_std_m": np.nanstd(xy[mask, j, 0]) if present[mask, j].any() else np.nan,
            "y_std_m": np.nanstd(xy[mask, j, 1]) if present[mask, j].any() else np.nan,
        })
    return pd.DataFrame(rows)


def goalkeeper_slots(players, team, mask):
    """{HOME: slot_index, AWAY: slot_index} (0-based) of each team's dedicated keeper slot.

    A keeper stays near the middle of his goal: among the slots active most of the match, his has
    by far the smallest spread across the pitch width. Found from the data, not hard-coded.
    """
    table = slot_table(players, team, mask)
    out = {}
    for side in (HOME, AWAY):
        busy = table[(table.team == TEAM_NAME[side]) & (table.active_graded > 0.5)]
        out[side] = int(busy.sort_values("y_std_m").iloc[0].slot) - 1
    return out


# -- strange patterns --------------------------------------------------------------------------

def step_speed(players):
    """(n, slots) speed in m/s between frame t-1 and t for the same slot; NaN if either is missing."""
    xy = fx.to_metres(players[:, :, :2])
    out = np.full(players.shape[:2], np.nan)
    out[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * FPS
    return out


def teleports(players, max_speed=12.0):
    """(n, slots) True where a slot moved faster than max_speed m/s in one frame.

    The fastest sprinters top out around 10-11 m/s, so a faster step means the slot jumped to a
    different person (an identity switch) or a tracking glitch.
    """
    with np.errstate(invalid="ignore"):
        return step_speed(players) > max_speed


def duplicate_episodes(players, team, radius_m=1.0, min_s=0.2):
    """Runs where two slots of the same team are within radius_m of each other.

    Per episode: the two slots, start, stop, duration, mean distance, how differently they move
    (mean |velocity difference|, m/s -- near 0 means they move as one, i.e. one person tracked
    twice), and whether one of the two slots appears or disappears during the episode (a new
    slot spawning on top of an existing player is the classic duplicate).
    """
    xy = fx.to_metres(players[:, :, :2])
    vel = np.full_like(xy, np.nan)
    vel[1:-1] = (xy[2:] - xy[:-2]) * FPS / 2.0
    present = np.isfinite(players[:, :, 0])
    rows = []
    for side in (HOME, AWAY):
        idx = np.flatnonzero(team == side)
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                i, j = idx[a_i], idx[b_i]
                both = present[:, i] & present[:, j]
                if not both.any():
                    continue
                d = np.full(len(xy), np.inf)
                d[both] = np.linalg.norm(xy[both, i] - xy[both, j], axis=-1)
                starts, stops = fx.runs(d < radius_m)
                for s, e in zip(starts, stops):
                    if (e - s) < min_s * FPS:
                        continue
                    dv = np.linalg.norm(vel[s:e, i] - vel[s:e, j], axis=-1)
                    edge = lambda k: (s > 0 and not present[s - 1, k]) or (e < len(xy) and not present[e, k])
                    rows.append({"team": TEAM_NAME[side], "slot_a": i + 1, "slot_b": j + 1,
                                 "start": s, "stop": e, "dur_s": (e - s) / FPS,
                                 "mean_dist_m": float(d[s:e].mean()),
                                 "vel_diff_ms": float(np.nanmean(dv)) if np.isfinite(dv).any() else np.nan,
                                 "slot_appears_or_vanishes": bool(edge(i) or edge(j))})
    return pd.DataFrame(rows)


# -- stitching ---------------------------------------------------------------------------------

def stitch(tl, max_gap_s=3.0, run_speed=8.0, slack_m=2.0):
    """Greedy re-identification: link each tracklet end to a same-team tracklet start.

    A start is a candidate for an end if it begins within max_gap_s after the end and no further
    away than a player could run in that time (run_speed * gap + slack_m). Each end and each start
    is used once; ends are processed in time order and take the closest candidate. Keepers'
    dedicated slots can be included or filtered out beforehand.

    Returns tl with a `chain` column: tracklets sharing a chain id are believed to be one person.
    """
    tl = tl.sort_values("start").reset_index(drop=True)
    chain = np.arange(len(tl))
    next_of = {}
    taken = set()
    by_team = {k: g for k, g in tl.groupby("team")}
    for i in tl.sort_values("stop").index:
        r = tl.loc[i]
        g = by_team[r.team]
        cand = g[(g.start >= r.stop) & (g.start <= r.stop + max_gap_s * FPS) & (g.index != i)]
        if not len(cand):
            continue
        gap = (cand.start - r.stop) / FPS
        dist = np.hypot(cand.start_x - r.end_x, cand.start_y - r.end_y)
        ok = cand[(dist <= run_speed * gap + slack_m) & ~cand.index.isin(taken)]
        if not len(ok):
            continue
        k = int(dist[ok.index].idxmin())
        next_of[i] = k
        taken.add(k)
    for i in tl.sort_values("start").index:          # propagate chain ids forward along links
        if i in next_of:
            chain[next_of[i]] = chain[i]
    out = tl.copy()
    out["chain"] = chain
    return out


# -- formation roles ---------------------------------------------------------------------------

def _attacking_frame(xy, side, half, sides):
    """Rotate positions so the team always attacks towards +x (the far goal), per half."""
    if sides[half][side] == "left":
        return xy
    out = xy.copy()
    out[..., 0] = fx.LENGTH - out[..., 0]
    out[..., 1] = fx.WIDTH - out[..., 1]
    return out


def _assign(points, template):
    """Hungarian match of points (k, 2) to template roles (r, 2). Returns role index per point,
    -1 for points left over when there are more points than roles."""
    cost = ((points[:, None, :] - template[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    out = np.full(len(points), -1)
    out[rows] = cols
    return out


def learn_template(xy_team, n_roles=10, iters=8, sample_every=10, seed=0):
    """Average formation (n_roles, 2), relative to the team centre, from frames where exactly
    n_roles outfield players are tracked. Alternates: assign players to roles (Hungarian), then
    move each role to the mean of what was assigned to it (Bialkowski et al. 2014, simplified)."""
    full = np.flatnonzero(np.isfinite(xy_team[:, :, 0]).sum(1) == n_roles)[::sample_every]
    rel = []
    for f in full:
        p = xy_team[f][np.isfinite(xy_team[f, :, 0])]
        rel.append(p - p.mean(0))
    rel = np.array(rel)                                          # (frames, n_roles, 2)
    rng = np.random.RandomState(seed)
    template = rel[rng.randint(len(rel))].copy()
    for _ in range(iters):
        sums = np.zeros_like(template)
        for p in rel:
            sums[_assign(p, template)] += p
        template = sums / len(rel)
    return template


def role_names(template):
    """Readable names from template positions: depth band (DEF/MID/ATT by x) + lane (L/C/R by y).

    L/R assume y grows to the team's left when it attacks towards +x; check one frame against the
    video and swap if the convention is the other way round.
    """
    x, y = template[:, 0], template[:, 1]
    depth = np.where(x < np.percentile(x, 33), "DEF", np.where(x > np.percentile(x, 67), "ATT", "MID"))
    lane = np.where(y > 7, "L", np.where(y < -7, "R", "C"))
    base = [f"{d}-{l}" for d, l in zip(depth, lane)]
    totals = {n: base.count(n) for n in base}
    seen, names = {}, []
    for n in base:                                               # DEF-C1, DEF-C2 when repeated
        seen[n] = seen.get(n, 0) + 1
        names.append(f"{n}{seen[n]}" if totals[n] > 1 else n)
    return names


def assign_roles(players, team, gk_slots, n_roles=10):
    """Role of every outfield slot on every graded frame.

    Returns (roles, templates, names):
      roles      (n, slots) int8, role index 0..n_roles-1, -1 if not assigned (absent, keeper,
                 extra player beyond n_roles, or outside the halves)
      templates  {(side, half): (n_roles, 2)} formation relative to the team centre
      names      {(side, half): [role names]}

    Per frame: centre the present players, Hungarian-match them to the template, then re-centre
    using the matched roles (so a missing player does not drag the centre) and match again.
    """
    n = len(players)
    halves = fx.half_index(n)
    sides = fx.defending_side(players, team)
    xy_all = fx.to_metres(players[:, :, :2])
    roles = np.full(players.shape[:2], -1, dtype=np.int8)
    templates, names = {}, {}
    for side in (HOME, AWAY):
        slots = np.flatnonzero((team == side) & (np.arange(len(team)) != gk_slots[side]))
        for half in (1, 2):
            frames = np.flatnonzero(halves == half)
            xy = _attacking_frame(xy_all[frames][:, slots], side, half, sides)
            template = learn_template(xy, n_roles)
            templates[(side, half)] = template
            names[(side, half)] = role_names(template)
            for k, f in enumerate(frames):
                ok = np.flatnonzero(np.isfinite(xy[k, :, 0]))
                if not len(ok):
                    continue
                p = xy[k, ok]
                r = _assign(p - p.mean(0), template)
                hit = r >= 0
                centre = (p[hit] - template[r[hit]]).mean(0)       # robust to missing players
                r = _assign(p - centre, template)
                roles[f, slots[ok]] = r
    return roles, templates, names


def missing_roles(roles, team, side, n_roles=10):
    """(n, n_roles) True where that role has no player assigned on that frame."""
    cols = roles[:, team == side]
    out = np.ones((len(roles), n_roles), dtype=bool)
    for r in range(n_roles):
        out[:, r] = ~(cols == r).any(1)
    return out
