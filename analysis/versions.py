"""The rule-based versions v0-v8, as one importable module.

Exactly the pipeline of notebook_ball sections 5-6 (which was rebuilt from notebook2), moved here
so other notebooks can compare against the same versions without copying cells:

    from analysis.versions import Context, build_versions
    ctx = Context.load()              # data + every shared signal (dead-ball masks, flights, ...)
    v = build_versions(ctx)           # {"v0": letters, ..., "v8": letters}

| version | what it does |
|---|---|
| v0 | organisers' baseline: nearest player, majority vote +/-2 s, ball not seen -> D |
| v1 | v0 but ball not seen -> carry the last team |
| v2 | v1 + D where outfield speed stays low (>= 4 s below 1.3 m/s) |
| v3 | v2 + D where the ball is out of the pitch (+/-1 s) |
| v4 | v3 + D for 5 s before every set-piece tag |
| v5 | v4 pipeline, but frames with the ball in flight get no vote |
| v6 | v4 pipeline, but only frames with a player within 2 m of the ball vote |
| v7 | v5, then suspicious possessions (< 3 s, no 1 s of control, no pass) merged into a neighbour |
| v8 | v5, then suspicious possessions written as X (the scorer fills X with the previous label) |
"""

import numpy as np
import pandas as pd

from games import GAME, half_video_blocks
from evaluation import possession_metrics as pm
from starter import baseline, load_smart_tagging
from analysis.atd import load_arrays, HOME, AWAY
from analysis import features as fx

FPS = fx.FPS

# dead-ball rules
SLOW_MS, SPEED_SMOOTH_S, SLOW_MIN_S = 1.3, 3, 4
OOB_PAD_S = 1
PRE_SET_S = 5
SMOOTH_W = 50
# flights and suspicious possessions
TOUCH_M = 2.0
FAST, BRIDGE_S, MIN_FLIGHT_S = 8.0, 0.5, 0.3
SHORT_S, CONTROL_S = 3.0, 1.0


def long_runs(mask, min_s):
    out = np.zeros_like(mask)
    for a, b in zip(*fx.runs(mask)):
        if b - a >= min_s * FPS:
            out[a:b] = True
    return out


def bridge(mask, gap_s, allowed):
    "fill False gaps of at most gap_s between True runs, only where `allowed`"
    out = mask.copy()
    for a, b in zip(*fx.runs(~mask)):
        if a > 0 and b < len(mask) and (b - a) <= gap_s * FPS and allowed[a:b].all():
            out[a:b] = True
    return out


def longest_run_s(mask):
    s, e = fx.runs(mask)
    return (e - s).max() / FPS if len(s) else 0.0


class Context:
    """The match arrays and every signal the versions share. Build once with Context.load()."""

    @classmethod
    def load(cls):
        c = cls()
        t = load_arrays()
        c.ball, c.players, c.team = t["ball"], t["players"], t["team"]
        c.N = len(c.ball)
        c.graded = fx.graded_mask(c.N)
        c.half = fx.half_index(c.N)
        c.bxy = fx.to_metres(c.ball[:, :2])
        c.pxy = fx.to_metres(c.players[:, :, :2])
        c.ok = np.isfinite(c.bxy[:, 0])
        c.tags = load_smart_tagging.load()
        c.set_piece_starts = np.array([e["start_frame"] for e in c.tags if e["code"].upper() == "SET PIECES"])

        # dead-ball signals
        v_player = fx.player_speed(c.players)
        c.speed_out = np.nanmean(fx.outfield_speed(v_player, c.players, c.team,
                                                   fx.goalkeepers(c.players, c.team)), axis=1)
        c.speed_out_smooth = pd.Series(c.speed_out).rolling(SPEED_SMOOTH_S * FPS, center=True,
                                                            min_periods=1).mean().values
        c.slow = long_runs(c.speed_out_smooth < SLOW_MS, SLOW_MIN_S)
        c.oob = fx.ball_out_of_pitch(c.ball)
        k = OOB_PAD_S * FPS
        c.out_of_pitch = np.convolve(c.oob.astype(int), np.ones(2 * k + 1, int), mode="same") > 0
        c.pre_set = np.zeros(c.N, bool)
        for s0 in c.set_piece_starts:
            c.pre_set[max(0, s0 - PRE_SET_S * FPS):s0] = True
        c.dead_any = c.slow | c.out_of_pitch | c.pre_set

        # nearest team and flights on the real ball track
        c.ball_speed_k2 = fx.ball_speed(c.ball, k=2)
        c.codes, c.dist = c.nearest_team_codes(c.bxy)
        c.in_flight, c.flights, c.touch, c.touch_team = c.flights_from(c.ok, c.dist, c.codes)
        return c

    def nearest_by_team(self, xy):
        "(n, 2) distance from positions xy (n, 2) to the nearest home / away player at the same frames"
        d = np.linalg.norm(self.pxy[:len(xy)] - xy[:, None, :], axis=2)
        return np.stack([np.nanmin(np.where((self.team == s)[None], d, np.nan), axis=1)
                         for s in (HOME, AWAY)], axis=1)

    def nearest_team_codes(self, xy):
        "per frame: 0 home / 1 away for the nearest player to xy, -1 where xy is missing; plus that distance"
        with np.errstate(invalid="ignore"):
            d = self.nearest_by_team(xy)
        codes = np.where(d[:, 0] <= d[:, 1], 0, 1).astype(np.int8)
        dist = np.fmin(d[:, 0], d[:, 1])
        codes[~np.isfinite(dist)] = -1
        return codes, dist

    def flights_from(self, xy_ok, dist, codes):
        "in-flight mask and a flight table (kicker / receiver team): 'free and fast' ball"
        touch = xy_ok & (dist <= TOUCH_M)
        team_ = np.where(codes == 0, "H", "A")
        free = bridge(xy_ok & (dist > TOUCH_M), BRIDGE_S, ~xy_ok) & self.graded
        mask = np.zeros(self.N, bool)
        rows = []
        for a, b in zip(*fx.runs(free)):
            v = self.ball_speed_k2[a:b]
            if (b - a) < MIN_FLIGHT_S * FPS or not np.isfinite(v).any() or np.nanmax(v) < FAST:
                continue
            mask[a:b] = True
            lo = max(0, a - 2 * FPS)
            before = np.flatnonzero(touch[lo:a])
            after = np.flatnonzero(touch[b:b + 2 * FPS])
            rows.append({"start": a, "kicker": team_[lo + before[-1]] if len(before) else "",
                         "receiver": team_[b + after[0]] if len(after) else ""})
        return mask, pd.DataFrame(rows, columns=["start", "kicker", "receiver"]), touch, team_


def carry_team(letters):
    "replace D by the last team inside each half (back-fill at the head of a half)"
    out = letters.copy()
    for _, a, z in half_video_blocks(GAME):
        out[a:z + 1] = pd.Series(out[a:z + 1]).replace("D", np.nan).ffill().bfill().fillna("H").values
    return out


def pipeline(ctx, raw):
    "smooth -> carry the team through gaps -> dead-ball rules"
    return np.where(ctx.dead_any, "D", carry_team(baseline.smooth(raw, SMOOTH_W)))


def possessions(letters):
    "(start, end, team, prev, next) for every H/A run inside the halves"
    codes = pm.label_codes(letters)
    segs = pm.segments(codes, [(a, z + 1) for _, a, z in half_video_blocks(GAME)])
    out = []
    for k, (a, b, c) in enumerate(segs):
        if c in (pm.H, pm.A):
            prev = pm.LABELS[segs[k - 1][2]] if k > 0 and segs[k - 1][1] == a else "|"
            nxt = pm.LABELS[segs[k + 1][2]] if k + 1 < len(segs) and segs[k + 1][0] == b else "|"
            out.append((a, b, pm.LABELS[c], prev, nxt))
    return out


def suspicious_policy(ctx, letters, policy):
    """Possessions shorter than SHORT_S with no CONTROL_S of own control and no completed pass:
    'merge' into the previous (else next) team, or overwrite with `policy` ('X' or 'D')."""
    out = np.asarray(letters).copy()
    F_, touch, team_ = ctx.flights, ctx.touch, ctx.touch_team
    for a, b, L, prev, nxt in possessions(letters):
        own = touch[a:b] & (team_[a:b] == L)
        passes = ((F_.start >= a) & (F_.start < b) & (F_.kicker == L) & (F_.receiver == L)).sum() if len(F_) else 0
        if (b - a) / FPS < SHORT_S and longest_run_s(own) < CONTROL_S and passes == 0:
            if policy == "merge":
                fill = prev if prev in ("H", "A") else (nxt if nxt in ("H", "A") else None)
                if fill:
                    out[a:b] = fill
            else:
                out[a:b] = policy
    return out


def build_versions(ctx):
    "v0..v8 from the real ball track"
    codes = ctx.codes
    v = {}
    v["v0"] = baseline.smooth(codes, SMOOTH_W)
    v["v1"] = carry_team(v["v0"])
    v["v2"] = np.where(ctx.slow, "D", v["v1"])
    v["v3"] = np.where(ctx.slow | ctx.out_of_pitch, "D", v["v1"])
    v["v4"] = np.where(ctx.dead_any, "D", v["v1"])
    r5 = codes.copy(); r5[ctx.in_flight] = -1
    r6 = codes.copy(); r6[~ctx.touch] = -1
    v["v5"] = pipeline(ctx, r5)
    v["v6"] = pipeline(ctx, r6)
    v["v7"] = suspicious_policy(ctx, v["v5"], "merge")
    v["v8"] = suspicious_policy(ctx, v["v5"], "X")
    return v
