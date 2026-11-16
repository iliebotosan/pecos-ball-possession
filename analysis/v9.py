"""v9 = v8 plus every add-on, each switchable, and the same pipeline on a reconstructed ball.

    from analysis import v9
    ing = v9.ingredients(ctx, kin, EK)           # restarts, contests, ghost carries, whistles, shape model
    labels = v9.build(ctx, ing)                  # everything on
    labels = v9.build(ctx, ing, ghost=False)     # leave one out
    ctx_f = v9.with_ball(ctx, xy_filled)         # same match, ball reconstructed in the gaps
    labels_f = v9.build(ctx_f, ing)

Order (each step only changes what it is about):
  1. votes: nearest team per frame; no vote while the ball is in flight (v5) or contested (both teams
     close, nobody dribbling)
  2. smooth (+-2 s majority) and carry the team through frames without a vote
  3. dead ball: slow players / ball out (+-1 s) / 5 s before a set-piece tag (v4) + whistle -> slow-down
     + ball-left-the-pitch -> restart
  4. ghost carries: a decided untracked carrier owns his carry
  5. team shape: inside ball blackouts >= 2 s, switch team if the shape model is confident (P >= 0.85 for 2 s)
  6. restarts: the decided restarting team owns the first 3 s after every set piece
  7. move every direct switch to the receiving team's first touch (within 1 s before / 0.5 s after)
  8. suspicious short possessions -> X (v8)
Ingredients never look at the ground truth.
"""

import copy

import numpy as np

from games import GAME, half_video_blocks
from starter import baseline
from analysis import features as fx
from analysis import restarts as rs, contest as ct, ghost as gh, whistle as wh, shape as sh, backdate as bd
from analysis import cascade as cs
from analysis.versions import carry_team, suspicious_policy, SMOOTH_W

FPS = fx.FPS
HALVES = [(a, z + 1) for _, a, z in half_video_blocks(GAME)]
PARTS = ("contested", "whistle", "out_dead", "ghost", "shape", "restarts", "backdate", "x_policy")


def whistle_dead(ctx, W, confirm_s=6.0):
    "D from each graded whistle to the slow-down that follows it within confirm_s (notebook_audio §8)"
    extra = np.zeros(ctx.N, bool)
    starts = np.array([a for a, _ in zip(*fx.runs(ctx.slow & ctx.graded))])
    for w in W[W.graded].start.values:
        nxt = starts[(starts >= w) & (starts <= w + confirm_s * FPS)]
        if len(nxt):
            extra[w:nxt[0]] = True
    return extra


def out_dead_mask(ctx, R):
    m = np.zeros(ctx.N, bool)
    for x in R[(R.out_frame >= 0) & (R.restart >= 0) & R.type.str.lower().isin(rs.OUT_OF_PLAY)].itertuples():
        m[x.out_frame:x.restart] = True
    return m


def shape_probability(ctx, kin, R, C):
    "P(home has the ball) from team shape, each half predicted by a model trained on the other half"
    S = cs.sources(ctx, kin, C, R)
    y = sh.pseudo_labels(ctx, S)
    F = sh.team_features(ctx)
    P = np.full(ctx.N, 0.5)
    for tr, te in ((1, 2), (2, 1)):
        m = sh.fit(F, y, ctx.half == tr)
        P[ctx.half == te] = sh.predict(m, F)[ctx.half == te]
    return P, y


def ingredients(ctx, kin, EK, whistle_thr=25.0):
    R = rs.restart_table(ctx, kin, EK)
    C = ct.contest(kin)
    GC = gh.ghost_carries(ctx, kin, EK)
    score, hz = wh.frame_score(ctx.N)
    W = wh.detect(score, hz, ctx.graded, thr=whistle_thr)
    P, y = shape_probability(ctx, kin, R, C)
    return {"R": R, "C": C, "GC": GC, "W": W, "whistle_dead": whistle_dead(ctx, W), "out_dead": out_dead_mask(ctx, R),
            "P": P, "pseudo": y, "events": bd.event_frames(EK), "ok_real": ctx.ok.copy()}


def build(ctx, ing, **parts):
    on = {p: parts.get(p, True) for p in PARTS}
    raw = ctx.codes.copy()
    raw[ctx.in_flight] = -1
    if on["contested"]:
        raw[ing["C"]["contested"]] = -1
    dead = ctx.dead_any.copy()
    if on["whistle"]:
        dead |= ing["whistle_dead"]
    if on["out_dead"]:
        dead |= ing["out_dead"]
    if "dead_add" in ing:                      # optional precise boundaries (analysis.whistle_context)
        dead |= ing["dead_add"]
    if "dead_remove" in ing:
        dead &= ~ing["dead_remove"]
    lab = np.where(dead, "D", carry_team(baseline.smooth(raw, SMOOTH_W)))
    if on["ghost"]:
        lab = gh.ghost_addon(lab, ing["GC"])
    if on["shape"]:
        lab = sh.blackout_addon(lab, ing["P"], ing["ok_real"], HALVES, min_gap_s=2, thr=0.85, min_s=2)
    if on["restarts"]:
        lab = rs.restart_addon(lab, ing["R"], dead_from_out=False)
    if on["backdate"]:
        lab = bd.move_switches(lab, ing["events"], HALVES, back_s=1.0, fwd_s=0.5)
    if on["x_policy"]:
        lab = suspicious_policy(ctx, lab, "X")
    return lab


def with_ball(ctx, xy, close_only_where=None, close_m=2.0):
    """A shallow copy of ctx whose ball votes come from `xy` (e.g. the reconstructed track).
    Dead-ball signals stay those of the real ball. If close_only_where is a mask (the filled frames),
    those frames only vote when a player is within close_m of the reconstructed ball."""
    c = copy.copy(ctx)
    c.bxy = xy
    c.ok = np.isfinite(xy[:, 0])
    c.ball_speed_k2 = fx.displacement_speed(xy, 2)
    c.codes, c.dist = c.nearest_team_codes(xy)
    if close_only_where is not None:
        far = close_only_where & ~(c.dist <= close_m)
        c.codes = c.codes.copy()
        c.codes[far] = -1
    c.in_flight, c.flights, c.touch, c.touch_team = c.flights_from(c.ok, c.dist, c.codes)
    return c
