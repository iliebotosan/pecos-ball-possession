"""The current best model, rebuilt in one call (for submissions and as the reference for new ideas).

    from analysis import best
    B = best.Bundle.load()          # data, ingredients, refined referee blows
    labels = best.final(B)          # the submitted model: v8 + whistle + three finishing touches
    labels = best.v8_whistle(B)     # v8 + whistle (refined blows, detected start, dead until the slow-down)

"v8 + whistle (refined, detected start, slow)" from notebook_whistle §5: every refined referee blow
(learned from kick-offs, ends of half and fouls before free kicks) starts a dead ball at its detected
start, running to the slow-down that follows within 6 s; blows with no slow-down are dropped. All else
is v8.

final() adds three finishing touches (notebook_frontier). None adds or removes a possession; each only moves
a boundary to a better place:
  1. restart end: at a restart announced by a start whistle and confirmed by a set-piece restart within 3 s,
     the dead ball runs right up to the restart kick and not past it (the speed signal lags the kick);
  2. tag-confirmed long passes: a completed or intercepted kick with a flight of >= 1 s and >= 15 m that lies
     inside a LONG BALL tag of the kicking team: the kicking team keeps the ball through the flight;
  3. blackout split: the ball disappears with one team and reappears with the other while play stays live
     (no dead ball inside): the switch goes to the middle of the blackout instead of near its end.
"""

import numpy as np

from analysis import features as fx, passes as ps, restarts as rs, tagfix as tf, whistle as wh, v9
from analysis import whistle_context as wc
from analysis.versions import Context, build_versions
from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks

FPS = fx.FPS
HALVES = [(a, z + 1) for _, a, z in half_video_blocks(GAME)]
RESTART_CONFIRM_S = 3.0                       # 1. a start whistle counts if a set-piece restart is this close
LONG_MIN_S, LONG_MIN_M, LONG_TAG_PAD_S = 1.0, 15.0, 2.0   # 2. long pass: flight length, distance, tag slack
SPLIT_MIN_GAP_S = 1.0                         # 3. shortest blackout that is split

OFF = {p: False for p in v9.PARTS}


def referee_blows(ctx, R, thr=20.0):
    score, hz = wh.frame_score(ctx.N)
    W = wh.detect(score, hz, ctx.graded, thr=thr)
    Wf = wh.event_features(W, score, hz)
    M = wh.match_anchors(wh.anchors(ctx, R, ctx.tags), Wf)
    prof = wh.learn_profile(Wf, M[M.event >= 0].event.values)
    return Wf[wh.is_referee(Wf, prof) & Wf.graded].copy(), prof


class Bundle:
    @classmethod
    def load(cls):
        b = cls()
        b.ctx = Context.load()
        b.kin = ps.kinematics(b.ctx.ball, b.ctx.players, b.ctx.team)
        b.EK = ps.events(b.kin, b.ctx.graded)
        b.V = build_versions(b.ctx)
        b.ING = v9.ingredients(b.ctx, b.kin, b.EK)
        b.REF, b.profile = referee_blows(b.ctx, b.ING["R"])
        return b


def v8_whistle(b, **extra_parts):
    ing = {**b.ING, "whistle_dead": wh.dead_from_whistles(b.ctx, b.REF.start.values, until="slow")}
    return v9.build(b.ctx, ing, **{**OFF, "x_policy": True, "whistle": True, **extra_parts})


def restart_end_adjustments(b):
    "1. (add, remove) dead-ball masks: dead ball ends at restarts announced by a confirmed start whistle"
    R = b.ING["R"]
    B = wc.classify(b.ctx, b.kin, b.EK, R, b.REF)
    sp = R[R.restart >= 0].restart.values
    confirmed = np.array([r >= 0 and len(sp) > 0 and np.abs(sp - r).min() <= RESTART_CONFIRM_S * FPS
                          for r in B.restart])
    B = B.copy()
    B.loc[(B["class"] == "start") & ~confirmed, "class"] = "unclear"
    return wc.dead_adjustments(b.ctx, B, R, stop=None, end_at_restart="whistle")


def tagged_long_passes(b):
    "2. long kicks (completed or intercepted) inside a LONG BALL tag of the kicking team"
    ctx = b.ctx
    P = ps.passes(b.kin, b.EK, ctx.graded, ctx.dead_any)
    home, away = tf.team_names(ctx.tags)
    tags = [(e["start_frame"], e["end_frame"], {home: "H", away: "A"}.get(e["tags"].get("Team") or "", ""))
            for e in ctx.tags if e["code"].upper() == "LONG BALL"]
    inside = np.array([any(s - LONG_TAG_PAD_S * FPS <= r.frame <= e and t in ("", r.kicker) for s, e, t in tags)
                       for r in P.itertuples()], bool)
    return P[P.outcome.isin(["completed", "intercepted"]).values & inside
             & (P.flight_s >= LONG_MIN_S).values & (P.flight_m >= LONG_MIN_M).values]


def blackout_split(ctx, labels):
    "3. ball lost with one team, back with the other, live throughout: switch in the middle of the blackout"
    f = np.array(pm.LABELS)[pm.fill_abstentions(pm.label_codes(np.asarray(labels)), HALVES)]
    out = np.asarray(labels).copy()
    g = ctx.graded
    for a, b in zip(*fx.runs(~ctx.ok & g)):
        if a < 1 or b >= ctx.N or (b - a) < SPLIT_MIN_GAP_S * FPS or not (g[a - 1] and g[b]):
            continue
        t0, t1 = f[a - 1], f[b]
        if t0 in ("H", "A") and t1 in ("H", "A") and t0 != t1 and not (f[a:b] == "D").any():
            m = a + (b - a) // 2
            out[a:m], out[m:b] = t0, t1
    return out


def final(b):
    """The submitted model: v8 + whistle + restart end + tag-confirmed long passes + blackout split."""
    add, remove = restart_end_adjustments(b)
    ing = {**b.ING, "whistle_dead": wh.dead_from_whistles(b.ctx, b.REF.start.values, until="slow"),
           "dead_add": add, "dead_remove": remove}
    lab = v9.build(b.ctx, ing, **{**OFF, "x_policy": True, "whistle": True})
    lab = ps.pass_addon(lab, tagged_long_passes(b))
    return blackout_split(b.ctx, lab)
