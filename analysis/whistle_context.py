"""What does each referee whistle mean? Stop, start, or nothing -- from the context around it.

The referee blows to STOP play (foul, offside), to START it (kick-off, some free kicks) and during a
stoppage (a fight, a substitution). So a whistle is not a dead ball by itself. Each blow is classified
from what happens around it, and only then used to place a dead-ball boundary precisely:

| class   | rule                                                                                | effect                       |
|---------|-------------------------------------------------------------------------------------|------------------------------|
| start   | players slow before (< SLOW_MS over the PRE window) and a restart follows: a kick by | dead ball ends at the restart|
|         | a player on a still ball, or a set-piece restart (analysis.restarts), within START_S |                              |
| stop    | players moving before (>= MOVING_MS) and slowing after (after <= SLOWDOWN x before, |dead ball starts at the onset |
|         | or a slow stretch starts within STOP_CONFIRM_S)                                     |                              |
| during  | slow before and slow after, no restart                                              | nothing                      |
| unclear | anything else                                                                       | nothing                      |

Blows closer than MERGE_S are one blow (the first onset counts).

    from analysis import whistle_context as wc
    B = wc.classify(ctx, kin, EK, R, REF)          # REF: refined referee blows (notebook_whistle)
    add, remove = wc.dead_adjustments(ctx, B, R, ...)
"""

import numpy as np
import pandas as pd

from analysis import features as fx
from analysis import passes as ps

FPS = fx.FPS
MERGE_S = 2.0
PRE = (-3.0, -0.5)          # window before the onset (s) for "before" speeds
POST = (1.0, 4.0)           # window after the onset (s) for "after" speeds
SLOW_MS = 1.3
MOVING_MS = 1.6
SLOWDOWN = 0.75
START_S = 4.0
STOP_CONFIRM_S = 6.0
STILL_MS = 1.0              # a restart kick is a kick of a ball that was (nearly) still


def merge_blows(onsets, merge_s=MERGE_S):
    onsets = np.sort(np.asarray(onsets))
    out = []
    for o in onsets:
        if not out or o - out[-1] > merge_s * FPS:
            out.append(int(o))
    return np.array(out, dtype=int)


def classify(ctx, kin, EK, R, REF):
    v = ctx.speed_out                                   # outfield speed per frame, both teams
    bspeed = kin["speed"]
    kicks = EK[EK.is_kick].frame.values
    still_before = np.array([np.nanmedian(bspeed[max(0, k - 12):k - 3]) if k > 12 else np.nan for k in kicks])
    restart_kicks = kicks[np.nan_to_num(still_before, nan=99) <= STILL_MS]
    set_restarts = R[R.restart >= 0].restart.values
    slow_starts = np.array([a for a, _ in zip(*fx.runs(ctx.slow & ctx.graded))])
    rows = []
    for w in merge_blows(REF.onset.values):
        pre = v[max(0, w + int(PRE[0] * FPS)):w + int(PRE[1] * FPS)]
        post = v[w + int(POST[0] * FPS):w + int(POST[1] * FPS)]
        before, after = np.nanmean(pre), np.nanmean(post)
        lo, hi = w - int(0.5 * FPS), w + int(START_S * FPS)
        rk = restart_kicks[(restart_kicks >= lo) & (restart_kicks <= hi)]
        sr = set_restarts[(set_restarts >= lo) & (set_restarts <= hi)]
        restart = int(min(np.r_[rk, sr])) if len(rk) or len(sr) else -1
        slow_next = slow_starts[(slow_starts >= w) & (slow_starts <= w + STOP_CONFIRM_S * FPS)]
        if before < SLOW_MS and restart >= 0:
            cls = "start"
        elif before >= MOVING_MS and (after <= SLOWDOWN * before or len(slow_next)):
            cls = "stop"
        elif before < SLOW_MS and after < SLOW_MS:
            cls = "during"
        else:
            cls = "unclear"
        rows.append({"onset": int(w), "class": cls, "speed_before": before, "speed_after": after,
                     "restart": restart, "set_piece_restart": bool(len(sr)),
                     "slow_starts_at": int(slow_next[0]) if len(slow_next) else -1})
    return pd.DataFrame(rows)


def dead_adjustments(ctx, B, R, stop="slow", end_at_restart="whistle", stop_hold_s=3.0, max_stretch_s=60.0):
    """(add, remove) boolean masks for the dead ball.

    stop:  'slow'  -> D from each stop blow's onset to the slow stretch that follows (if any);
           'hold'  -> D from the onset for stop_hold_s at least (to the slow stretch if it comes);
           None    -> stop blows unused.
    end_at_restart: 'whistle' -> at restarts announced by a start blow, the dead ball runs right up to the
                    restart and not past it; 'all' -> at every set-piece restart too; None -> unused.
    """
    n = ctx.N
    add, remove = np.zeros(n, bool), np.zeros(n, bool)
    base = ctx.dead_any
    if stop:
        for r in B[B["class"] == "stop"].itertuples():
            end = r.slow_starts_at if r.slow_starts_at >= 0 else (r.onset + int(stop_hold_s * FPS) if stop == "hold" else -1)
            if stop == "hold":
                end = max(end, r.onset + int(stop_hold_s * FPS))
            if end > r.onset:
                add[r.onset:end] = True
    if end_at_restart:
        restarts = list(B[(B["class"] == "start") & (B.restart >= 0)].restart.values)
        if end_at_restart == "all":
            restarts += list(R[R.restart >= 0].restart.values)
        for r in sorted(set(int(x) for x in restarts)):
            # before the restart: dead back to the start of the stoppage it ends (the last dead frame
            # within max_stretch_s), so a gap between "players stopped" and "the kick" is dead too
            lo = max(0, r - int(max_stretch_s * FPS))
            d = np.flatnonzero(base[lo:r] | add[lo:r])
            if len(d):
                add[lo + d[-1]:r] = True
            # after the restart: play is live, whatever the (lagging) speed signal says
            k = r
            while k < n and base[k] and k - r < max_stretch_s * FPS:
                k += 1
            remove[r:k] = True
    return add, remove                              # apply as (dead | add) & ~remove: removal wins after a restart


# -- the right team after every dead ball -----------------------------------------------------

def restart_points(B, R, kin, merge_s=2.0):
    """Every restart we can place, with the restarting team:
    - set pieces (analysis.restarts): their decided team (rule / ball out / thrower / first touch);
    - start blows whose restart is not a known set piece (an untagged free kick, a drop ball): the team of
      the player who kicks the still ball.
    One row per restart frame; a start blow within merge_s of a set piece only sharpens its frame."""
    rows = [{"frame": int(r.restart), "team": r.team, "source": f"set piece ({r.by})"}
            for r in R[(R.restart >= 0) & (R.team != "")].itertuples()]
    known = np.array([r["frame"] for r in rows])
    tn = kin["team_near"]
    for b in B[(B["class"] == "start") & (B.restart >= 0)].itertuples():
        if len(known) and np.abs(known - b.restart).min() <= merge_s * FPS:
            continue
        t = tn[b.restart]
        if t >= 0:
            rows.append({"frame": int(b.restart), "team": "H" if t == 0 else "A", "source": "start blow (kicker)"})
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def restart_team_addon(letters, P, hold_s=3.0, through_next_switch=False):
    """The restarting team owns the first hold_s after each restart (dead-ball frames untouched). With
    through_next_switch, it owns everything up to the labels' next change of team instead (the whole first
    possession after the restart), capped at 20 s."""
    out = np.asarray(letters).copy()
    n = len(out)
    for r in P.itertuples():
        end = r.frame + int(hold_s * FPS)
        if through_next_switch:
            k = r.frame
            other = "A" if r.team == "H" else "H"
            while k < n and k < r.frame + 20 * FPS and out[k] != other and out[k] != "D":
                k += 1
            end = max(end, k)
        seg = out[r.frame:end]
        out[r.frame:end] = np.where(np.isin(seg, ["D", "X"]), seg, r.team)
    return out
