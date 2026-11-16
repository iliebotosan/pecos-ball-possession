"""Who restarts play after every set-piece tag -- including the 36 tags with Team = N/A.

    from analysis import restarts as rs
    R = rs.restart_table(ctx, kin, EK)         # one row per SET PIECES tag
    rs.validate(R)                            # each clue vs the tag's own team, where the tag has one

Every throw-in and every kick-off is tagged N/A by the Smart Tagging. The team can be recovered
from the tracking and from football's rules. Each clue names a team or abstains:

| clue          | rule                                                                     | applies to |
|---------------|--------------------------------------------------------------------------|------------|
| `rule`        | kick-off after a goal: the team that conceded (previous GOALS tag);      | kick-offs  |
|               | kick-off of half 2: the team that did NOT kick off half 1                |            |
| `out_touch`   | the ball left the pitch; the restart goes to the opponent of the last    | throw-ins, corners, |
|               | team that touched it before it went out                                  | goal kicks |
| `thrower`     | at the restart, the player nearest the ball stands on / beyond the       | throw-ins  |
|               | touchline (within THROW_LINE_M) -- his team                              |            |
| `first_touch` | team of the player who restarts it (the first kick/touch after the tag)  | all        |

Cascade (most reliable first, see validate()): rule -> agreement of the others -> out_touch ->
thrower -> first_touch. The decided team and how it was decided are in `team` and `by`.

Timing. The tag start is not the restart moment. The restart frame is the first kick event with a
player within 2 m in [tag start - BEFORE_S, tag start + AFTER_S]; if there is none, the first frame
where a player holds the ball. `out_frame` is when the ball last left the pitch before it, if it did
within OUT_LOOK_S -- the moment the dead ball really starts.
"""

import numpy as np
import pandas as pd

from analysis import features as fx
from analysis import passes as ps

FPS = fx.FPS
LENGTH, WIDTH = fx.LENGTH, fx.WIDTH

BEFORE_S, AFTER_S = 3.0, 10.0      # search window for the restart around the tag start
OUT_LOOK_S = 45.0                  # the ball must have gone out at most this long before the restart
LAST_TOUCH_S = 3.0                 # last touch before going out: within this long before the exit
THROW_LINE_M = 1.5                 # thrower stands within this distance of the touchline (or beyond)
HOLD_S = 3.0                       # restart_addon: the restarting team owns the first HOLD_S

OUT_OF_PLAY = {"throw-in", "corner kick", "goal kick"}


def _letter(code):
    return "H" if code == 0 else ("A" if code == 1 else "")


def _team_letter(tag_team):
    t = (tag_team or "").lower()
    return "H" if t.startswith("barcelona") else ("A" if t.startswith("at") else "")


def restart_table(ctx, kin, EK):
    """One row per SET PIECES tag with every clue, the decided team and the restart timing."""
    ok, dn, tn, bxy, pxy = kin["ok"], kin["d_near"], kin["team_near"], kin["bxy"], kin["pxy"]
    N = ctx.N
    touch = ok & (dn <= ps.TOUCH_M)
    held = ps._held(ok & (dn <= ps.RECEIVE_M), ps.RECEIVE_HOLD)
    ev = EK[EK.player_m <= ps.TOUCH_M].frame.values
    tags = sorted(ctx.tags, key=lambda e: e["start_frame"])
    goals = [(e["start_frame"], _team_letter(e["tags"].get("Team"))) for e in tags if e["code"].upper() == "GOALS"]

    rows = []
    for e in tags:
        if e["code"].upper() != "SET PIECES":
            continue
        s0 = e["start_frame"]
        typ = (e["tags"].get("Type") or "").strip()
        lo, hi = max(0, s0 - int(BEFORE_S * FPS)), min(N - 1, s0 + int(AFTER_S * FPS))
        k = ev[(ev >= lo) & (ev <= hi)]
        if len(k):
            r = int(k[0])
        else:
            h = np.flatnonzero(held[lo:hi])
            r = int(lo + h[0]) if len(h) else -1

        row = {"tag_start": s0, "type": typ, "tag_team": _team_letter(e["tags"].get("Team")),
               "half": int(ctx.half[s0]) if 0 <= s0 < N else 0, "restart": r, "out_frame": -1,
               "rule": "", "out_touch": "", "thrower": "", "first_touch": ""}
        if r >= 0:
            row["first_touch"] = _letter(tn[r])

            # ball out of play before the restart: last frame it was tracked inside the pitch
            a = max(0, r - int(OUT_LOOK_S * FPS))
            inside = ok[a:r] & ~ctx.oob[a:r]
            outside = ok[a:r] & ctx.oob[a:r]
            if outside.any():
                last_out = a + np.flatnonzero(outside)[-1]
                ins = np.flatnonzero(inside[:last_out - a])
                if len(ins):
                    exit_f = a + ins[-1] + 1
                    row["out_frame"] = int(exit_f)
                    w0 = max(0, exit_f - int(LAST_TOUCH_S * FPS))
                    t_ = np.flatnonzero(touch[w0:exit_f])
                    if len(t_):
                        last = _letter(tn[w0 + t_[-1]])
                        row["out_touch"] = {"H": "A", "A": "H"}[last]

            # thrower on the touchline
            if "throw" in typ.lower() and ok[r]:
                d = np.linalg.norm(pxy[r] - bxy[r], axis=1)
                j = int(np.nanargmin(np.where(np.isfinite(d), d, np.inf)))
                y = pxy[r, j, 1]
                if np.isfinite(y) and (y <= THROW_LINE_M or y >= WIDTH - THROW_LINE_M):
                    row["thrower"] = _letter(ctx.team[j])
        rows.append(row)

    R = pd.DataFrame(rows)

    # rule: kick-offs
    for i, x in R.iterrows():
        tl = x.type.lower()
        if "after goal" in tl:
            prev = [g for g in goals if g[0] < x.tag_start]
            if prev and prev[-1][1]:
                R.at[i, "rule"] = {"H": "A", "A": "H"}[prev[-1][1]]
    starts = R[R.type.str.lower().str.contains("kick off") & R.type.str.lower().str.contains("start")]
    first = starts[starts.half == 1]
    if len(first) and first.first_touch.iloc[0]:
        k1 = first.first_touch.iloc[0]
        R.loc[first.index[0], "rule"] = k1
        R.loc[starts[starts.half == 2].index, "rule"] = {"H": "A", "A": "H"}[k1]

    R[["team", "by"]] = R.apply(_decide, axis=1, result_type="expand")
    return R


def _decide(x):
    "cascade: rule, then agreement, then the clues in order of reliability"
    if x.rule:
        return x.rule, "rule"
    clues = [c for c in (x.out_touch if x.type.lower() in OUT_OF_PLAY else "", x.thrower, x.first_touch) if c]
    if len(clues) >= 2 and len(set(clues)) == 1:
        return clues[0], f"{len(clues)} clues agree"
    for name in ("out_touch", "thrower", "first_touch"):
        val = x[name] if (name != "out_touch" or x.type.lower() in OUT_OF_PLAY) else ""
        if val:
            return val, name + (" (conflict)" if len(set(clues)) > 1 else "")
    return "", "none"


def validate(R):
    """Each clue (and the decision) against the tag's own team, on tags that have one."""
    known = R[R.tag_team != ""]
    out = []
    for c in ("rule", "out_touch", "thrower", "first_touch", "team"):
        s = known[known[c] != ""]
        out.append({"clue": c, "tags with this clue": len(s),
                    "agrees with tag team": (s[c] == s.tag_team).mean() if len(s) else np.nan})
    return pd.DataFrame(out).set_index("clue")


def restart_addon(letters, R, hold_s=HOLD_S, dead_from_out=True):
    """Apply decided restarts to a label array:
    - the restarting team owns [restart, restart + hold_s) wherever the labels say the ball is alive;
    - dead_from_out: D from the moment the ball left the pitch to the restart (out-of-play restarts)."""
    out = np.asarray(letters).copy()
    for x in R[(R.team != "") & (R.restart >= 0)].itertuples():
        if dead_from_out and x.out_frame >= 0 and x.type.lower() in OUT_OF_PLAY:
            out[x.out_frame:x.restart] = np.where(out[x.out_frame:x.restart] == "X", "X", "D")
        seg = slice(x.restart, x.restart + int(hold_s * FPS))
        out[seg] = np.where(np.isin(out[seg], ["D", "X"]), out[seg], x.team)
    return out


def restart_evidence(R, n, w=3.0, hold_s=2.0, w_dead=2.0):
    """Extra Viterbi evidence: team x (+ home) for the restarting team over [restart, restart + hold_s),
    dead y over [out_frame, restart) for out-of-play restarts."""
    x, y = np.zeros(n), np.zeros(n)
    for r in R[(R.team != "") & (R.restart >= 0)].itertuples():
        x[r.restart:r.restart + int(hold_s * FPS)] += w if r.team == "H" else -w
        if r.out_frame >= 0 and r.type.lower() in OUT_OF_PLAY:
            y[r.out_frame:r.restart] += w_dead
    return x, y
