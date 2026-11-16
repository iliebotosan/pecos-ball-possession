"""Kicks, receptions and passes from the ball track.

    from analysis import passes as ps
    kin = ps.kinematics(t["ball"], t["players"], t["team"])   # per-frame ball motion + nearest players
    E = ps.events(kin, graded)                                # ball events from changes in velocity
    P = ps.passes(kin, E, graded, dead)                       # one row per kick: kicker, receiver, outcome

The ball track is already smoothed by Metrica, so a kick is spread over a few frames. Events are
therefore found from the change in velocity over +/- K frames (dv), not from acceleration
(notebook_ball, sections 2-3).

A pass is a kick followed by a reception:
  1. kick: a dv peak with a player within TOUCH_M, where the ball speeds up by KICK_GAIN to at
     least KICK_MIN;
  2. the ball leaves: the first frame after the kick where nobody is within LEAVE_M of it (or it is
     not tracked);
  3. reception: from then on, the first frame where some player stays within RECEIVE_M of the
     ball for RECEIVE_HOLD tracked frames in a row. That player's team is the receiver;
  4. outcome: 'completed' (same team), 'intercepted' (other team), 'dead' (a dead-ball frame
     comes first), 'lost' (no reception within MAX_FLIGHT_S, or the half ends);
  5. control: in the CONTROL_S after the reception the receiving team is the nearest team on at
     least CONTROL_SHARE of the frames where someone is within CONTROL_M of the ball, or it makes
     the next kick.

predict_receiver() guesses the receiver from the kick alone (direction + where the players are),
for when the ball is not seen arriving. It never looks at the reception.

Nothing here is fitted on the ground truth. Teams are letters 'H' / 'A'; '' means unknown.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from analysis.atd import HOME, AWAY
from analysis import features as fx

FPS = fx.FPS

# events (as notebook_ball section 3)
K = 6                  # frames each side of the event (0.24 s)
DV_THR = 5.0           # m/s change in velocity to count as an event
TOUCH_M = 2.0          # a player this close gets the event
MIN_GAP_S = 0.3        # two events at least this far apart

# kicks and passes
KICK_GAIN = 3.0        # m/s the ball must speed up across a kick ...
KICK_MIN = 5.0         # ... to at least this speed
LEAVE_M = 2.0          # the ball has left the kicker once nobody is this close
LEAVE_MAX_S = 1.0      # if it has not left by then, it was a touch while dribbling, not a pass
RECEIVE_M = 2.0        # the first player this close afterwards receives it ...
RECEIVE_HOLD = 3       # ... for this many tracked frames in a row (0.12 s)
MAX_FLIGHT_S = 6.0     # no reception within this long = lost
CONTROL_S = 1.5        # after the reception, look this long for control ...
CONTROL_M = 3.0        # ... counting frames where someone is this close to the ball ...
CONTROL_SHARE = 0.6    # ... and the receiving team is nearest on at least this share of them

# receiver prediction from the kick alone
PRED_LOOK_S = 0.5      # players' positions this long after the kick
PRED_MIN_M, PRED_MAX_M = 3.0, 60.0   # candidates in front of the ball, between these distances
PRED_ALONG = 0.15      # score = distance from the line + this x distance along it

LETTER = np.array(["H", "A"])


def kinematics(ball, players, team):
    """Per-frame ball motion and nearest players, all in metres.

    Returns a dict:
      bxy (n, 2), ok (n,)            ball position, tracked flag
      vel (n, 2), speed (n,)         central-difference velocity, m/s
      dv (n,)                        |vel(t + K) - vel(t - K)|, m/s
      near (n, 2)                    distance to the nearest home / away player (NaN if none)
      slot (n, 2)                    that player's slot index (-1 if none)
      d_near (n,), team_near (n,)    distance to the nearest player of either team, and his team
                                     (0 home, 1 away, -1 nobody / ball not tracked)
    """
    n = len(ball)
    bxy = fx.to_metres(ball[:, :2])
    pxy = fx.to_metres(players[:, :, :2])
    ok = np.isfinite(bxy[:, 0])

    vel = np.full_like(bxy, np.nan)
    vel[1:-1] = (bxy[2:] - bxy[:-2]) * FPS / 2
    speed = np.linalg.norm(vel, axis=1)
    dv = np.full(n, np.nan)
    dv[K:-K] = np.linalg.norm(vel[2 * K:] - vel[:-2 * K], axis=1)

    dist = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = np.full((n, 2), np.nan)
    slot = np.full((n, 2), -1, dtype=np.int16)
    rows = np.arange(n)
    for col, side in enumerate((HOME, AWAY)):
        d = np.where((team == side)[None, :], dist, np.inf)
        best = d.argmin(1)
        val = d[rows, best]
        has = np.isfinite(val)
        near[has, col] = val[has]
        slot[has, col] = best[has]

    d_near = np.fmin(near[:, 0], near[:, 1])
    team_near = np.where(np.isnan(near[:, 1]) | (near[:, 0] <= near[:, 1]), HOME, AWAY).astype(np.int8)
    team_near[~np.isfinite(d_near)] = -1
    return {"bxy": bxy, "pxy": pxy, "ok": ok, "vel": vel, "speed": speed, "dv": dv, "near": near,
            "slot": slot, "d_near": d_near, "team_near": team_near, "team": team}


def events(kin, graded):
    """One row per velocity-change event on graded frames (notebook_ball section 3, plus is_kick)."""
    dv, vel, d_near, tn = kin["dv"], kin["vel"], kin["d_near"], kin["team_near"]
    peaks, _ = find_peaks(np.nan_to_num(dv * graded), height=DV_THR, distance=int(MIN_GAP_S * FPS))
    peaks = peaks[(peaks >= K) & (peaks < len(dv) - K)]
    before = np.linalg.norm(vel[peaks - K], axis=1)
    after = np.linalg.norm(vel[peaks + K], axis=1)
    close = d_near[peaks] <= TOUCH_M
    E = pd.DataFrame({"frame": peaks, "dv": dv[peaks], "speed_before": before, "speed_after": after,
                      "player_m": d_near[peaks],
                      "team": np.where(tn[peaks] >= 0, LETTER[np.maximum(tn[peaks], 0)], "")})
    E["kind"] = np.select([~close, after > before + 3, after < before - 3],
                          ["no player near", "kick (ball speeds up)", "reception (ball slows down)"],
                          "touch / deflection")
    E["is_kick"] = close & (after >= before + KICK_GAIN) & (after >= KICK_MIN)
    return E


def _held(close, hold):
    "held[g] = close[g : g + hold] all True"
    c = np.concatenate([[0], np.cumsum(close.astype(np.int32))])
    out = np.zeros(len(close), bool)
    out[:len(close) - hold + 1] = (c[hold:] - c[:-hold]) == hold
    return out


def passes(kin, E, graded, dead):
    """One row per kick: kicker, receiver, outcome, flight and control. See the module docstring."""
    ok, d_near, tn, slot = kin["ok"], kin["d_near"], kin["team_near"], kin["slot"]
    bxy, n = kin["bxy"], len(ok)
    near_ball = ok & (d_near <= LEAVE_M)
    held = _held(ok & (d_near <= RECEIVE_M), RECEIVE_HOLD)
    kicks = E[E.is_kick].frame.values
    kick_team = tn[kicks]

    rows = []
    for i, f in enumerate(kicks):
        kt = tn[f]
        row = {"frame": int(f), "kicker": LETTER[kt], "kicker_slot": int(slot[f, kt]),
               "kick_speed": float(np.linalg.norm(kin["vel"][f + K])),
               "kicker_margin_m": float(abs(kin["near"][f, 0] - kin["near"][f, 1]))
               if np.isfinite(kin["near"][f]).all() else np.inf,
               "left": -1, "reception": -1, "receiver": "", "receiver_slot": -1,
               "outcome": "", "controlled": False, "flight_s": np.nan, "flight_m": np.nan,
               "unseen_s": np.nan}

        # 2. when does the ball leave the kicker?
        stop_leave = min(n, f + int(LEAVE_MAX_S * FPS))
        g = f + 1
        while g < stop_leave and graded[g] and near_ball[g]:
            g += 1
        if g >= stop_leave or not graded[g]:
            row["outcome"] = "no flight"            # stayed on the foot: a dribble touch
            rows.append(row)
            continue
        row["left"] = int(g)

        # 3. first held reception after that, unless a dead ball or the half end comes first
        stop = min(n, f + int(MAX_FLIGHT_S * FPS))
        r = -1
        while g < stop and graded[g]:
            if dead[g]:
                row["outcome"] = "dead"
                break
            if held[g]:
                r = g
                break
            g += 1
        if r < 0:
            row["outcome"] = row["outcome"] or "lost"
            rows.append(row)
            continue

        rt = int(np.bincount(tn[r:r + RECEIVE_HOLD][tn[r:r + RECEIVE_HOLD] >= 0], minlength=2).argmax())
        row.update({"reception": int(r), "receiver": LETTER[rt], "receiver_slot": int(slot[r, rt]),
                    "outcome": "completed" if rt == kt else "intercepted",
                    "flight_s": (r - f) / FPS, "flight_m": float(np.linalg.norm(bxy[r] - bxy[f])),
                    "unseen_s": float((~ok[f:r]).sum() / FPS)})

        # 5. control by the receiving team
        w = slice(r, min(n, r + int(CONTROL_S * FPS)))
        close = ok[w] & (d_near[w] <= CONTROL_M)
        share = np.mean(tn[w][close] == rt) if close.any() else 0.0
        nxt = np.flatnonzero((kicks > r) & (kicks <= r + CONTROL_S * FPS))
        next_kick_mine = len(nxt) > 0 and kick_team[nxt[0]] == rt
        row["controlled"] = bool(share >= CONTROL_SHARE or next_kick_mine)
        rows.append(row)

    P = pd.DataFrame(rows)
    P["self"] = (P.outcome == "completed") & (P.receiver_slot == P.kicker_slot)   # same tracklet got it back
    return P


def predict_receiver(kin, P):
    """Receiver team guessed from the kick alone: the player closest to the kick's line of travel
    (PRED_LOOK_S after the kick), in front of the ball. Adds 'pred_team' and 'pred_slot'."""
    bxy, pxy, vel, team = kin["bxy"], kin["pxy"], kin["vel"], kin["team"]
    n = len(bxy)
    look = int(PRED_LOOK_S * FPS)
    pred_team, pred_slot = [], []
    for r in P.itertuples():
        f = r.frame
        v = vel[min(n - 1, f + K)]
        sp = np.linalg.norm(v)
        if not np.isfinite(sp) or sp == 0:
            pred_team.append(""); pred_slot.append(-1); continue
        u = v / sp
        p = pxy[min(n - 1, f + look)] - bxy[f]                      # players relative to the kick
        along = p @ u
        perp = np.abs(p[:, 0] * u[1] - p[:, 1] * u[0])
        score = perp + PRED_ALONG * along
        ok = np.isfinite(score) & (along >= PRED_MIN_M) & (along <= PRED_MAX_M)
        ok[r.kicker_slot] = False
        if not ok.any():
            pred_team.append(""); pred_slot.append(-1); continue
        j = int(np.flatnonzero(ok)[np.argmin(score[ok])])
        pred_team.append(LETTER[team[j]]); pred_slot.append(j)
    P = P.copy()
    P["pred_team"] = pred_team
    P["pred_slot"] = pred_slot
    return P


def pass_addon(letters, P, controlled_only=True):
    """Possession from passes, on top of an existing label array:
    - completed pass: the kicker's team keeps the ball through the flight (kick -> reception), so a
      defender the ball flies past cannot create a fake possession;
    - intercepted pass (with control, unless controlled_only=False): kicker's team through the
      flight, the receiver's team from the reception for CONTROL_S.
    Dead-ball frames are never overwritten. Passes are applied in time order."""
    out = np.asarray(letters).copy()
    for r in P[P.outcome.isin(["completed", "intercepted"])].sort_values("frame").itertuples():
        seg = slice(r.frame, r.reception)
        out[seg] = np.where(out[seg] == "D", "D", r.kicker)
        if r.outcome == "intercepted" and (r.controlled or not controlled_only):
            seg = slice(r.reception, r.reception + int(CONTROL_S * FPS))
            out[seg] = np.where(out[seg] == "D", "D", r.receiver)
    return out
