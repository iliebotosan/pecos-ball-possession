"""Crowded balls and dribbles.

    from analysis import contest as ct
    C = ct.contest(kin)
    C["crowded"]     (n,)  players of BOTH teams within CROWD_M of the ball, and neither team clearly closer
    C["dribble"]     (n,)  a tracked player moves WITH the ball (close, same velocity) for >= DRIBBLE_S
    C["dribbler"]    (n,)  his team, 0 home / 1 away, -1 none
    C["contested"]   (n,)  crowded and nobody dribbling: the frames where "nearest player" is a coin toss

Why: in a crowd the nearest player flips between teams every few frames, and every flip is a
candidate fake possession. A dribble is the exception: someone visibly carries the ball out of it.
"""

import numpy as np

from analysis import features as fx
from analysis.atd import HOME, AWAY

FPS = fx.FPS
CROWD_M = 3.0        # both teams have a player this close ...
CROWD_MARGIN_M = 1.0 # ... and the nearer team is less than this much nearer
CARRY_M = 2.0        # a dribbler is this close to the ball ...
CO_MOVE_MS = 1.5     # ... his velocity within this of the ball's ...
MIN_BALL_MS = 1.0    # ... the ball actually moving ...
DRIBBLE_S = 0.4      # ... for at least this long (gaps up to BRIDGE_S bridged)
BRIDGE_S = 0.2


def contest(kin):
    ok, near, bxy, pxy, vel, team = kin["ok"], kin["near"], kin["bxy"], kin["pxy"], kin["vel"], kin["team"]
    n = len(ok)
    with np.errstate(invalid="ignore"):
        crowded = ok & (near[:, 0] <= CROWD_M) & (near[:, 1] <= CROWD_M) & (np.abs(near[:, 0] - near[:, 1]) < CROWD_MARGIN_M)

        pvel = np.full_like(pxy, np.nan)
        pvel[1:-1] = (pxy[2:] - pxy[:-2]) * FPS / 2
        d = np.linalg.norm(pxy - bxy[:, None, :], axis=2)
        vd = np.linalg.norm(pvel - vel[:, None, :], axis=2)
        moving = np.linalg.norm(vel, axis=1) >= MIN_BALL_MS
        co = (d <= CARRY_M) & (vd <= CO_MOVE_MS) & moving[:, None]
    co &= ok[:, None]
    any_co = co.any(1)
    # the co-moving player closest to the ball
    j = np.argmin(np.where(co, d, np.inf), axis=1)
    who = np.where(any_co, team[j], -1).astype(np.int8)

    # sustained: bridge short breaks, keep runs >= DRIBBLE_S
    mask = any_co.copy()
    s, e = fx.runs(~mask)
    for a, b in zip(s, e):
        if 0 < a and b < n and (b - a) <= BRIDGE_S * FPS:
            mask[a:b] = True
    dribble = np.zeros(n, bool)
    dribbler = np.full(n, -1, np.int8)
    for a, b in zip(*fx.runs(mask)):
        if b - a >= DRIBBLE_S * FPS:
            w = who[a:b][who[a:b] >= 0]
            if len(w):
                dribble[a:b] = True
                dribbler[a:b] = np.bincount(w, minlength=2).argmax()
    return {"crowded": crowded, "dribble": dribble, "dribbler": dribbler, "contested": crowded & ~dribble}
