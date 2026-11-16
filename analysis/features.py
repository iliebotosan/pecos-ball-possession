"""Derived per-frame features built from the raw ATD arrays.

Every function takes the arrays from analysis.atd.load_arrays() and returns plain numpy arrays,
one row per video frame. Nothing here is fitted or tuned on the ground truth.

    from analysis.atd import load_arrays
    from analysis import features as fx
    t = load_arrays()
    v_ball = fx.ball_speed(t["ball"])                     # (n,)     m/s
    v_play = fx.player_speed(t["players"])                # (n, 46)  m/s
    gk = fx.goalkeepers(t["players"], t["team"])          # (n, 2)   slot index per team, -1 if none
    v_out = fx.outfield_speed(v_play, t["players"], t["team"], gk)   # (n, 2) home, away

Units: positions in the ATD are 0-1 fractions of the pitch; everything here is converted to metres.
"""

import numpy as np

from games import GAMES, GAME, half_video_blocks
from analysis.atd import HOME, AWAY

FPS = 25
LENGTH, WIDTH = GAMES[GAME]["pitch"]          # 105.3 x 68.0 m


# -- basics ------------------------------------------------------------------------------------

def to_metres(xy):
    """(..., 2) array of 0-1 pitch fractions -> metres."""
    return xy * np.array([LENGTH, WIDTH])


def graded_mask(n):
    """True on the frames that are scored (the two halves)."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def half_index(n):
    """1 or 2 inside each half, 0 elsewhere (before kickoff, halftime, after the whistle)."""
    out = np.zeros(n, dtype=np.int8)
    for half, first, last in half_video_blocks(GAME):
        out[first:last + 1] = half
    return out


def runs(mask):
    """(starts, stops) of the True runs in a boolean array, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# -- speeds ------------------------------------------------------------------------------------

def displacement_speed(xy_m, k=1):
    """Speed in m/s from positions in metres, by central difference over +/- k frames.

        speed[t] = |pos[t + k] - pos[t - k]| / (2k / FPS)

    k = 1 is the rawest estimate; a larger k averages out position jitter (the ball track is
    noisy) at the cost of blurring sudden changes like a kick. NaN wherever either end is missing,
    so a gap in the track never produces a fake speed.
    Works for (n, 2) (the ball) and (n, slots, 2) (players).
    """
    out = np.full(xy_m.shape[:-1], np.nan)
    step = np.linalg.norm(xy_m[2 * k:] - xy_m[:-2 * k], axis=-1) / (2.0 * k / FPS)
    out[k:-k] = step
    return out


def ball_speed(ball, k=2):
    """Ball speed in m/s. The provided ball-speed channel is empty on ~92% of frames, so this is
    built from displacement instead. k = 2 (a 0.16 s window) is a mild smoothing."""
    return displacement_speed(to_metres(ball[:, :2]), k)


def player_speed(players, k=1):
    """Player speed in m/s, (n, slots).

    The provided channel is filled on ~92% of tracked positions and agrees with displacement
    (correlation 0.93, median difference 0.1 m/s -- the metadata says kph but the values are m/s).
    Use it where it exists and fill the rest from displacement.
    """
    provided = players[:, :, 2].copy()
    computed = displacement_speed(to_metres(players[:, :, :2]), k)
    hole = ~np.isfinite(provided) & np.isfinite(computed)
    provided[hole] = computed[hole]
    return provided


# -- who is on the pitch -----------------------------------------------------------------------

def team_counts(players, team):
    """(n, 2) number of tracked slots per team: column 0 home, column 1 away."""
    present = np.isfinite(players[:, :, 0])
    return np.stack([present[:, team == HOME].sum(1), present[:, team == AWAY].sum(1)], axis=1)


def duplicates(players, team, radius_m=1.0, chunk=20000):
    """(n, 2) number of same-team slot pairs closer than radius_m -- likely one player tracked twice.

    Two real teammates can be that close for a moment (a wall, a celebration), so this flags
    candidates, not certainties. Computed in chunks to keep memory small.
    """
    xy = to_metres(players[:, :, :2])
    out = np.zeros((len(xy), 2), dtype=np.int16)
    for col, side in enumerate((HOME, AWAY)):
        slots = xy[:, team == side]                                   # (n, s, 2)
        upper = np.triu(np.ones((slots.shape[1],) * 2, dtype=bool), 1)
        for start in range(0, len(xy), chunk):
            block = slots[start:start + chunk]
            d = np.linalg.norm(block[:, :, None, :] - block[:, None, :, :], axis=-1)
            out[start:start + chunk, col] = ((d < radius_m) & upper).sum((1, 2))
    return out


def defending_side(players, team):
    """{half: {HOME: 'left'|'right', AWAY: ...}} -- which goal each team defends in each half.

    A team's median position over a half sits in its own half of the pitch, so whichever team is
    nearer x = 0 defends the left goal. Teams swap at halftime.
    """
    x = to_metres(players[:, :, :2])[:, :, 0]
    halves = half_index(len(x))
    out = {}
    for half in (1, 2):
        m = halves == half
        home_x = np.nanmedian(x[m][:, team == HOME])
        away_x = np.nanmedian(x[m][:, team == AWAY])
        out[half] = {HOME: "left" if home_x < away_x else "right",
                     AWAY: "right" if home_x < away_x else "left"}
    return out


def goalkeepers(players, team):
    """(n, 2) slot index of each team's goalkeeper per frame, -1 if no slot for that team.

    No player identities exist, so this is a heuristic: the goalkeeper is the tracked player of
    that team closest to its own goal line. It is right almost all the time in open play and can
    be wrong when the keeper is not tracked or comes far out (then a defender gets picked).
    Outside the halves (no defending side known) it returns -1.
    """
    x = to_metres(players[:, :, :2])[:, :, 0]
    sides = defending_side(players, team)
    halves = half_index(len(x))
    out = np.full((len(x), 2), -1, dtype=np.int16)
    for col, side in enumerate((HOME, AWAY)):
        idx = np.flatnonzero(team == side)
        for half in (1, 2):
            m = halves == half
            xs = x[m][:, idx]
            # distance to own goal line: x itself if defending the left goal, LENGTH - x otherwise
            to_goal = xs if sides[half][side] == "left" else LENGTH - xs
            to_goal = np.where(np.isfinite(to_goal), to_goal, np.inf)
            pick = np.argmin(to_goal, axis=1)
            has = np.isfinite(to_goal[np.arange(len(pick)), pick])
            out[np.flatnonzero(m)[has], col] = idx[pick[has]]
    return out


def outfield_speed(speed, players, team, gk):
    """(n, 2) mean speed of each team's tracked players excluding its goalkeeper (m/s)."""
    present = np.isfinite(players[:, :, 0]) & np.isfinite(speed)
    out = np.full((len(speed), 2), np.nan)
    rows = np.arange(len(speed))
    for col, side in enumerate((HOME, AWAY)):
        use = present & (team == side)[None, :]
        has_gk = gk[:, col] >= 0
        use[rows[has_gk], gk[has_gk, col]] = False
        total = np.where(use, speed, 0.0).sum(1)
        count = use.sum(1)
        out[:, col] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def ball_out_of_pitch(ball, margin_m=0.0):
    """True where the tracked ball is outside the pitch lines (by more than margin_m)."""
    xy = to_metres(ball[:, :2])
    with np.errstate(invalid="ignore"):
        return ((xy[:, 0] < -margin_m) | (xy[:, 0] > LENGTH + margin_m)
                | (xy[:, 1] < -margin_m) | (xy[:, 1] > WIDTH + margin_m))


def nearest_by_team(players, team, ball):
    """(n, 2) distance in metres from the ball to the nearest tracked player of each team."""
    dist = np.linalg.norm(to_metres(players[:, :, :2]) - to_metres(ball[:, None, :2]), axis=-1)
    out = np.full((len(ball), 2), np.nan)
    with np.errstate(invalid="ignore"):
        for col, side in enumerate((HOME, AWAY)):
            d = np.where((team == side)[None, :] & np.isfinite(dist), dist, np.inf)
            best = d.min(1)
            out[:, col] = np.where(np.isfinite(best), best, np.nan)
    return out
