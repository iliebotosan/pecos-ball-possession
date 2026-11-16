"""Possession as a hidden-state sequence, decoded with Viterbi.

Each graded frame t has a hidden state s_t in {H, A, D}. We pick the whole sequence of a half at
once by minimising

    J(s) = sum_t  e_t(s_t)  +  sum_{t>=1}  c(s_{t-1}, s_t)

  e_t(s)   emission cost: how badly frame t's evidence fits state s (a negative log-likelihood
           ratio, in nats; 0 = no opinion)
  c(i, j)  transition cost: 0 to stay, lam_switch for H<->A, lam_stop for H/A -> D, lam_start for
           D -> H/A (a negative log transition probability, up to a constant)

This is exactly MAP decoding of a hidden Markov model, and Viterbi solves it exactly in O(T * 9)
by dynamic programming. Nothing is fitted on the ground truth: the weights below are set by hand
from what the data notebooks showed, and the transition costs are the one real knob.

Evidence (all from analysis.versions.Context and analysis.passes.events):
  team, per frame:  x_t = w_touch * sign * closeness * margin, positive = home
                    sign: +1 if the nearest player is home; closeness: 1 at <= 1 m, 0 at >= touch_r;
                    margin: how much closer that team is than the other (0 .. 1 at >= margin_m);
                    zero where the ball is not tracked or is in flight (nobody "has" a flying ball)
                    + w_event at every kick / reception with a player within 2 m, for his team
                    e_t(H) = -x_t / 2,  e_t(A) = +x_t / 2
  dead, per frame:  y_t = w_slow * clip((slow_ms - outfield speed) / slow_scale, +-slow_clip)
                        + w_oob * [ball outside the pitch] + w_set * [5 s before a set-piece tag]
                    e_t(D) = -y_t

Frames with no evidence cost the same in every state, so the path keeps its current state: "carry
the last team" falls out of the model instead of being a rule. A short run of the other team's
touches only flips the state if its accumulated evidence beats the 2 * lam_switch it costs to go
there and back -- which is the "spurious blip" error spellF1 punishes hardest.

    from analysis import viterbi as vt
    labels = vt.decode_match(ctx, EK, vt.Params())        # letters, X outside the halves
"""

from dataclasses import dataclass, replace

import numpy as np

from games import GAME, half_video_blocks
from analysis import passes as ps

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


@dataclass(frozen=True)
class Params:
    # team evidence
    w_touch: float = 1.0       # nats per frame at full closeness and margin
    touch_r: float = 3.0       # m: closeness falls from 1 at 1 m to 0 at touch_r
    margin_m: float = 1.5      # m: full margin when one team is this much closer
    w_event: float = 8.0       # nats at each kick / reception with a player within 2 m
    # dead-ball evidence
    w_slow: float = 0.6        # nats per frame per unit of (slow_ms - speed) / slow_scale
    slow_ms: float = 1.3       # outfield speed (3 s rolling) where slow evidence is 0
    slow_scale: float = 0.4    # m/s per unit
    slow_clip: float = 1.5     # cap, in units, either way
    w_oob: float = 3.0         # nats per frame with the ball outside the pitch
    w_set: float = 1.0         # nats per frame in the 5 s before a set-piece tag
    # transitions
    lam_switch: float = 20.0   # H <-> A
    lam_stop: float = 8.0      # H/A -> D
    lam_start: float = 8.0     # D -> H/A
    late: bool = True          # on ties, switch as late as possible (at the new team's first touch)

    def with_(self, **kw):
        return replace(self, **kw)


def evidence(ctx, EK, p):
    """(x_team, y_dead) per frame -- see the module docstring."""
    near = ctx.nearest_by_team(ctx.bxy) if not hasattr(ctx, "_near") else ctx._near
    ctx._near = near
    dh, da = near[:, 0], near[:, 1]
    with np.errstate(invalid="ignore"):
        d_near = np.fmin(dh, da)
        sign = np.where(np.isnan(da) | (dh <= da), 1.0, -1.0)
        close = np.clip((p.touch_r - d_near) / (p.touch_r - 1.0), 0, 1)
        margin = np.clip(np.abs(dh - da) / p.margin_m, 0, 1)
        margin[np.isnan(margin)] = 1.0                      # only one team tracked near the ball
    x = p.w_touch * sign * close * margin
    x[~ctx.ok | ctx.in_flight | ~np.isfinite(x)] = 0.0

    ev = EK[(EK.player_m <= ps.TOUCH_M) & EK.kind.isin(["kick (ball speeds up)", "reception (ball slows down)"])]
    np.add.at(x, ev.frame.values, np.where(ev.team.values == "H", p.w_event, -p.w_event))

    with np.errstate(invalid="ignore"):
        slow = np.clip((p.slow_ms - ctx.speed_out_smooth) / p.slow_scale, -p.slow_clip, p.slow_clip)
    slow[~np.isfinite(slow)] = 0.0
    y = p.w_slow * slow + p.w_oob * ctx.oob + p.w_set * ctx.pre_set
    return x, y


def costs(x, y):
    "(T, 3) emission costs for H, A, D"
    return np.stack([-x / 2, x / 2, -y], axis=1)


def transitions(p):
    "(3, 3) cost of going from state i (row) to state j (column)"
    s, a, b = p.lam_switch, p.lam_stop, p.lam_start
    return np.array([[0, s, a],
                     [s, 0, a],
                     [b, b, 0]], dtype=float)


def viterbi(cost, trans):
    """Exact minimiser of sum e_t(s_t) + sum c(s_{t-1}, s_t). Ties keep the current state.
    Returns the state sequence (T,) and the minimal total cost."""
    T = len(cost)
    c = cost.tolist()
    tr = trans.tolist()
    back = bytearray(3 * T)
    acc = list(c[0])
    for t in range(1, T):
        ct = c[t]
        new = [0.0, 0.0, 0.0]
        for j in range(3):
            best, arg = acc[j], j                          # staying wins ties
            for i in range(3):
                if i != j:
                    v = acc[i] + tr[i][j]
                    if v < best:
                        best, arg = v, i
            new[j] = best + ct[j]
            back[3 * t + j] = arg
        acc = new
    path = [0] * T
    path[-1] = int(np.argmin(acc))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[3 * t + path[t]]          # plain ints: int8 arithmetic would overflow
    return np.array(path, dtype=np.int8), float(min(acc))


def decode(cost, trans, late=True):
    """Viterbi forward in time switches as EARLY as the evidence allows (right after the old team's
    last touch). Decoding the reversed sequence with the transposed costs gives the same optimum
    but switches as LATE as possible (at the new team's first touch), which is when the truth
    hands possession over: on a controlled reception."""
    if not late:
        return viterbi(cost, trans)[0]
    return viterbi(cost[::-1], trans.T)[0][::-1]


def decode_match(ctx, EK, p=Params()):
    """Letters for every frame: Viterbi inside each half, X outside."""
    x, y = evidence(ctx, EK, p)
    cost = costs(x, y)
    trans = transitions(p)
    out = np.full(ctx.N, "X", dtype="<U1")
    for _, a, z in half_video_blocks(GAME):
        out[a:z + 1] = LETTERS[decode(cost[a:z + 1], trans, p.late)]
    return out
