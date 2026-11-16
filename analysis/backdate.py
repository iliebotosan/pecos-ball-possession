"""Move each team switch to the receiving team's first touch.

The truth hands possession over on a controlled reception, but a model switches wherever its evidence
tips: a smoothing window centres it, Viterbi puts it at the first frame the new team's evidence
outweighs the switch cost -- often a little after the reception. This post-process moves every direct
switch (H -> A or A -> H, no dead ball between) to the receiving team's first reception / touch event
inside [switch - back_s, switch + fwd_s]; if there is none it stays where it is.

    from analysis import backdate as bd
    out = bd.move_switches(letters, EK, runs, back_s=1.0, fwd_s=0.5)

Only the boundary moves; the number of possessions never changes, so spurious possessions are
neither created nor removed. X is treated as "previous label" (as the scorer does), and preserved.
"""

import numpy as np

from analysis import passes as ps

FPS = 25
CONTROL_KINDS = ("reception (ball slows down)", "touch / deflection", "kick (ball speeds up)")


def event_frames(EK, kinds=CONTROL_KINDS):
    "{'H': sorted frames, 'A': sorted frames} of the team's events with a player within 2 m"
    e = EK[(EK.player_m <= ps.TOUCH_M) & EK.kind.isin(kinds)]
    return {t: np.sort(e[e.team == t].frame.values) for t in ("H", "A")}


def move_switches(letters, EK, runs, back_s=1.0, fwd_s=0.5, kinds=CONTROL_KINDS):
    ev = event_frames(EK, kinds) if not isinstance(EK, dict) else EK
    out = np.asarray(letters).copy()
    back, fwd = int(back_s * FPS), int(fwd_s * FPS)
    for a, b in runs:
        seg = out[a:b]
        filled = seg.copy()                                   # X -> previous label, like the scorer
        for i in range(1, len(filled)):
            if filled[i] == "X":
                filled[i] = filled[i - 1]
        # direct switches in the filled labels
        sw = [i for i in range(1, len(filled)) if filled[i] != filled[i - 1]
              and filled[i] in ("H", "A") and filled[i - 1] in ("H", "A")]
        new = filled.copy()
        for k, i in enumerate(sw):
            to, frm = filled[i], filled[i - 1]
            # stay inside the neighbouring runs so possessions are not merged or split
            lo = i - 1                                        # earlier moves already applied: use `new`
            while lo > 0 and new[lo - 1] == frm:
                lo -= 1
            hi = i
            while hi + 1 < len(filled) and filled[hi + 1] == to:
                hi += 1
            e = ev[to]
            cand = e[(e >= a + max(lo + 1, i - back)) & (e <= a + min(hi, i + fwd))]
            if not len(cand):
                continue
            j = int(cand[0]) - a
            if j < i:
                new[j:i] = to
            elif j > i:
                new[i:j] = frm
        # put X back where it was (only where the moved label equals the original fill)
        # ... and only where it repeats the label before it (an X at the start of a run would be
        # filled with the previous team and erase the run)
        prev = np.concatenate([new[:1], new[:-1]])
        keep_x = (seg == "X") & (new == filled) & (new == prev)
        new[keep_x] = "X"
        out[a:b] = new
    return out
