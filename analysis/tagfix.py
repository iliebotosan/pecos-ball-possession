"""Fill in the 36 Smart Tagging tags whose Team is N/A.

    from analysis import tagfix as tf
    tags, table = tf.fix_tags(ctx, kin, R, y)     # tags: same list with Team filled; table: one row per tag
    tf.validate_phases(ctx, kin, y)               # the phase method tried on phase tags that DO have a team

Set pieces (26 throw-ins, 6 kick-offs): the decided team of analysis.restarts (rule / ball went out /
thrower on the line / first touch).
Phases (2 CREATING CHANCES, 2 DEFENDING IN DEFENSIVE THIRD): two clues for the ATTACKING team --
  touch: the team with clear control of the ball (pseudo-labels) on most of the phase's frames;
  side:  the ball's median position: the team attacking that half of the pitch.
Agreement -> that team; otherwise touch if it is clear (>= 25 labelled frames, >= 70% one team), else
side. An attacking code takes the attacking team, a defending code the other one.
The fixed table is written to data/derived/ (gitignored, like all match data).
"""

import os

import numpy as np
import pandas as pd

from games import DATA_DIR
from analysis import features as fx
from analysis.atd import HOME, AWAY
from analysis import cascade as cs

LENGTH = fx.LENGTH


def team_names(tags):
    names = [(e["tags"].get("Team") or "") for e in tags]
    home = next(n for n in names if n.lower().startswith("barcelona"))
    away = next(n for n in names if n.lower().startswith("at"))
    return home, away


def phase_attacker(ctx, kin, y, a, b):
    "(team 'H'/'A'/'', how) of the attacking team during [a, b)"
    lab = y[a:b]
    lab = lab[lab >= 0]
    touch = ""
    if len(lab):
        share = np.mean(lab == 1)
        touch = "H" if share >= 0.5 else "A"
        clear = len(lab) >= 25 and max(share, 1 - share) >= 0.7
    else:
        clear = False
    side = ""
    xs = kin["bxy"][a:b, 0]
    xs = xs[np.isfinite(xs)]
    h = int(np.bincount(ctx.half[a:b]).argmax()) if b > a else 0
    if len(xs) and h in (1, 2):
        sides = fx.defending_side(ctx.players, ctx.team) if not hasattr(ctx, "_sides") else ctx._sides
        ctx._sides = sides
        in_right = np.median(xs) > LENGTH / 2
        # the attacker in the right half defends the left goal
        side = "H" if (sides[h][HOME] == "left") == in_right else "A"
    if touch and side and touch == side:
        return touch, "touch + side agree"
    if touch and clear:
        return touch, "touch" + (" (side disagrees)" if side else "")
    if side:
        return side, "side" + (" (touch unclear)" if touch else "")
    return touch, "touch (weak)" if touch else "none"


def fix_tags(ctx, kin, R, y):
    home, away = team_names(ctx.tags)
    name = {"H": home, "A": away}
    decided = dict(zip(R.tag_start, zip(R.team, R.by)))
    out, rows = [], []
    for e in ctx.tags:
        e2 = {**e, "tags": dict(e["tags"])}
        team0 = e["tags"].get("Team") or "N/A"
        new, how = team0, "original"
        if team0 == "N/A":
            code = e["code"].upper()
            if code == "SET PIECES":
                t, by = decided.get(e["start_frame"], ("", "none"))
                new, how = (name[t], "restart: " + by) if t else ("N/A", "undecided")
            elif code in cs.ATTACKING | cs.DEFENDING:
                att, by = phase_attacker(ctx, kin, y, e["start_frame"], e["end_frame"])
                if att:
                    t = att if code in cs.ATTACKING else {"H": "A", "A": "H"}[att]
                    new, how = name[t], "phase: " + by
                else:
                    new, how = "N/A", "undecided"
        e2["tags"]["Team"] = new
        e2["tags"]["Team source"] = how
        out.append(e2)
        rows.append({"code": e["code"], "type": e["tags"].get("Type"), "start_frame": e["start_frame"],
                     "end_frame": e["end_frame"], "team_original": team0, "team_fixed": new, "source": how})
    return out, pd.DataFrame(rows)


def validate_phases(ctx, kin, y):
    "the phase method on phase tags that have a team: how often it names the tag's own team"
    home, away = team_names(ctx.tags)
    rows = []
    for e in ctx.tags:
        code = e["code"].upper()
        t = (e["tags"].get("Team") or "")
        if code not in cs.ATTACKING | cs.DEFENDING or t not in (home, away):
            continue
        att, by = phase_attacker(ctx, kin, y, e["start_frame"], e["end_frame"])
        if not att:
            continue
        pred = att if code in cs.ATTACKING else {"H": "A", "A": "H"}[att]
        rows.append({"code": code, "how": by.split(" (")[0], "right": pred == ("H" if t == home else "A")})
    return pd.DataFrame(rows)


def export(table, name="BAR-ATH_tactical_smart_tagging_fixed.csv"):
    path = os.path.join(DATA_DIR, "derived", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return path
