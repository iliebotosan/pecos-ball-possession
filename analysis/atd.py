"""Fast array loader for the raw ATD, cached as .npy next to the data (never committed).

    from analysis.atd import load_arrays
    t = load_arrays()            # tactical by default
    t["ball"]      (n, 3)        x, y in 0-1 pitch units, speed; NaN where missing
    t["players"]   (n, 46, 3)    same per track slot
    t["team"]      (46,)         0 = home (Barcelona), 1 = away (Athletic)

Row i is video frame i (ATD frame_id - 1), i.e. submission row i.

Import from the project root (a notebook in the root, or `python -m analysis.<script>`).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from games import DATA_DIR, require, source_frames

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0 for home slots, 1 for away; the first Team in the metadata is home (FIFA/EPTS)."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def _parse(raw_path, n_frames):
    with open(raw_path) as handle:
        text = handle.read()
    flat = np.array(text.replace(":", ",").replace(";", ",").split(), dtype=object)
    # one line -> "id,x,y,s,...,bx,by,bs"; split() above only breaks on newlines
    rows = [np.array(line.split(","), dtype=np.float64) for line in flat]
    table = np.vstack(rows)
    frame_id = table[:, 0].astype(np.int64)
    if len(frame_id) != n_frames or frame_id[0] != 1 or frame_id[-1] != n_frames:
        raise ValueError("unexpected ATD frame ids {}..{} over {} rows".format(
            frame_id[0], frame_id[-1], len(frame_id)))
    players = table[:, 1:-3].reshape(n_frames, -1, 3)
    ball = table[:, -3:]
    return ball, players


def load_arrays(source="tactical", refresh=False):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], source=source)
    cache = os.path.join(DATA_DIR, "cache", "atd_{}.npz".format(source))
    if os.path.isfile(cache) and not refresh:
        loaded = np.load(cache)
        return {key: loaded[key] for key in loaded.files}

    ball, players = _parse(raw_path, source_frames(source))
    out = {"ball": ball, "players": players, "team": team_of_each_track(meta_path)}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out
