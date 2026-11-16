# pecos: ball possession, frame by frame

Team **pecos**'s entry to **More than a Hack 2027, Phase 1: Ball Possession**: FC Barcelona vs Athletic. Every frame of the
tactical video is labelled `H` (Barcelona has the ball), `A` (Athletic), `D` (dead ball) or `X` (no answer), using
only the provided automatic data: tracking, Smart Tagging and the audio of the tactical video. No hand labels, no
ground truth used for fitting.

The technical document is [docs/pecos.md](docs/pecos.md).

## Reproduce the submission

**1. Requirements.** Python 3.14 (tested with 3.14.0) and **ffmpeg** on the `PATH` (tested with 8.1; the pipeline
reads the match audio through it).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Data.** The match data is not in this repository (it is licensed per participant). Unpack the bundle into
`data/` exactly as [data/README.md](data/README.md) shows, or point `MTAH_DATA_DIR` at it:

```bash
export MTAH_DATA_DIR=/path/to/mtah2027
```

**3. Run.**

```bash
python scripts/make_submission.py --name pecos
```

This writes `evaluation/results/pecos.csv` (199,319 rows), checks it with the official validator's rules, and prints
`valid`. The first run also builds caches in `data/cache/` (tracking arrays, the decoded audio and the whistle
score), so it takes a few minutes; later runs take about a minute.

**4. Check it is the same file.**

```bash
md5 -q evaluation/results/pecos.csv          # macOS;  Linux: md5sum evaluation/results/pecos.csv
# 19c22efab8064c4384e32268eefd680a
python evaluation/validate_submission.py evaluation/results/pecos.csv
```

`python scripts/make_submission.py --model v8_whistle` rebuilds the previous model (without the three finishing
touches) for comparison.

## How it works

```
tracking ──> 1. vote ──> 2. smooth & carry ──> 3. dead ball ──> 4. anti-flicker ──> 5. finishing touches ──> labels
                                                  ▲
                         audio ──> referee whistle┘        Smart Tagging ──> set pieces, long balls
```

1. **Vote** (`analysis/versions.py`, `analysis/v9.py`): each frame, the team of the player nearest the ball; no vote
   while the ball is in flight or contested.
2. **Smooth and carry**: a 2-second majority vote; when the ball is lost in live play, the last team keeps it.
3. **Dead ball**: outfield players slow down, the ball leaves the pitch, a set-piece tag is about to start
   (`analysis/features.py`, `analysis/versions.py`), and the **referee's whistle** detected in the audio
   (`analysis/whistle.py`): candidates in the 3.5–4.6 kHz band are matched to a signature learned from blows that
   are certain (kick-offs, ends of halves, fouls before free kicks), and a blow counts only if the players slow down
   within 6 s.
4. **Anti-flicker**: a possession shorter than 3 s with under 1 s of control and no completed pass is written `X`;
   the scorer then repeats the previous label, so the spurious switch disappears.
5. **Finishing touches** (`analysis/best.py`, `final()`): the dead ball ends exactly at a restart announced by a
   start whistle and confirmed by a set-piece restart; long passes inside a LONG BALL tag stay with the kicking team
   through the flight; when the ball disappears with one team and reappears with the other in live play, the switch
   goes to the middle of the blackout.

The entry point is `analysis/best.py`: `final(Bundle.load())` returns the labels. Module docstrings mention
research notebooks that are not part of this repository; everything needed to rebuild the submission is here.

## Rules we followed

- **No hand labels** on this match, for anything. Every label comes from code.
- The public 5-minute ground truth (`evaluation/gt/`) was used **only to check** the pipeline, never to fit or tune
  a parameter. Changes were also judged on full-match checks that need no ground truth.
- **No match data** in this repository; caches and outputs are written to `data/` and `evaluation/results/`, which
  are git-ignored.

## Layout

```
analysis/        our pipeline (entry point: best.py)
scripts/         make_submission.py: data in, validated submission out
docs/            the technical document
games.py, starter/, evaluation/, data/README.md
                 from the organisers' starter kit (metrica-sports/more-than-a-hack-2027), MIT,
                 see LICENSE-starter-kit; evaluation/ holds the official scorer and validator
```
