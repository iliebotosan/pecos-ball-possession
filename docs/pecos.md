# pecos: technical document

**More than a Hack 2027, Phase 1: Ball Possession** · FC Barcelona vs Athletic · team **pecos**

We label each of the 199,319 frames of the tactical video `H` (Barcelona has the ball), `A` (Athletic), `D` (dead
ball) or `X` (no answer) with an algorithmic pipeline built from the provided automatic data. No hand labels are
used anywhere, and the public ground truth was used only to check the pipeline.

| | |
|---|---|
| public 5-minute window | spellF1 **0.579** (5–95% bootstrap range 0.40–0.74), macroF1 **0.852** |
| organisers' baselines, same window | spellF1 0.038–0.424 (best: nearest player, majority vote over ±5 s) |
| full match (no ground truth) | 307 possessions (2.9 per minute), 227 changes of team, 136 dead-ball stretches |
| reproduction | `python scripts/make_submission.py --name pecos` → identical file (md5 `19c22efab8064c4384e32268eefd680a`) |

---

## 1. The approach, and why

Two facts shaped every decision.

**The metric scores whole possessions.** spellF1 matches predicted possessions one-to-one with true ones by
temporal overlap (IoU 0.3–0.7). A per-frame classifier that is right 85% of the time can still score poorly: one
spurious half-second switch splits a true possession into pieces that no longer overlap enough to match, and adds a
false one. Being a second late costs little. So the pipeline is built around **stability**: decide when the
nearest-player vote can be trusted, hold the last team when it cannot, and suppress short switches.

**The ball is missing on 41% of graded frames.** About two thirds of those frames are stoppages, where the task is to
recognise that play is dead rather than to find the ball. The rest is live play the camera loses. Detecting
stoppages precisely is therefore worth more than any improvement to the ball itself.

We chose explicit rules over a learned model because the rules forbid hand labels on this match and the public
5 minutes may not be used for training or tuning. A learned model would have had nothing legitimate to learn the
competition's possession definition from (contested balls stay with the last team; a switch happens at the first
*controlled* touch). The rules encode that definition directly; where the pipeline does learn (the referee's whistle),
it learns from events it finds itself.

## 2. The pipeline

```
tracking ──> 1. vote ──> 2. smooth & carry ──> 3. dead ball ──> 4. anti-flicker ──> 5. finishing touches ──> labels
                                                  ▲
                         audio ──> referee whistle┘        Smart Tagging ──> set pieces, long balls
```

Entry point: `analysis/best.py`, `final(Bundle.load())`.

1. **Vote.** Each frame, the team of the player nearest the ball. No vote while the ball is **in flight**:
   free (no player within 2 m) for at least 0.3 s and reaching 8 m/s. Without this rule, a long ball is handed to
   whoever it passes over.
2. **Smooth and carry.** A majority vote over ±2 s (±50 frames, the organisers' own `smooth` from `starter/baseline.py`); frames without a vote take the last team
   (inside the half). When the ball is lost in live play, the last team keeps it.
3. **Dead ball** (`D`), any of:
   - outfield players' mean speed (smoothed over 3 s) below 1.3 m/s for at least 4 s;
   - the ball outside the pitch, padded by 1 s;
   - the 5 s before a SET PIECES tag;
   - a **referee whistle** (below), from its first frame to the slow-down that follows within 6 s.
4. **Anti-flicker.** A possession shorter than 3 s, with less than 1 s of continuous control by that team and no
   completed pass, is written `X`. The scorer fills `X` with our previous label, so the spurious switch disappears
   and the possession it interrupted stays whole.
5. **Finishing touches.** None adds or removes a possession; each moves a boundary (23 in the whole match):
   - **restart end:** at a restart announced by a *start* whistle and confirmed by a set-piece restart within 3 s,
     the dead ball runs up to the restart kick and not past it (the speed signal lags the kick by 1–3 s);
   - **tag-confirmed long passes:** a kick with a flight of at least 1 s and 15 m, inside a LONG BALL tag of the
     kicking team: the kicking team keeps the ball until the reception;
   - **blackout split:** the ball disappears with one team and reappears with the other while play stays live: the
     switch goes to the middle of the blackout instead of near its end (12 cases).

**The referee whistle** (`analysis/whistle.py`, `analysis/best.py`). The audio track of the tactical video is
decoded with ffmpeg and analysed per video frame. Whistle candidates are tonal peaks in the 3.5–4.6 kHz band. Their
signature (pitch, pitch stability, prominence of the second harmonic, duration) is **learned from anchor events the
pipeline locates by itself**: kick-offs, ends of halves and fouls before tagged free kicks, where a referee whistle
is certain. Candidates matching that profile are kept. A blow starts a dead ball only if the players slow down within
6 s: whistles that restart play, or that sound during a stoppage, change nothing. This step alone moved public
spellF1 from 0.533 to 0.579.

## 3. Assets: used, and tried and dropped

| asset | used for |
|---|---|
| tactical tracking (ATD) | everything: ball and player positions, teams, speeds, kicks, flights |
| Smart Tagging | SET PIECES (dead ball before the tag; restarts), LONG BALL (confirming long passes), anchors for the whistle profile, team names |
| tactical video | **audio only**: the referee whistle |
| homography | not in the final pipeline (used to align the broadcast and in the computer-vision experiment) |
| broadcast video and tracking | tested and dropped (below) |

How each step changed public spellF1 (a check on 17 possessions, not a tuning target):

| step | spellF1 |
|---|---|
| nearest team, majority over ±2 s, ball unseen → D (the organisers' `smooth50`) | 0.326 |
| + ball unseen → carry the last team | 0.361 |
| + dead ball from slow players / ball out / set-piece tags | 0.380–0.390 |
| + no votes while the ball is in flight | 0.432 |
| + short suspicious possessions merged into a neighbour | 0.500 |
| + written `X` instead | 0.533 |
| + referee whistle | **0.579** |
| + finishing touches (submitted) | 0.579 (they act only outside the window) |

**Tried and dropped**, with the reason:

| idea | outcome |
|---|---|
| state models (HMM / Viterbi, semi-Markov, posterior decoding) with hand-set costs | never beat the rules |
| passes as an add-on (kicker keeps the flight, receiver gets the ball) | 0.533 → 0.489: our pass detector finds ~45% of passes and its errors add switches |
| "air ball" rule on every kick | 0.511: fired on dribble touches and deflections |
| long passes without tag confirmation (≥ 1.5 s & 20 m; ≥ 2 s & 30 m) | 0.555 and 0.542, more short possessions |
| broadcast tracking: aligned to the exact frame from player positions (one fixed offset per half; replays and non-match shots recognised and excluded) | the same model on both cameras agrees on 86.5% of shared frames; replacing tactical with broadcast where live: 0.575 alone, but more flicker in combination; where the broadcast sees the ball: 0.493; targeted fusion: 0.518 |
| restart team (the restarting team owns the first 3 s) | short possessions over the match 3 → 14 |
| team shape / movement to place turnovers during ball blackouts | weak signal (AUC 0.63 on the unseen half) |
| our own ball detector on the raw video (YOLO11 pre-trained on COCO, guided search between Metrica's ball positions) | recovers 7–14% of missing frames on test gaps, not reliable enough to use |

**How a change was accepted.** Because the public window is for checking only, a change had to pass three checks
fixed before running it: (1) possessions shorter than 2 s over the **full match** do not grow by more than 5%;
(2) possessions per minute over the full match stay within ±10% of the current model; (3) public spellF1 drops by no
more than 0.01. Ideas also needed a reason to help. Several that looked good alone failed in combination and were
left out.

## 4. Training data

**None.** No labelled dataset of any kind was used, from this match or any other. No parameter was fitted on the
public ground truth. The only learned component in the submission is the whistle profile, estimated from anchor
events that the pipeline itself locates in this match's tracking and tags. The COCO-pre-trained detector appears only
in a dropped experiment.

## 5. Possession rules we implemented

- **Who has it:** the team of the player nearest the ball, by majority vote over ±2 s. A switch needs the new team to win the
  vote, so a touch or deflection that does not keep the ball does not switch.
- **Loose and contested balls:** stay with the last team. In flight, nobody votes; a short possession without control
  or a pass is written `X` (the previous team continues).
- **Ball not tracked, play live:** the last team keeps it until the ball reappears; if it reappears with the other
  team, the switch is placed mid-way through the blackout.
- **Dead ball:** from the moment play stops (whistle with a slow-down, ball off the pitch, players stop, set piece
  imminent) to the restart. Where a confirmed start whistle announces the restart, the dead ball ends exactly at the
  kick. After a restart, possession is decided by the vote again.
- **Long passes:** with a LONG BALL tag confirming them, the kicking team keeps the ball through the flight.
- **`X`:** 2.2% of rows, only for short suspicious possessions; for spellF1 an `X` means "keep the previous label".

## 6. Limitations, and what we would do with more time

- **Live play without the ball** (13% of graded frames): we can only keep the last team. A sequence model that infers
  the ball and the team in possession from player movement, trained self-supervised by hiding the ball where it is
  visible, is the natural next step.
- **Short real possessions are merged:** in the public window we find 12 possessions against 17 true ones, mostly
  missing short ones. That is the price of the anti-flicker rule.
- **Fixed thresholds:** the dead-ball thresholds (1.3 m/s for 4 s, 6 s whistle confirmation) are fixed by reasoning,
  not learned. A live / dead state model with parameters estimated from the data itself would place stoppages more
  precisely. Some long carried stretches and goal celebrations may be stoppages we call live.
- **The whistle profile** is learned from only 7 anchor blows. More anchors (other matches) would make it sturdier.
- **Learned models need labelled matches:** human labels on *other* matches are allowed. Possession labels built from
  public tracking-and-event datasets would let a model learn the competition's definition instead of our encoding of it.
- **Our own ball detector** could be fine-tuned on this camera using Metrica's own ball positions as automatic labels.

## 7. Reproduction

Python 3.14 (tested 3.14.0), `pip install -r requirements.txt` (numpy, pandas, scipy, pinned), ffmpeg on the `PATH`
(tested 8.1), and the data bundle in `data/` or `MTAH_DATA_DIR`. Then:

```bash
python scripts/make_submission.py --name pecos
```

It writes `evaluation/results/pecos.csv`, checks it with the official validator's rules and prints `valid`. The first
run builds caches in `data/cache/` (tracking arrays, decoded audio, whistle score); expected md5 of the output:
`19c22efab8064c4384e32268eefd680a`.
