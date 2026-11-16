"""Referee whistles from the tactical audio, as a module (the detector of notebook_audio §2 and §5).

    from analysis import whistle as wh
    score, peak_hz = wh.frame_score()          # per video frame, cached in data/cache/
    W = wh.detect(score, peak_hz, graded)      # one row per whistle event

Audio time t s = video frame t * 25 (the soundtrack starts with the video). For each frame: the
spectrum of a 64 ms slice between 1.5 and 5.5 kHz; a whistle is a bin that is at the same time
SUDDEN (louder than the same bin over the preceding 0.5-3 s) and NARROW (louder than the bins
50-125 Hz either side). score = max over bins of min(novelty, peakiness) in dB, peak_hz = that bin.
An event is a run with score > thr inside the referee's band (3.5-4.6 kHz, calibrated in
notebook_audio §4 on kick-offs and ends of half), at least MIN_S long, gaps < MERGE_S bridged.
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from games import DATA_DIR, asset_paths
from analysis import features as fx

FPS = 25
SR = 16000
HOP = SR // FPS
NFFT = 1024
THR_DB = 25.0
F_LO, F_HI = 3500, 4600
MIN_S, MERGE_S = 0.12, 0.4
CACHE = os.path.join(DATA_DIR, "cache")


def load_audio():
    path = os.path.join(CACHE, "audio_tactical_16k.npy")
    if os.path.isfile(path):
        return np.load(path, mmap_mode="r")
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    raw = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", asset_paths()["video"], "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
    audio = np.frombuffer(raw, np.int16)
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, audio)
    return audio


def frame_score(n_frames=199319, refresh=False):
    """(score_db, peak_hz) per video frame. Cached as data/cache/whistle_score.npz (never committed)."""
    path = os.path.join(CACHE, "whistle_score.npz")
    if os.path.isfile(path) and not refresh:
        z = np.load(path)
        return z["score"], z["peak_hz"]
    audio = load_audio()
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    band = np.flatnonzero((freqs >= 1500) & (freqs <= 5500))
    f_band = freqs[band]

    win = np.hanning(NFFT).astype(np.float32)
    xf = np.asarray(audio, dtype=np.float32) / 32768
    pad = np.concatenate([np.zeros(NFFT // 2, np.float32), xf, np.zeros(NFFT, np.float32)])
    view = np.lib.stride_tricks.sliding_window_view(pad, NFFT)
    S = np.zeros((n_frames, len(band)), np.float16)          # as notebook_audio
    for s in range(0, n_frames, 20000):
        idx = np.arange(s, min(n_frames, s + 20000))
        idx = idx[idx * HOP < len(view)]
        P = np.abs(np.fft.rfft(view[idx * HOP] * win, axis=1)[:, band]) ** 2
        S[idx] = (10 * np.log10(P + 1e-12)).astype(np.float16)
    S = S.astype(np.float32)

    # novelty: vs the mean of the same bin 3 s .. 0.5 s before; peakiness: vs bins 3..8 either side
    c = np.vstack([np.zeros((1, S.shape[1]), np.float32), np.cumsum(S, 0)])
    n = len(S)
    a = np.clip(np.arange(n) - 3 * FPS, 0, n)
    b = np.clip(np.arange(n) - FPS // 2, 1, n)
    novelty = S - (c[b] - c[a]) / np.maximum(b - a, 1)[:, None]
    nb = np.zeros_like(S)
    for d in range(3, 9):
        nb[:, d:] += S[:, :-d]
        nb[:, :-d] += S[:, d:]
    peakiness = S - nb / 12
    both = np.minimum(novelty, peakiness)
    j = both.argmax(1)
    score = both[np.arange(n), j].astype(np.float32)
    peak_hz = f_band[j].astype(np.float32)
    np.savez(path, score=score, peak_hz=peak_hz)
    return score, peak_hz


def detect(score, peak_hz, graded, thr=THR_DB, f_lo=F_LO, f_hi=F_HI, min_s=MIN_S, merge_s=MERGE_S):
    """Whistle events: start, stop (exclusive), duration, strength, pitch."""
    m = (score > thr) & (peak_hz >= f_lo) & (peak_hz <= f_hi)
    out = []
    for a, b in zip(*fx.runs(m)):
        if out and a - out[-1][1] <= merge_s * FPS:
            out[-1][1] = b
        else:
            out.append([a, b])
    rows = [{"start": a, "stop": b, "dur_s": (b - a) / FPS, "max_db": float(score[a:b].max()),
             "hz": float(np.median(peak_hz[a:b])), "graded": bool(graded[a])}
            for a, b in out if b - a >= min_s * FPS]
    return pd.DataFrame(rows, columns=["start", "stop", "dur_s", "max_db", "hz", "graded"])


def dead_evidence(W, n, w=1.0, lead_s=3.0, slow=None, confirm_s=6.0):
    """Viterbi dead evidence y: +w per frame for lead_s after every graded whistle.
    With `slow` (a bool mask of slowed-down play), only whistles followed by a slow-down within
    confirm_s count, and the evidence runs from the whistle to that slow-down (at least lead_s)."""
    y = np.zeros(n)
    starts = np.array([a for a, _ in zip(*fx.runs(slow))]) if slow is not None else None
    for s0 in W[W.graded].start.values:
        end = s0 + int(lead_s * FPS)
        if starts is not None:
            nxt = starts[(starts >= s0) & (starts <= s0 + confirm_s * FPS)]
            if not len(nxt):
                continue
            end = max(end, int(nxt[0]))
        y[s0:end] += w
    return y


# -- the referee's whistle: anchors, signature, refined detector ------------------------------
#
# The detector above finds every sudden narrow tone in 3.5-4.6 kHz -- which includes fans whistling.
# The referee uses ONE pea whistle, so his blows share a pitch, hold it steadily, and (a pea whistle
# is a rich tone) carry a second harmonic at ~2 x f0, around 8 kHz -- above the 16 kHz audio's
# Nyquist, so the signature uses a 24 kHz copy. The signature is learned from moments where a referee
# whistle is certain (kick-offs, ends of half, the foul before a free kick), never from the truth.

SR_HI = 24000
HOP_HI = SR_HI // FPS            # 960 samples per video frame
NFFT_HI = 2048                   # 85 ms


def load_audio_hi():
    path = os.path.join(CACHE, "audio_tactical_24k.npy")
    if os.path.isfile(path):
        return np.load(path, mmap_mode="r")
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    raw = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", asset_paths()["video"], "-vn", "-ac", "1",
                          "-ar", str(SR_HI), "-f", "s16le", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
    audio = np.frombuffer(raw, np.int16)
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, audio)
    return audio


def _spectrum_hi(audio, frame):
    "dB magnitude spectrum (NFFT_HI) of the 85 ms centred on a video frame, 24 kHz audio"
    c = frame * HOP_HI
    seg = np.asarray(audio[max(0, c - NFFT_HI // 2):c + NFFT_HI // 2], dtype=np.float32) / 32768
    if len(seg) < NFFT_HI:
        seg = np.pad(seg, (0, NFFT_HI - len(seg)))
    P = np.abs(np.fft.rfft(seg * np.hanning(NFFT_HI))) ** 2
    return 10 * np.log10(P + 1e-12)


FREQ_HI = np.fft.rfftfreq(NFFT_HI, 1 / SR_HI)


def _prominence(spec, f, lo=60, hi=250):
    "dB of the strongest bin within +-40 Hz of f over the median of the bins lo..hi Hz either side"
    near = np.abs(FREQ_HI - f) <= 40
    side = (np.abs(FREQ_HI - f) >= lo) & (np.abs(FREQ_HI - f) <= hi)
    return float(spec[near].max() - np.median(spec[side]))


def event_features(W, score, peak_hz, audio_hi=None, onset_db=12.0, back_s=0.4):
    """Adds, per event: f0 (median pitch), pitch_std (Hz), f0_prom (dB), h2_prom (dB of the second
    harmonic at 2 x f0 over its neighbours), and onset = the first frame up to back_s before the event
    where the score already exceeded onset_db (the moment the blow starts, not when it got loud)."""
    audio_hi = load_audio_hi() if audio_hi is None else audio_hi
    W = W.copy()
    rows = []
    for r in W.itertuples():
        hz = peak_hz[r.start:r.stop]
        f0 = float(np.median(hz))
        frames = np.arange(r.start, r.stop)
        steady = frames[np.abs(peak_hz[frames] - f0) <= 150]
        frames = steady if len(steady) else frames          # a jumpy pitch can leave nothing near the median
        specs = [_spectrum_hi(audio_hi, f) for f in frames[:12]]
        f0p = np.median([_prominence(s, f0) for s in specs]) if specs else np.nan
        h2p = np.median([_prominence(s, 2 * f0) for s in specs]) if specs else np.nan
        k = r.start
        lo = max(0, r.start - int(back_s * FPS))
        while k > lo and score[k - 1] > onset_db:
            k -= 1
        rows.append({"f0": f0, "pitch_std": float(np.std(hz)), "f0_prom": f0p, "h2_prom": h2p, "onset": int(k)})
    return pd.concat([W.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def anchors(ctx, R, tags):
    """Moments with a certain referee whistle, with the window to look for it in (seconds, relative):
    kick-offs (from the restart frame), ends of half (last frame), the foul before a free kick (between
    the ball last being in normal play and the restart)."""
    from games import GAME, half_video_blocks
    out = []
    for x in R[R.type.str.lower().str.contains("kick off") & (R.restart >= 0)].itertuples():
        out.append({"anchor": f"kick-off ({x.type})", "frame": int(x.restart), "lo_s": -5.0, "hi_s": 1.0})
    for h, a, z in half_video_blocks(GAME):
        out.append({"anchor": f"end of half {h}", "frame": int(z), "lo_s": -6.0, "hi_s": 8.0})
    for x in R[R.type.str.lower().str.contains("free kick") & (R.restart >= 0)].itertuples():
        out.append({"anchor": "foul before a free kick", "frame": int(x.restart), "lo_s": -60.0, "hi_s": -1.0})
    return pd.DataFrame(out)


def match_anchors(A, Wf):
    "the strongest whistle event inside each anchor's window (or none)"
    rows = []
    for a in A.itertuples():
        lo, hi = a.frame + int(a.lo_s * FPS), a.frame + int(a.hi_s * FPS)
        c = Wf[(Wf.start >= lo) & (Wf.start <= hi)]
        if len(c):
            best = c.loc[c.max_db.idxmax()]
            rows.append({**a._asdict(), "event": int(best.name), "offset_s": (best.start - a.frame) / FPS,
                         "max_db": best.max_db, "f0": best.f0, "h2_prom": best.h2_prom})
        else:
            rows.append({**a._asdict(), "event": -1})
    return pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")


def learn_profile(Wf, idx, pitch_k=3.0, min_pitch_tol=120.0):
    """The referee's signature from anchor-matched events: pitch window (median +- max(min_pitch_tol,
    pitch_k x MAD)), the worst pitch stability, the weakest harmonic and duration range seen at anchors."""
    P = Wf.loc[idx]
    med = float(P.f0.median())
    mad = float((P.f0 - med).abs().median() * 1.4826)
    return {"f0_lo": med - max(min_pitch_tol, pitch_k * mad), "f0_hi": med + max(min_pitch_tol, pitch_k * mad),
            "pitch_std_max": float(P.pitch_std.max()) * 1.2, "h2_min": float(P.h2_prom.min()) - 3.0,
            "dur_min": float(P.dur_s.min()) * 0.7, "dur_max": float(P.dur_s.max()) * 1.5, "n_anchors": len(P)}


def is_referee(Wf, prof):
    return ((Wf.f0 >= prof["f0_lo"]) & (Wf.f0 <= prof["f0_hi"]) & (Wf.pitch_std <= prof["pitch_std_max"])
            & (Wf.h2_prom >= prof["h2_min"]) & (Wf.dur_s >= prof["dur_min"]) & (Wf.dur_s <= prof["dur_max"]))


def dead_from_whistles(ctx, starts, until="slow", confirm_s=6.0, hold_s=3.0):
    """D mask from whistle onsets. until='slow': to the slow-down that follows within confirm_s (whistles
    with none are dropped); 'hold': hold_s seconds regardless; 'slow_or_hold': the slow-down if there is one,
    else hold_s."""
    m = np.zeros(ctx.N, bool)
    slow_starts = np.array([a for a, _ in zip(*fx.runs(ctx.slow & ctx.graded))])
    for w in starts:
        if not ctx.graded[w]:
            continue
        nxt = slow_starts[(slow_starts >= w) & (slow_starts <= w + confirm_s * FPS)]
        if until == "slow":
            if len(nxt):
                m[w:nxt[0]] = True
        elif until == "hold":
            m[w:w + int(hold_s * FPS)] = True
        else:
            m[w:(nxt[0] if len(nxt) else w + int(hold_s * FPS))] = True
    return m
