"""Synthesize the SFX pack (no third-party samples, no music).

Every file is peak-normalized to PEAK_DBFS so nothing in the mix can exceed
the -10 dB ceiling asked for in the brief.

    python3 make_sfx.py OUT_DIR
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
PEAK_DBFS = -14.0  # hard ceiling is -10 dBFS; leave headroom
rng = np.random.default_rng(7)


def envelope(n, attack, release, curve=3.0):
    t = np.linspace(0, 1, n)
    a = int(n * attack)
    r = int(n * release)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a) ** 2
    env[n - r:] = np.linspace(1, 0, r) ** curve
    return env * np.sin(np.pi * t) ** 0.2


def resonant_bandpass(x, centers, q=1.6):
    """Time-varying 2-pole bandpass (RBJ biquad, per-sample coefficients)."""
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, (xi, fc) in enumerate(zip(x, centers)):
        w0 = 2 * np.pi * fc / SR
        alpha = np.sin(w0) / (2 * q)
        b0, b2 = alpha, -alpha
        a0, a1, a2 = 1 + alpha, -2 * np.cos(w0), 1 - alpha
        yi = (b0 * xi + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def whoosh(dur=0.65, f_lo=250, f_hi=2800, reverse=False):
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    sweep = f_lo + (f_hi - f_lo) * np.sin(np.pi * t) ** 1.5
    mono = resonant_bandpass(rng.standard_normal(n), sweep)
    mono *= envelope(n, 0.55, 0.45)
    if reverse:
        mono = mono[::-1]
    pan = np.linspace(-0.6, 0.6, n)  # left -> right movement
    left = mono * np.sqrt((1 - pan) / 2)
    right = mono * np.sqrt((1 + pan) / 2)
    return np.stack([left, right], axis=1)


def pop(dur=0.14):
    n = int(SR * dur)
    t = np.arange(n) / SR
    freq = 320 + 700 * np.exp(-t * 45)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = np.sin(phase) * np.exp(-t * 28)
    click = rng.standard_normal(n) * np.exp(-t * 900) * 0.25
    mono = tone + click
    return np.stack([mono, mono], axis=1)


def tap(dur=0.06):
    n = int(SR * dur)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 160)
    tick = rng.standard_normal(n) * np.exp(-t * 1400) * 0.5
    mono = body + tick
    return np.stack([mono, mono], axis=1)


def chime(dur=1.4):
    """Soft bell for section titles (C6 + G6 + C7 partials)."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    mono = np.zeros(n)
    for f, amp, decay in [(1046.5, 1.0, 3.2), (1568.0, 0.55, 4.0),
                          (2093.0, 0.35, 5.5), (3136.0, 0.12, 8.0)]:
        mono += amp * np.sin(2 * np.pi * f * t) * np.exp(-t * decay)
    mono *= np.minimum(1, t / 0.004)
    l = mono
    r = np.concatenate([np.zeros(int(SR * 0.008)), mono])[:n]  # slight width
    return np.stack([l, r], axis=1)


def write(path, data):
    peak = np.max(np.abs(data))
    data = data / peak * 10 ** (PEAK_DBFS / 20)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "whoosh_in.wav", whoosh())
    write(out / "whoosh_out.wav", whoosh(dur=0.55, f_lo=200, f_hi=2200, reverse=True))
    write(out / "pop.wav", pop())
    write(out / "tap.wav", tap())
    write(out / "chime.wav", chime())
    print("wrote", sorted(p.name for p in out.glob("*.wav")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/sfx")
