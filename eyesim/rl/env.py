"""
Closed-loop refraction environment: an agent controls the lens in front of an
eye that refocuses, and must report the eye's RELAXED refraction.

One step = one camera frame (30 Hz). Each step the agent picks a lens change
(or reports). The camera sees eye + lens: S = M0 - A(t) - L. The sensor model
is built from REAL data: the network's readings of the model-eye photos at
37 cm (blocks A, B, C), so the blind band, the bias of each reading and the
per-frame noise are the measured ones, not assumptions:

  * S inside the measured blind band (no crescent)  -> no reading
  * otherwise reading = S + bias(S) + N(0, 0.10 D)   (bias interpolated from
    the real readings; per-frame noise = measured per-shot spread)

The report is fixed, not learned: the median of (reading + L) over the last
REPORT_FRAMES readable frames. So the agent cannot "guess" the answer; it can
only choose WHEN and HOW to look: which lens powers, for how long, when to stop.
That isolates the control problem (fogging, timing, the blind band).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .accommodation import Accommodation, sample_params

ROOT = Path(__file__).resolve().parents[2]
DT = 1 / 30
MAX_STEPS = 300                         # 10 s
REPORT_FRAMES = 6
STEPS = np.array([-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0])   # lens changes per frame (D)
REPORT = len(STEPS)                     # action index for "report"
HIST = 15                               # frames of history in the observation (0.5 s)
FRAME_NOISE = 0.10
TIME_COST = 0.0005                      # reward per frame (0.15 D-equivalent for the full 10 s)
LENS_RANGE = (-12.0, 12.0)


def _sensor_table():
    """(S, bias) for real sessions with a crescent, and the blind band, from the 37 cm blocks."""
    net = "checkpoints/wide_m_only/best.pt"
    S, B, dark = [], [], []
    for b in ("A_37cm_plus4_7mm", "B_37cm_plus8_7mm", "C_37cm_nolens_7mm"):
        for r in csv.DictReader(open(ROOT / "model_eye/results" / b / "results.csv")):
            if r["network"] != net:
                continue
            L = float(r["lens_d"])
            s = float(r["dial_d"]) - L / (1 - 0.005 * L)
            if r["crescent_seen"] == "True":
                S.append(s); B.append(float(r["net_system_ir"]) - s)
            else:
                dark.append(s)
    o = np.argsort(S)
    S, B = np.array(S)[o], np.array(B)[o]
    myo = S[S < -2.7]; hyp = S[S > -2.7]
    band = (float(myo.max()), float(hyp.min()))       # last lit setting on each side
    return S, B, band


class ClosedLoopRefraction:
    def __init__(self, seed: int | None = None, m_range=(-9.0, 4.0), lens_rate=None):
        self.rng = np.random.default_rng(seed)
        self.S_tab, self.B_tab, self.band = _sensor_table()
        self.m_range = m_range
        self.lens_rate = lens_rate          # None = fast lens (any step per frame); else D/s limit

    # ------------------------------------------------------------------ sensor
    def _read(self, S: float):
        if self.band[0] < S < self.band[1]:
            return None
        s_clip = float(np.clip(S, self.S_tab[0], self.S_tab[-1]))     # network range ends: saturate
        bias = float(np.interp(s_clip, self.S_tab, self.B_tab))
        return s_clip + bias + FRAME_NOISE * self.rng.standard_normal()

    # ------------------------------------------------------------------ episode
    def reset(self, M0: float | None = None, params=None):
        self.M0 = float(self.rng.uniform(*self.m_range)) if M0 is None else float(M0)
        self.p = params or sample_params(self.rng)
        self.L = 0.0
        self.eye = Accommodation(self.p, DT, self.rng, start_demand=self.M0 - self.L)
        self.t = 0
        self.hist = np.zeros((HIST, 4), np.float32)     # est, readable, lens, time
        self.readings = []                               # (t, est) of readable frames
        self.lens_moved = 0
        return self._observe()

    def _observe(self):
        A = self.eye.step(self.M0 - self.L)
        self.A_true = A
        S = self.M0 - A - self.L
        r = self._read(S)
        est = (r + self.L) if r is not None else 0.0
        if r is not None:
            self.readings.append((self.t, est))
        row = np.array([est / 10.0, float(r is not None), self.L / 10.0, self.t / MAX_STEPS], np.float32)
        self.hist = np.vstack([self.hist[1:], row])
        return self._obs()

    def _obs(self):
        """Last 0.5 s of frames, plus a memory summary: last estimate, time since it,
        number of readings so far, and time since the lens last moved."""
        last_t, last_e = self.readings[-1] if self.readings else (-MAX_STEPS, 0.0)
        summary = np.array([last_e / 10.0, (self.t - last_t) / MAX_STEPS, min(len(self.readings), 60) / 60.0,
                            (self.t - self.lens_moved) / MAX_STEPS], np.float32)
        return np.concatenate([self.hist.flatten(), summary])

    def report(self):
        if len(self.readings) < REPORT_FRAMES:
            return None
        return float(np.median([e for _, e in self.readings[-REPORT_FRAMES:]]))

    def step(self, a: int):
        if a == REPORT or self.t >= MAX_STEPS - 1:
            est = self.report()
            err = 5.0 if est is None else abs(est - self.M0)
            reward = -min(err, 5.0)
            return self._obs(), reward, True, {"err": err, "est": est, "t": self.t}
        dL = STEPS[a]
        if self.lens_rate is not None:
            dL = float(np.clip(dL, -self.lens_rate * DT, self.lens_rate * DT))
        newL = float(np.clip(self.L + dL, *LENS_RANGE))
        self.t += 1
        if newL != self.L:
            self.lens_moved = self.t
        self.L = newL
        return self._observe(), -TIME_COST, False, {}

    @property
    def obs_dim(self):
        return HIST * 4 + 4

    @property
    def n_actions(self):
        return len(STEPS) + 1
