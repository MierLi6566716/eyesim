"""
A simple, randomised model of how an eye refocuses (accommodates) while it is
being measured. Used as the "patient" in the closed-loop refraction agent.

Conventions (dioptres, D):
  M0      the eye's relaxed refraction (what a cycloplegic exam would give).
  L       the lens in front of the eye (device lens / trial lens).
  demand  M0 - L: how much the eye must accommodate to see a distant
          fixation target sharply through the lens. demand > 0: it has to
          focus (an over-minus lens). demand < 0: the target is fogged by
          -demand dioptres (a plus lens), so there is nothing to focus on.
  A       accommodation actually in play (>= 0). It makes the eye more
          myopic: the camera sees M0 - A - L.

Behaviour (each number drawn per person, ranges from the literature cited in
docs/rl_closed_loop_refraction.md; they are deliberately wide):
  * the eye reacts to a change only after a latency (~0.25-0.45 s);
  * it then moves exponentially toward a goal, faster to focus than to relax;
  * goal for a sharp-able target: lag/lead around the resting (tonic) level,
    tonic + gain * (demand - tonic), clipped to [0, amplitude];
  * mild fog (up to a person-specific blur limit) relaxes accommodation fully:
    that is why clinical fogging works;
  * heavy fog opens the loop: there is no usable blur cue, and the eye drifts
    slowly to its resting focus (dark focus, ~0.25-1.75 D), so too much plus
    makes the reading MYOPIC again;
  * "instrument myopia": extra accommodation from knowing the device is close,
    which only fades while the eye is mildly fogged;
  * a dead zone (depth of focus) where small blur changes nothing;
  * microfluctuations: AR(1) noise of 0.05-0.25 D RMS.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AccomParams:
    latency_s: float
    tau_focus_s: float
    tau_relax_s: float
    tau_open_s: float
    gain: float
    tonic_d: float
    fog_limit_d: float
    prox_d: float
    tau_prox_s: float
    micro_rms_d: float
    amplitude_d: float
    dof_d: float


def sample_params(rng: np.random.Generator) -> AccomParams:
    age = rng.uniform(6, 50)
    return AccomParams(
        latency_s=rng.uniform(0.25, 0.45),
        tau_focus_s=rng.uniform(0.15, 0.5),
        tau_relax_s=rng.uniform(0.25, 0.9),
        tau_open_s=rng.uniform(1.5, 6.0),
        gain=rng.uniform(0.7, 0.95),
        tonic_d=rng.uniform(0.25, 1.75),
        fog_limit_d=rng.uniform(1.0, 3.0),
        prox_d=rng.uniform(0.0, 1.0) * (1.5 if age < 15 else 1.0),
        tau_prox_s=rng.uniform(1.5, 6.0),
        micro_rms_d=rng.uniform(0.05, 0.25),
        amplitude_d=max(1.0, 15.0 - 0.25 * age),      # Hofstetter minimum amplitude
        dof_d=rng.uniform(0.15, 0.4),
    )


class Accommodation:
    def __init__(self, p: AccomParams, dt: float, rng: np.random.Generator, start_demand: float):
        self.p, self.dt, self.rng = p, dt, rng
        n = max(1, int(round(p.latency_s / dt)))
        self.buf = deque([start_demand] * n, maxlen=n)
        self.A = self._goal(start_demand, 0.0)[0]       # settled on the starting view
        self.prox = p.prox_d
        self.micro = 0.0
        self.rho = np.exp(-dt / 0.4)                      # microfluctuation correlation time ~0.4 s

    def _goal(self, demand: float, A: float):
        p = self.p
        if demand >= -p.dof_d:                            # target can be focused
            g = p.tonic_d + p.gain * (max(demand, 0.0) - p.tonic_d)
            return float(np.clip(g, 0.0, p.amplitude_d)), "closed"
        if -demand <= p.fog_limit_d:                      # mild fog: relax fully
            return 0.0, "fog"
        return float(min(p.tonic_d, p.amplitude_d)), "open"   # heavy fog: drift to dark focus

    def step(self, demand: float) -> float:
        p, dt = self.p, self.dt
        self.buf.append(demand)
        delayed = self.buf[0]
        goal, mode = self._goal(delayed, self.A)
        if abs((delayed - self.A)) < p.dof_d and mode == "closed":
            goal = self.A                                  # inside depth of focus: no drive
        tau = p.tau_open_s if mode == "open" else (p.tau_focus_s if goal > self.A else p.tau_relax_s)
        self.A += (goal - self.A) * (1 - np.exp(-dt / tau))
        if mode == "fog":
            self.prox *= np.exp(-dt / p.tau_prox_s)
        sd = p.micro_rms_d * np.sqrt(1 - self.rho ** 2)
        self.micro = self.rho * self.micro + sd * self.rng.standard_normal()
        return max(0.0, self.A + self.prox + self.micro)
