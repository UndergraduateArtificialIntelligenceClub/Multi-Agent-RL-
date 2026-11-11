"""Pursuit environment factory. Every Phase 1 script and test builds the
environment through here, so the environment version lives in one place."""
from pettingzoo import make

PURSUIT_ID = "sisl/pursuit-v5"


def make_pursuit(render_mode: str | None = None, **kwargs):
    """PettingZoo SISL Pursuit, parallel API.

    The learning agents are the pursuers ("pursuer_0", "pursuer_1", ...); they are
    rewarded for catching the scripted evaders. Extra keyword arguments go to the
    environment (e.g. n_pursuers, n_evaders, max_cycles).
    """
    return make("parallel", PURSUIT_ID, render_mode=render_mode, **kwargs)


def evaders_captured(env) -> int:
    """Number of evaders caught so far in the current episode."""
    return int(env.unwrapped.env.evaders_gone.sum())
