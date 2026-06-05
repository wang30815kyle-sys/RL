"""
ML Arena — Chess Agent (Stable Baselines 3 / sb3-contrib MaskablePPO)
Environment: PettingZoo chess_v6 (2-player)

Observation (observation):
    numpy.ndarray, shape=(8, 8, 111), dtype=int8 — board state

Action mask (action_mask):
    numpy.ndarray, shape=(4672,), dtype=int8
    Only indices where value==1 are legal moves.

Score: win count across multiple games, higher is better.
"""

import os

import numpy as np

from model import ALGORITHM  # ✅ Algorithm defined in model.py — do not edit here

# ┌─────────────────────────────────────────────────────────────────┐
# │  What to change                                                  │
# │  ✅ Change: model.py (algorithm, policy, network architecture)   │
# │  ❌ Fixed:  class Agent, act() signature, return type int        │
# └─────────────────────────────────────────────────────────────────┘

class Agent:
    def __init__(self):
        weights_path = os.path.join(os.path.dirname(__file__), "model.zip")
        self.model = ALGORITHM.load(weights_path, device="cpu")

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        ❌ Fixed: method name, signature, return type
        ✅ Change: deterministic=False for stochastic play

        Args:
            observation: shape (8, 8, 111) int8 — board state
            action_mask: shape (4672,) int8 — 1 = legal move
        Returns:
            int: action index in [0, 4671]
        """
        obs_flat = observation.flatten().astype(np.float32)
        action, _ = self.model.predict(
            obs_flat,
            action_masks=action_mask.astype(bool),
            deterministic=True,
        )
        return int(action)
