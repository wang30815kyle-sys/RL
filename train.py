"""
ML Arena — Chess SB3 Self-Play Training Script
Environment: PettingZoo chess_v6 (wrapped as single-agent Gym env)

Install dependencies:
    pip install -r requirements.txt

Train then upload:
    python run.py

Switching algorithms:
    Edit model.py — change ALGORITHM (must support action masking, e.g. MaskablePPO).
    Re-run train.py and run.py.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import chess as chess_lib
import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils
from stable_baselines3.common.vec_env import DummyVecEnv

from model import ALGORITHM, POLICY, POLICY_KWARGS, SAVE_PATH

# ═══ ✅ Tune freely: training hyperparameters ══════════════════════
TOTAL_TIMESTEPS = 500_000  # staged runs: 5_000, 50_000, 200_000, 500_000, 1_000_000+
N_ENVS          = 4         # parallel self-play environments
RANDOM_OPPONENT_PROB = 0.60
SNAPSHOT_POOL = [
    ("model_2949k_snapshot_pool_champion.zip", 0.40),
    ("model_3457k_80random_specialist.zip", 0.30),
    ("model_2441k_episode_fixed_champion.zip", 0.15),
    ("model_1933k_50_50_win33.zip", 0.10),
    ("model_1425k_white_primary_before_color_mix.zip", 0.05),
]
SNAPSHOT_LOAD_CUSTOM_OBJECTS = {"n_envs": 1, "n_steps": 1}
TERMINAL_WIN_REWARD = 5.0
TERMINAL_LOSS_REWARD = -5.0
DRAW_REWARD = -1.0
STALEMATE_REWARD = -1.5
INSUFFICIENT_MATERIAL_REWARD = -2.0
REPETITION_REWARD = -1.5
FIFTY_MOVE_REWARD = -1.5
MATERIAL_DELTA_COEF = 0.05
MATERIAL_DIFF_COEF = 0.0
CHECK_REWARD = 0.005
PROMOTION_REWARD = 0.50
# ══════════════════════════════════════════════════════════════════


class ChessSelfPlayEnv(gym.Env):
    """Single-agent Gym wrapper for chess_v6.

    The learning agent alternates between white (player_0) and black (player_1).
    Opponent moves are selected from random legal moves or a historical snapshot pool.
    Supports action_masks() for MaskablePPO.
    """

    def __init__(self):
        super().__init__()
        self._env = chess_v6.env()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(8 * 8 * 111,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4672)
        self._action_mask = np.ones(4672, dtype=np.int8)
        self._learning_agent: str = ""
        self._episode_index = 0
        self._snapshot_models = []
        self._snapshot_names = []
        self._snapshot_weights = []
        self._snapshot_probs = None
        self._episode_opponent_model = None
        self._episode_opponent_name = "random"
        self._prev_material_diff = 0

        for path, weight in SNAPSHOT_POOL:
            if os.path.exists(path):
                self._snapshot_models.append(
                    ALGORITHM.load(
                        path,
                        device="cpu",
                        custom_objects=SNAPSHOT_LOAD_CUSTOM_OBJECTS,
                    )
                )
                self._snapshot_names.append(path)
                self._snapshot_weights.append(float(weight))
            else:
                print(f"Warning: snapshot model not found: {path}")

        if self._snapshot_models:
            weights = np.array(self._snapshot_weights, dtype=np.float64)
            total_weight = weights.sum()
            if total_weight > 0.0:
                self._snapshot_probs = weights / total_weight
            else:
                print("Warning: snapshot weights sum to zero; using random legal move only")
                self._snapshot_models = []
                self._snapshot_names = []
                self._snapshot_weights = []

    # ═══ ✅ Reward shaping (change freely) ═══════════════════════════
    # PettingZoo chess rewards: win=+1, loss=-1, draw=0 (terminal only).
    # You can add intermediate rewards here, e.g. based on material count.
    def _material_diff(self) -> int:
        board = self._env.env.board
        piece_values = {
            chess_lib.PAWN: 1,
            chess_lib.KNIGHT: 3,
            chess_lib.BISHOP: 3,
            chess_lib.ROOK: 5,
            chess_lib.QUEEN: 9,
        }
        our_color = (
            chess_lib.WHITE
            if self._learning_agent == "player_0"
            else chess_lib.BLACK
        )
        return sum(
            piece_values.get(piece.piece_type, 0)
            if piece.color == our_color
            else -piece_values.get(piece.piece_type, 0)
            for piece in board.piece_map().values()
        )

    def _draw_reward(self) -> float:
        board = self._env.env.board

        if board.is_stalemate():
            return STALEMATE_REWARD

        if board.is_insufficient_material():
            return INSUFFICIENT_MATERIAL_REWARD

        if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            return REPETITION_REWARD

        if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
            return FIFTY_MOVE_REWARD

        return DRAW_REWARD

    def _shape_reward(
        self,
        reward: float,
        *,
        is_terminal: bool = False,
        gave_check: bool = False,
        promoted: bool = False,
    ) -> float:
        # ── 範例 1：子力差獎勵（取消注釋啟用）─────────────────────────
        # import chess as _chess
        # board = self._env.env.board
        # _vals = {_chess.PAWN:1, _chess.KNIGHT:3, _chess.BISHOP:3,
        #          _chess.ROOK:5, _chess.QUEEN:9}
        # our_color = _chess.WHITE if self._learning_agent == "player_0" else _chess.BLACK
        # mat = sum(_vals.get(p.piece_type, 0)
        #           for p in board.piece_map().values() if p.color == our_color) \
        #     - sum(_vals.get(p.piece_type, 0)
        #           for p in board.piece_map().values() if p.color != our_color)
        # reward += mat * 0.001   # 小額中間獎勵，避免蓋過終局 ±1
        # ──────────────────────────────────────────────────────────────
        # ── 範例 2：每步存活小獎勵 ────────────────────────────────────
        # reward += 0.001   # 鼓勵撐住，不要輕易被將死
        # ──────────────────────────────────────────────────────────────
        material_diff = self._material_diff()
        material_delta = material_diff - self._prev_material_diff
        self._prev_material_diff = material_diff

        if is_terminal:
            if reward > 0.0:
                shaped = TERMINAL_WIN_REWARD
            elif reward < 0.0:
                shaped = TERMINAL_LOSS_REWARD
            else:
                shaped = self._draw_reward()
        else:
            shaped = reward

        shaped += material_delta * MATERIAL_DELTA_COEF
        shaped += material_diff * MATERIAL_DIFF_COEF
        if gave_check:
            shaped += CHECK_REWARD
        if promoted:
            shaped += PROMOTION_REWARD
        return shaped
    # ══════════════════════════════════════════════════════════════════

    def action_masks(self) -> np.ndarray:
        return self._action_mask.astype(bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        self._learning_agent = "player_0" if self._episode_index % 2 == 0 else "player_1"
        self._episode_index += 1
        self._episode_opponent_model = None
        self._episode_opponent_name = "random"
        if self._snapshot_models and np.random.random() >= RANDOM_OPPONENT_PROB:
            snapshot_index = int(
                np.random.choice(len(self._snapshot_models), p=self._snapshot_probs)
            )
            self._episode_opponent_model = self._snapshot_models[snapshot_index]
            self._episode_opponent_name = self._snapshot_names[snapshot_index]
        self._advance_to_learning_agent()
        self._prev_material_diff = self._material_diff()
        obs, _, _, _, _ = self._env.last()
        self._action_mask = obs["action_mask"].copy()
        return obs["observation"].flatten().astype(np.float32), {}

    def _is_done(self) -> bool:
        return (
            self._env.terminations.get(self._learning_agent, False)
            or self._env.truncations.get(self._learning_agent, False)
        )

    def _agent_index(self, agent: str) -> int:
        return 0 if agent == "player_0" else 1

    def _learner_move_flags(self, action: int) -> tuple[bool, bool]:
        board = self._env.env.board
        try:
            move = chess_utils.action_to_move(
                board,
                int(action),
                self._agent_index(self._learning_agent),
            )
            return board.gives_check(move), move.promotion is not None
        except Exception:
            return False, False

    def _opponent_action(self, observation: dict, action_mask: np.ndarray) -> int:
        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            return 0

        if self._episode_opponent_model is not None:
            obs_flat = observation["observation"].flatten().astype(np.float32)
            try:
                action, _ = self._episode_opponent_model.predict(
                    obs_flat,
                    action_masks=action_mask.astype(bool),
                    deterministic=False,
                )
                action = int(action)
                if 0 <= action < len(action_mask) and action_mask[action]:
                    return action
            except Exception as exc:
                print(
                    "Warning: snapshot opponent action failed "
                    f"({self._episode_opponent_name}): {exc}"
                )

        return int(np.random.choice(legal))

    def _advance_to_learning_agent(self) -> None:
        for _ in range(500):
            if self._is_done() or self._env.agent_selection == self._learning_agent:
                return

            opp_obs, _, opp_term, opp_trunc, _ = self._env.last()
            if opp_term or opp_trunc:
                self._env.step(None)
                continue

            self._env.step(
                self._opponent_action(opp_obs, opp_obs["action_mask"])
            )

    def step(self, action: int):
        # Ensure the action is legal
        if not self._action_mask[int(action)]:
            legal = np.where(self._action_mask)[0]
            action = int(np.random.choice(legal)) if len(legal) else 0

        gave_check, promoted = self._learner_move_flags(int(action))
        self._env.step(int(action))

        # Let opponent(s) play until it is our turn or the game ends
        for _ in range(500):
            if self._is_done():
                reward = self._shape_reward(
                    float(self._env.rewards.get(self._learning_agent, 0.0)),
                    is_terminal=True,
                    gave_check=gave_check,
                    promoted=promoted,
                )
                self._action_mask = np.ones(4672, dtype=np.int8)
                return np.zeros(8 * 8 * 111, dtype=np.float32), reward, True, False, {}

            if self._env.agent_selection == self._learning_agent:
                break

            opp_obs, _, opp_term, opp_trunc, _ = self._env.last()
            if opp_term or opp_trunc:
                self._env.step(None)
                continue
            opp_mask = opp_obs["action_mask"]
            # ── Self-Play（啟用：在 __init__ 加下列兩行，取消下方注釋）──────────────────
            #      from model import ALGORITHM, SAVE_PATH
            #      self._opponent = ALGORITHM.load(SAVE_PATH)
            # 快照更新範例（在 reset() 或 __init__ 中加入步數計數器）：
            #      self._total_steps = getattr(self, "_total_steps", 0) + 1
            #      if self._total_steps % 100_000 == 0:
            #          self._opponent = ALGORITHM.load(SAVE_PATH)  # 更新對手至最新版本
            # ─────────────────────────────────────────────────────────────────────────────
            # opp_flat = opp_obs["observation"].flatten().astype(np.float32)
            # action, _ = self._opponent.predict(opp_flat,
            #                                    action_masks=opp_mask.astype(bool),
            #                                    deterministic=False)
            # self._env.step(int(action))
            # continue
            # ─────────────────────────────────────────────────────────────────────────────
            self._env.step(self._opponent_action(opp_obs, opp_mask))

        if self._is_done():
            reward = self._shape_reward(
                float(self._env.rewards.get(self._learning_agent, 0.0)),
                is_terminal=True,
                gave_check=gave_check,
                promoted=promoted,
            )
            self._action_mask = np.ones(4672, dtype=np.int8)
            return np.zeros(8 * 8 * 111, dtype=np.float32), reward, True, False, {}

        obs, _, term, trunc, info = self._env.last()
        self._action_mask = obs["action_mask"].copy()
        reward = self._shape_reward(
            float(self._env.rewards.get(self._learning_agent, 0.0)),
            gave_check=gave_check,
            promoted=promoted,
        )
        return obs["observation"].flatten().astype(np.float32), reward, term, trunc, info

    def close(self):
        self._env.close()


def main():
    env = DummyVecEnv([ChessSelfPlayEnv for _ in range(N_ENVS)])
    model_path = SAVE_PATH + ".zip"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = ALGORITHM.load(SAVE_PATH, env=env, device="cpu")
    else:
        print("Creating new model")
        model = ALGORITHM(
            POLICY,
            env,
            policy_kwargs=POLICY_KWARGS or None,
            verbose=1,
        )

    print(f"Parallel envs: {N_ENVS}")
    print("Model training color: 50% white / 50% black alternating")
    print("Opponent selection: fixed per episode")
    print("Opponent ratio: 60% random / 40% snapshot pool")
    print("Snapshot pool:")
    for snapshot_path, snapshot_weight in SNAPSHOT_POOL:
        overall_weight = (1.0 - RANDOM_OPPONENT_PROB) * snapshot_weight
        print(
            f"- {snapshot_path}: "
            f"{snapshot_weight:.0%} of snapshot episodes, "
            f"{overall_weight:.0%} overall"
        )
    print("Snapshot update during run: False")
    print("Reward perspective: learning-agent perspective")
    print("Reward: unchanged from previous run")
    print(
        "Draw rewards: "
        f"default={DRAW_REWARD}, "
        f"stalemate={STALEMATE_REWARD}, "
        f"insufficient_material={INSUFFICIENT_MATERIAL_REWARD}, "
        f"repetition={REPETITION_REWARD}, "
        f"fifty_move={FIFTY_MOVE_REWARD}"
    )
    print(
        "Reward shaping: "
        f"terminal_win={TERMINAL_WIN_REWARD}, "
        f"terminal_loss={TERMINAL_LOSS_REWARD}, "
        f"material_delta_coef={MATERIAL_DELTA_COEF}, "
        f"material_diff_coef={MATERIAL_DIFF_COEF}, "
        f"check={CHECK_REWARD}, "
        f"promotion={PROMOTION_REWARD}"
    )
    print(f"Policy kwargs: {POLICY_KWARGS or None}")
    print(f"Training {ALGORITHM.__name__} for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
    model.save(SAVE_PATH)
    print(f"\nModel saved as {SAVE_PATH}.zip — ready to upload with run.py")
    env.close()


if __name__ == "__main__":
    main()
