#!/usr/bin/env python3
"""Evaluate a trained chess model against a random legal-move opponent."""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import chess_v6

from model import ALGORITHM, SAVE_PATH


DRAW_REASONS = (
    "Stalemate",
    "Insufficient material",
    "Threefold repetition",
    "Fifty-move rule",
    "Max-moves / truncation",
    "Other draw",
)


@dataclass
class GameResult:
    score: float
    moves: int
    model_agent: str
    draw_reason: str
    terminated: bool
    truncated: bool


@dataclass
class EvalResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_moves: int = 0
    draw_reasons: dict[str, int] = None

    def __post_init__(self) -> None:
        if self.draw_reasons is None:
            self.draw_reasons = {reason: 0 for reason in DRAW_REASONS}

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def add_game(self, result: GameResult) -> None:
        score = result.score
        moves = result.moves
        self.total_moves += moves
        if score > 0:
            self.wins += 1
        elif score < 0:
            self.losses += 1
        else:
            self.draws += 1
            self.draw_reasons[result.draw_reason] += 1

    @property
    def total_draw_reasons(self) -> int:
        return sum(self.draw_reasons.values())


def _resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path

    zip_path = f"{model_path}.zip"
    if os.path.exists(zip_path):
        return zip_path

    raise FileNotFoundError(
        f"Could not find model at '{model_path}' or '{zip_path}'. "
        "Run train.py first to create model.zip."
    )


def _model_action(model, observation: np.ndarray, action_mask: np.ndarray, deterministic: bool) -> int:
    obs_flat = observation.flatten().astype(np.float32)
    action, _ = model.predict(
        obs_flat,
        action_masks=action_mask.astype(bool),
        deterministic=deterministic,
    )
    return int(action)


def _random_legal_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        return 0
    return int(rng.choice(legal_actions))


def _classify_draw(board, truncated: bool, exhausted_without_terminal: bool) -> str:
    if truncated or exhausted_without_terminal:
        return "Max-moves / truncation"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Insufficient material"
    if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
        return "Threefold repetition"
    if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
        return "Fifty-move rule"
    return "Other draw"


def play_game(
    model,
    seed: int,
    deterministic: bool,
    max_iter: int,
    model_agent: str,
) -> GameResult:
    env = chess_v6.env()
    env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    rewards = {agent: 0.0 for agent in env.possible_agents}
    moves = 0
    terminated = False
    truncated = False

    try:
        for agent in env.agent_iter(max_iter=max_iter):
            observation, reward, termination, truncation, _ = env.last()
            rewards[agent] += float(reward)
            terminated = terminated or termination
            truncated = truncated or truncation

            if termination or truncation:
                action = None
            elif agent == model_agent:
                action = _model_action(
                    model,
                    observation["observation"],
                    observation["action_mask"],
                    deterministic,
                )
                moves += 1
            else:
                action = _random_legal_action(observation["action_mask"], rng)
                moves += 1

            env.step(action)

        score = rewards[model_agent]
        exhausted_without_terminal = not terminated and not truncated
        draw_reason = "-"
        if score == 0.0:
            draw_reason = _classify_draw(
                env.env.board,
                truncated=truncated,
                exhausted_without_terminal=exhausted_without_terminal,
            )

        return GameResult(
            score=score,
            moves=moves,
            model_agent=model_agent,
            draw_reason=draw_reason,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        env.close()


def evaluate(model_path: str, games: int, seed: int, deterministic: bool, max_iter: int) -> dict[str, EvalResult]:
    random.seed(seed)
    np.random.seed(seed)

    resolved_path = _resolve_model_path(model_path)
    model = ALGORITHM.load(resolved_path, device="cpu")
    results = {
        "overall": EvalResult(),
        "white": EvalResult(),
        "black": EvalResult(),
    }

    for game_idx in range(games):
        model_agent = "player_0" if game_idx % 2 == 0 else "player_1"
        color_key = "white" if model_agent == "player_0" else "black"
        game_result = play_game(
            model=model,
            seed=seed + game_idx,
            deterministic=deterministic,
            max_iter=max_iter,
            model_agent=model_agent,
        )
        results["overall"].add_game(game_result)
        results[color_key].add_game(game_result)

        print(
            f"Game {game_idx + 1:>3}/{games}: "
            f"model={'white' if model_agent == 'player_0' else 'black'}, "
            f"score={game_result.score:+.1f}, moves={game_result.moves}, "
            f"draw_reason={game_result.draw_reason}"
        )

    return results


def print_result(label: str, result: EvalResult) -> None:
    played = result.games
    win_rate = result.wins / played if played else 0.0
    loss_rate = result.losses / played if played else 0.0
    draw_rate = result.draws / played if played else 0.0
    avg_moves = result.total_moves / played if played else 0.0

    print(f"\n{label}")
    print(f"Games:     {played}")
    print(f"Wins:      {result.wins}")
    print(f"Losses:    {result.losses}")
    print(f"Draws:     {result.draws}")
    print(f"Win rate:  {win_rate:.1%}")
    print(f"Loss rate: {loss_rate:.1%}")
    print(f"Draw rate: {draw_rate:.1%}")
    print(f"Avg moves: {avg_moves:.1f}")
    print("Draw reasons:")
    for reason in DRAW_REASONS:
        print(f"  {reason}: {result.draw_reasons[reason]}")
    print(f"  Total draws: {result.total_draw_reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model.zip against a random legal-move opponent with alternating colors."
    )
    parser.add_argument("--model", default=SAVE_PATH, help="Model path, default: model/model.zip")
    parser.add_argument("--games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic model actions instead of deterministic actions",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum PettingZoo agent iterations per game",
    )
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model,
        games=args.games,
        seed=args.seed,
        deterministic=not args.stochastic,
        max_iter=args.max_iter,
    )

    print("\nEvaluation summary")
    print_result("Overall", results["overall"])
    print_result("As White", results["white"])
    print_result("As Black", results["black"])


if __name__ == "__main__":
    main()
