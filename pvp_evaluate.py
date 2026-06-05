#!/usr/bin/env python3
"""Evaluate two trained chess models against each other with alternating colors."""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
from pettingzoo.classic import chess_v6

from seaturtle_model import ChessCNN
import model as model_config

model_config.ChessCNN = ChessCNN
ALGORITHM = model_config.ALGORITHM


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
    model_a_agent: str
    model_b_agent: str
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
        self.total_moves += result.moves
        if result.score > 0:
            self.wins += 1
        elif result.score < 0:
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

    raise FileNotFoundError(f"Could not find model at '{model_path}' or '{zip_path}'.")


def _model_action(model, observation: np.ndarray, action_mask: np.ndarray, deterministic: bool) -> int:
    obs_flat = observation.flatten().astype(np.float32)
    action, _ = model.predict(
        obs_flat,
        action_masks=action_mask.astype(bool),
        deterministic=deterministic,
    )
    return int(action)


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
    model_a,
    model_b,
    seed: int,
    deterministic: bool,
    max_iter: int,
    model_a_agent: str,
) -> GameResult:
    env = chess_v6.env()
    env.reset(seed=seed)

    model_b_agent = "player_1" if model_a_agent == "player_0" else "player_0"
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
            elif agent == model_a_agent:
                action = _model_action(
                    model_a,
                    observation["observation"],
                    observation["action_mask"],
                    deterministic,
                )
                moves += 1
            else:
                action = _model_action(
                    model_b,
                    observation["observation"],
                    observation["action_mask"],
                    deterministic,
                )
                moves += 1

            env.step(action)

        score = rewards[model_a_agent]
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
            model_a_agent=model_a_agent,
            model_b_agent=model_b_agent,
            draw_reason=draw_reason,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        env.close()


def evaluate(
    model_a_path: str,
    model_b_path: str,
    games: int,
    seed: int,
    deterministic: bool,
    max_iter: int,
) -> dict[str, EvalResult]:
    random.seed(seed)
    np.random.seed(seed)

    resolved_a_path = _resolve_model_path(model_a_path)
    resolved_b_path = _resolve_model_path(model_b_path)
    model_a = ALGORITHM.load(resolved_a_path, device="cpu")
    model_b = ALGORITHM.load(resolved_b_path, device="cpu")
    results = {
        "overall": EvalResult(),
        "white": EvalResult(),
        "black": EvalResult(),
    }

    print("PvP evaluation parameters")
    print("Environment: PettingZoo chess_v6")
    print(f"Games: {games}")
    print(f"Evaluation seed: {seed}")
    print(f"Deterministic model action: {deterministic}")
    print(f"Stochastic mode: {not deterministic}")
    print(f"Max iterations: {max_iter}")
    print("Score perspective: Model A")
    print("Random opponent: none")
    print(f"Model A file: {resolved_a_path}")
    print(f"Model B file: {resolved_b_path}")
    print("Model A role: current model")
    print("Model B role: opponent model")
    print("Compatibility loader: seaturtle_model.ChessCNN registered as model.ChessCNN")

    print("\nColor assignment")
    print("Color assignment: alternating")
    print(f"Model A white games: {(games + 1) // 2}")
    print(f"Model A black games: {games // 2}")
    if games >= 1:
        print("Game 1: Model A white")
    if games >= 2:
        print("Game 2: Model A black")

    for game_idx in range(games):
        model_a_agent = "player_0" if game_idx % 2 == 0 else "player_1"
        color_key = "white" if model_a_agent == "player_0" else "black"
        model_a_color = "white" if model_a_agent == "player_0" else "black"
        model_b_color = "black" if model_a_color == "white" else "white"
        game_result = play_game(
            model_a=model_a,
            model_b=model_b,
            seed=seed + game_idx,
            deterministic=deterministic,
            max_iter=max_iter,
            model_a_agent=model_a_agent,
        )
        results["overall"].add_game(game_result)
        results[color_key].add_game(game_result)

        print(
            f"Game {game_idx + 1}/{games}: "
            f"Model A color={model_a_color}, "
            f"Model B color={model_b_color}, "
            f"score from Model A perspective={game_result.score:+.1f}, "
            f"moves={game_result.moves}, "
            f"terminated={game_result.terminated}, "
            f"truncated={game_result.truncated}, "
            f"draw_reason={game_result.draw_reason}"
        )

    return results


def print_result(label: str, result: EvalResult) -> None:
    played = result.games
    win_rate = result.wins / played if played else 0.0
    loss_rate = result.losses / played if played else 0.0
    draw_rate = result.draws / played if played else 0.0
    avg_moves = result.total_moves / played if played else 0.0

    print(f"\nModel A {label}")
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


def print_markdown_tables(results: dict[str, EvalResult]) -> None:
    overall = results["overall"]
    white = results["white"]
    black = results["black"]

    def rate(numerator: int, denominator: int) -> str:
        return f"{(numerator / denominator):.1%}" if denominator else "0.0%"

    def avg_moves(result: EvalResult) -> str:
        return f"{(result.total_moves / result.games):.1f}" if result.games else "0.0"

    print("\nModel A PvP results")
    print("| Metric | Overall | As White | As Black |")
    print("|---|---:|---:|---:|")
    print(f"| Games | {overall.games} | {white.games} | {black.games} |")
    print(f"| Wins | {overall.wins} | {white.wins} | {black.wins} |")
    print(f"| Losses | {overall.losses} | {white.losses} | {black.losses} |")
    print(f"| Draws | {overall.draws} | {white.draws} | {black.draws} |")
    print(f"| Win rate | {rate(overall.wins, overall.games)} | {rate(white.wins, white.games)} | {rate(black.wins, black.games)} |")
    print(f"| Loss rate | {rate(overall.losses, overall.games)} | {rate(white.losses, white.games)} | {rate(black.losses, black.games)} |")
    print(f"| Draw rate | {rate(overall.draws, overall.games)} | {rate(white.draws, white.games)} | {rate(black.draws, black.games)} |")
    print(f"| Avg moves | {avg_moves(overall)} | {avg_moves(white)} | {avg_moves(black)} |")

    print("\nPvP draw reasons")
    print("| Draw reason | Overall | Model A White | Model A Black |")
    print("|---|---:|---:|---:|")
    for reason in DRAW_REASONS:
        print(
            f"| {reason} | {overall.draw_reasons[reason]} | "
            f"{white.draw_reasons[reason]} | {black.draw_reasons[reason]} |"
        )
    print(f"| Total draws | {overall.total_draw_reasons} | {white.total_draw_reasons} | {black.total_draw_reasons} |")

    print("\nRandom opponent baseline comparison")
    print("| Metric | Previous vs Random | Current vs Random | Current vs Previous PvP |")
    print("|---|---:|---:|---:|")
    print(f"| Games | 1000 | 1000 | {overall.games} |")
    print(f"| Win rate | 28.1% | 33.1% | {rate(overall.wins, overall.games)} |")
    print(f"| Loss rate | 1.4% | 1.2% | {rate(overall.losses, overall.games)} |")
    print(f"| Draw rate | 70.5% | 65.7% | {rate(overall.draws, overall.games)} |")
    print(f"| Avg moves | 242.1 | 224.9 | {avg_moves(overall)} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate two chess models against each other with alternating colors."
    )
    parser.add_argument("--model-a", required=True, help="Model A path")
    parser.add_argument("--model-b", required=True, help="Model B path")
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
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        games=args.games,
        seed=args.seed,
        deterministic=not args.stochastic,
        max_iter=args.max_iter,
    )

    print("\nEvaluation summary")
    print_result("Overall", results["overall"])
    print_result("As White", results["white"])
    print_result("As Black", results["black"])
    print_markdown_tables(results)


if __name__ == "__main__":
    main()
