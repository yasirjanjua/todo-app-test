"""Expectimax solver for 2048.

Pure by design: everything here operates on the bitboard ``int`` from ``core.board`` and the
weights from ``core.heuristics``. No I/O, no screen access, no OS calls -- which is what lets
it be unit-tested against fixed boards (see ``tests/test_solver.py``) and reused unchanged
by any future frontend beyond the desktop app in ``app/``.

Search structure follows the standard 2048-bot alternation:

* **Max nodes** are the player's turn: try each of the four moves, recurse into the chance
  node that follows, and keep the best.
* **Chance nodes** are the game's turn: a tile spawns in one of the empty cells, 90% of the
  time as a tier-1 tile and 10% of the time as tier-2, uniformly over empty cells. The
  expected value over all spawns feeds back up to the max node above it.

Two things keep this tractable in pure Python:

* **Adaptive depth.** An open board (many empty cells) is searched shallowly because most
  moves are safe and the branching factor is large; a nearly-full board is searched deeply
  because that's exactly when a wrong move ends the game.
* **Transposition table + probability pruning.** Each node result is cached by
  ``(board, depth)`` for the lifetime of a single top-level decision, and chance branches
  whose cumulative reach-probability has fallen below ``prob_threshold`` are evaluated
  directly with the static heuristic instead of being expanded further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.board import Move, count_empty, empty_cells, move, set_cell
from core.heuristics import DEFAULT_WEIGHTS, HeuristicWeights, evaluate

_MOVES: tuple[Move, ...] = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

# Applied as a static penalty when a max node has no legal move (the game is over there).
# Large enough to always be dominated by any position that still has a legal move.
GAME_OVER_PENALTY: float = 200_000.0


@dataclass(frozen=True)
class SolverConfig:
    """Tunable knobs for the expectimax search. Exposed via the UI's advanced panel."""

    base_depth: int = 3
    max_depth: int = 6
    prob_threshold: float = 1e-4
    weights: HeuristicWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)


@dataclass(frozen=True)
class SolverDecision:
    """Result of a single top-level search: the chosen move plus enough detail to explain it."""

    best_move: Move | None
    ranked_moves: tuple[tuple[Move, float], ...]
    depth: int
    nodes_evaluated: int


def _adaptive_depth(empty: int, config: SolverConfig) -> int:
    """Search deeper as the board fills up; shallower when it's wide open."""
    if empty <= 1:
        bonus = 3
    elif empty <= 3:
        bonus = 2
    elif empty <= 5:
        bonus = 1
    else:
        bonus = 0
    return min(config.max_depth, config.base_depth + bonus)


def _max_node(
    board: int, depth: int, prob: float, cache: dict[tuple[int, int], float], config: SolverConfig
) -> float:
    """Player-turn node: return the value of the best legal move, or a static eval at depth 0."""
    if depth <= 0 or prob < config.prob_threshold:
        return evaluate(board, config.weights)

    key = (board, depth)
    cached = cache.get(key)
    if cached is not None:
        return cached

    best = -math.inf
    moved_any = False
    for direction in _MOVES:
        result = move(board, direction)
        if not result.moved:
            continue
        moved_any = True
        value = _chance_node(result.board, depth, prob, cache, config)
        if value > best:
            best = value

    if not moved_any:
        best = evaluate(board, config.weights) - GAME_OVER_PENALTY

    cache[key] = best
    return best


def _chance_node(
    board: int, depth: int, prob: float, cache: dict[tuple[int, int], float], config: SolverConfig
) -> float:
    """Game-turn node: expected value over every possible tile spawn."""
    key = (board, -depth)  # negate depth to keep chance-node keys disjoint from max-node keys
    cached = cache.get(key)
    if cached is not None:
        return cached

    empties = empty_cells(board)
    if not empties:
        value = _max_node(board, depth - 1, prob, cache, config)
        cache[key] = value
        return value

    slot_count = len(empties)
    per_slot_prob = prob / slot_count
    total = 0.0
    for index in empties:
        low_board = set_cell(board, index, 1)
        total += (0.9 / slot_count) * _max_node(low_board, depth - 1, per_slot_prob * 0.9, cache, config)
        if per_slot_prob * 0.1 >= config.prob_threshold:
            high_board = set_cell(board, index, 2)
            total += (0.1 / slot_count) * _max_node(high_board, depth - 1, per_slot_prob * 0.1, cache, config)
        else:
            # Too improbable to expand separately; fold its share into the low-tier branch's
            # weight so probability mass isn't silently discarded.
            total += (0.1 / slot_count) * _max_node(low_board, depth - 1, per_slot_prob * 0.1, cache, config)

    cache[key] = total
    return total


def get_best_move(board: int, config: SolverConfig | None = None) -> SolverDecision:
    """Run one top-level expectimax search and return the recommended move.

    ``ranked_moves`` lists every legal move best-first, so a caller whose keystroke for
    ``best_move`` turns out to be illegal (screen didn't change) can fall through to the
    next-ranked move without re-running the search.
    """
    config = config or SolverConfig()
    empties = count_empty(board)
    depth = _adaptive_depth(empties, config)
    cache: dict[tuple[int, int], float] = {}

    ranked: list[tuple[Move, float]] = []
    for direction in _MOVES:
        result = move(board, direction)
        if not result.moved:
            continue
        value = _chance_node(result.board, depth, 1.0, cache, config)
        ranked.append((direction, value))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    best = ranked[0][0] if ranked else None

    return SolverDecision(
        best_move=best,
        ranked_moves=tuple(ranked),
        depth=depth,
        nodes_evaluated=len(cache),
    )
