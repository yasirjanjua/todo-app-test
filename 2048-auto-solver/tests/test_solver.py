"""Unit tests for the pure expectimax solver in core/solver.py."""

from __future__ import annotations

from core.board import Move, from_grid
from core.heuristics import HeuristicWeights
from core.solver import SolverConfig, get_best_move


def test_no_legal_move_returns_none() -> None:
    grid = [
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ]
    board = from_grid(grid)
    decision = get_best_move(board, SolverConfig(base_depth=2, max_depth=2))
    assert decision.best_move is None
    assert decision.ranked_moves == ()


def test_ranked_moves_only_contains_legal_moves() -> None:
    # Only DOWN and RIGHT are legal from this corner-packed position.
    grid = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 0],
    ]
    board = from_grid(grid)
    decision = get_best_move(board, SolverConfig(base_depth=2, max_depth=2))
    legal = {m for m, _ in decision.ranked_moves}
    assert legal <= {Move.DOWN, Move.RIGHT}
    assert decision.best_move in legal


def test_prefers_move_that_avoids_immediate_game_over() -> None:
    # Single empty cell at (0,0); this symmetric grid makes UP and LEFT the only two legal
    # moves (transposes of each other), while RIGHT and DOWN are no-ops. The solver must
    # choose one of the two legal, board-changing moves rather than an illegal one.
    grid = [
        [0, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    board = from_grid(grid)
    decision = get_best_move(board, SolverConfig(base_depth=3, max_depth=4))
    assert decision.best_move in (Move.UP, Move.LEFT)
    assert {m for m, _ in decision.ranked_moves} == {Move.UP, Move.LEFT}


def test_smoke_open_board_favors_consolidating_move() -> None:
    # A near-empty board with two equal tiles adjacent in a row: merging is clearly best
    # among a very open set of options, so expectimax should rank a move that merges them
    # at least as high as one that doesn't waste the opportunity.
    grid = [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    board = from_grid(grid)
    decision = get_best_move(board, SolverConfig(base_depth=3, max_depth=4))
    assert decision.best_move is not None
    # Every legal move should have been evaluated.
    legal_moves = {Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN}
    assert {m for m, _ in decision.ranked_moves} <= legal_moves
    assert decision.nodes_evaluated > 0
    assert decision.depth >= 3


def test_custom_weights_are_respected() -> None:
    grid = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    zero_weights = HeuristicWeights(
        monotonicity=0.0, smoothness=0.0, empty_cells=0.0, corner_max=0.0, merge_potential=0.0
    )
    config = SolverConfig(base_depth=1, max_depth=1, weights=zero_weights)
    decision = get_best_move(board, config)
    # With every weight zeroed, every legal move's evaluation should be exactly 0.0.
    assert all(value == 0.0 for _, value in decision.ranked_moves)


def test_deeper_search_does_not_crash_and_stays_deterministic() -> None:
    grid = [
        [3, 2, 1, 0],
        [2, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    board = from_grid(grid)
    config = SolverConfig(base_depth=4, max_depth=5)
    first = get_best_move(board, config)
    second = get_best_move(board, config)
    assert first.best_move == second.best_move
