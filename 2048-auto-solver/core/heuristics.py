"""Static board evaluation used by the expectimax solver.

The evaluation function is a weighted sum of five signals, each cheap to compute from the
4x4 grid of tier exponents. Weights are documented, named, and centralized in
:class:`HeuristicWeights` so they can be tuned (or exposed in the UI's advanced panel)
without touching the search code.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.board import to_grid

BOARD_SIZE = 4


@dataclass(frozen=True)
class HeuristicWeights:
    """Weights for each term of the static evaluation function.

    Defaults are hand-tuned starting points broadly consistent with published 2048-bot
    heuristics: empty cells matter most (keeping options open is the single best predictor
    of survival), monotonicity keeps large tiles ordered toward a corner, smoothness keeps
    neighboring tiles close in value (cheap future merges), and the corner/merge terms give
    small nudges toward stacking the max tile and setting up immediate merges.
    """

    monotonicity: float = 1.0
    smoothness: float = 0.1
    empty_cells: float = 2.7
    corner_max: float = 1.0
    merge_potential: float = 0.5


DEFAULT_WEIGHTS = HeuristicWeights()


def _monotonicity(grid: list[list[int]]) -> float:
    """Reward rows/columns that are monotonic (non-increasing or non-decreasing).

    For each row and each column we accumulate a "penalty" walking in both directions and
    keep whichever direction is cheaper, then return the negated total so higher is better.
    """
    totals = [0.0, 0.0, 0.0, 0.0]  # row-increasing, row-decreasing, col-increasing, col-decreasing

    for row in grid:
        for i in range(BOARD_SIZE - 1):
            current, nxt = row[i], row[i + 1]
            if current > nxt:
                totals[0] += current - nxt
            elif nxt > current:
                totals[1] += nxt - current

    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE - 1):
            current, nxt = grid[row][col], grid[row + 1][col]
            if current > nxt:
                totals[2] += current - nxt
            elif nxt > current:
                totals[3] += nxt - current

    penalty = min(totals[0], totals[1]) + min(totals[2], totals[3])
    return -penalty


def _smoothness(grid: list[list[int]]) -> float:
    """Reward neighboring non-empty tiles having close tier values (cheap future merges)."""
    score = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            value = grid[row][col]
            if value == 0:
                continue
            if col + 1 < BOARD_SIZE and grid[row][col + 1] != 0:
                score -= abs(value - grid[row][col + 1])
            if row + 1 < BOARD_SIZE and grid[row + 1][col] != 0:
                score -= abs(value - grid[row + 1][col])
    return score


def _empty_cell_count(grid: list[list[int]]) -> float:
    return float(sum(1 for row in grid for cell in row if cell == 0))


def _corner_max_bonus(grid: list[list[int]]) -> float:
    """Bonus (scaled by the max tile's own value) for keeping the max tile in a corner."""
    highest = max(cell for row in grid for cell in row)
    if highest == 0:
        return 0.0
    corners = (grid[0][0], grid[0][BOARD_SIZE - 1], grid[BOARD_SIZE - 1][0], grid[BOARD_SIZE - 1][BOARD_SIZE - 1])
    return float(highest) if highest in corners else 0.0


def _merge_potential(grid: list[list[int]]) -> float:
    """Count adjacent equal-tier pairs: each is an immediately available merge."""
    count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            value = grid[row][col]
            if value == 0:
                continue
            if col + 1 < BOARD_SIZE and grid[row][col + 1] == value:
                count += 1
            if row + 1 < BOARD_SIZE and grid[row + 1][col] == value:
                count += 1
    return float(count)


def evaluate(board: int, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    """Return a scalar desirability score for ``board``; higher is better."""
    grid = to_grid(board)
    return (
        weights.monotonicity * _monotonicity(grid)
        + weights.smoothness * _smoothness(grid)
        + weights.empty_cells * _empty_cell_count(grid)
        + weights.corner_max * _corner_max_bonus(grid)
        + weights.merge_potential * _merge_potential(grid)
    )
