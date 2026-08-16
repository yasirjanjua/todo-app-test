"""Bitboard representation and move rules for 2048.

Design assumption: the board is a single 64-bit unsigned Python ``int``. It holds sixteen
4-bit "cells", one per grid position, in row-major order (index = row * 4 + col). Each cell
stores a *tier exponent* ``n`` meaning "tile rank n", which the rest of the app treats as the
value ``2**n`` (0 means empty). Because callers only ever deal in tiers -- never in on-screen
values -- this same representation works for a numeric board (2, 4, 8, ...) or an arbitrary
sprite board (fruit, anime icons, ...); see ``vision/tile_learning.py`` for how tiers are
learned from a screen.

A "row" is the 16-bit slice of the board holding one row's four cells, with cell ``c``'s
nibble at bit offset ``4 * c`` within the row (column 0 = least-significant nibble). All
possible left-moves and right-moves of a row are precomputed once at import time into
``ROW_LEFT_TABLE`` / ``ROW_RIGHT_TABLE`` (65,536 entries each), so applying a horizontal move
to a full board is four table lookups instead of shifting sixteen individual cells. Vertical
moves reuse the same tables by transposing the board, applying a horizontal move, and
transposing back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)

BOARD_SIZE: Final[int] = 4
NUM_CELLS: Final[int] = BOARD_SIZE * BOARD_SIZE
CELL_MASK: Final[int] = 0xF
ROW_MASK: Final[int] = 0xFFFF
FULL_BOARD_MASK: Final[int] = (1 << 64) - 1
MAX_TIER: Final[int] = 0xF  # a nibble tops out at 15 (tier 15 == on-screen value 32768)

# Empty-cell spawn tiers and their probabilities, per the standard 2048 spawn rule.
SPAWN_LOW_TIER: Final[int] = 1
SPAWN_HIGH_TIER: Final[int] = 2
SPAWN_LOW_PROBABILITY: Final[float] = 0.9
SPAWN_HIGH_PROBABILITY: Final[float] = 0.1


class Move(Enum):
    """The four cardinal moves a player can make."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class MoveResult:
    """Outcome of applying a :class:`Move` to a board."""

    board: int
    score: int
    moved: bool


def _row_cells(row_value: int) -> list[int]:
    return [(row_value >> (4 * i)) & CELL_MASK for i in range(BOARD_SIZE)]


def _pack_row(cells: list[int]) -> int:
    row_value = 0
    for i, cell in enumerate(cells):
        row_value |= (cell & CELL_MASK) << (4 * i)
    return row_value


def _compute_row_left(row_value: int) -> tuple[int, int, bool]:
    """Slide-and-merge a single row toward column 0 (a "move left")."""
    compacted = [cell for cell in _row_cells(row_value) if cell != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(compacted):
        current = compacted[i]
        if i + 1 < len(compacted) and compacted[i + 1] == current and current != MAX_TIER:
            new_tier = current + 1
            merged.append(new_tier)
            score += 1 << new_tier
            i += 2
        else:
            merged.append(current)
            i += 1
    merged.extend([0] * (BOARD_SIZE - len(merged)))
    new_row = _pack_row(merged)
    return new_row, score, new_row != row_value


def _reverse_row(row_value: int) -> int:
    cells = _row_cells(row_value)
    cells.reverse()
    return _pack_row(cells)


def _build_row_tables() -> tuple[list[tuple[int, int, bool]], list[tuple[int, int, bool]]]:
    """Precompute every possible row outcome for left and right moves (65,536 rows each)."""
    left: list[tuple[int, int, bool]] = [(0, 0, False)] * (ROW_MASK + 1)
    right: list[tuple[int, int, bool]] = [(0, 0, False)] * (ROW_MASK + 1)
    for row_value in range(ROW_MASK + 1):
        left[row_value] = _compute_row_left(row_value)
    for row_value in range(ROW_MASK + 1):
        reversed_value = _reverse_row(row_value)
        new_reversed, score, _ = left[reversed_value]
        new_row = _reverse_row(new_reversed)
        right[row_value] = (new_row, score, new_row != row_value)
    return left, right


logger.debug("Precomputing 2048 row-move lookup tables (2 x 65,536 entries)...")
ROW_LEFT_TABLE, ROW_RIGHT_TABLE = _build_row_tables()
logger.debug("Row-move lookup tables ready.")


def _apply_row_table(
    board: int, table: list[tuple[int, int, bool]]
) -> tuple[int, int, bool]:
    new_board = 0
    total_score = 0
    moved = False
    for row_index in range(BOARD_SIZE):
        shift = 16 * row_index
        row_value = (board >> shift) & ROW_MASK
        new_row, score, row_moved = table[row_value]
        new_board |= new_row << shift
        total_score += score
        moved = moved or row_moved
    return new_board, total_score, moved


def transpose(board: int) -> int:
    """Swap rows and columns, mapping cell (r, c) to (c, r).

    Used to implement vertical moves as horizontal moves: transpose, apply a row table,
    transpose back.
    """
    grid = to_grid(board)
    transposed = [[grid[c][r] for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)]
    return from_grid(transposed)


def move(board: int, direction: Move) -> MoveResult:
    """Apply ``direction`` to ``board`` and return the resulting board, score, and moved flag."""
    if direction is Move.LEFT:
        new_board, score, moved = _apply_row_table(board, ROW_LEFT_TABLE)
    elif direction is Move.RIGHT:
        new_board, score, moved = _apply_row_table(board, ROW_RIGHT_TABLE)
    elif direction is Move.UP:
        transposed = transpose(board)
        new_transposed, score, moved = _apply_row_table(transposed, ROW_LEFT_TABLE)
        new_board = transpose(new_transposed)
    elif direction is Move.DOWN:
        transposed = transpose(board)
        new_transposed, score, moved = _apply_row_table(transposed, ROW_RIGHT_TABLE)
        new_board = transpose(new_transposed)
    else:  # pragma: no cover - Move is an exhaustive Enum
        raise ValueError(f"Unknown move direction: {direction!r}")
    return MoveResult(board=new_board, score=score, moved=moved)


def get_cell(board: int, index: int) -> int:
    """Return the tier exponent stored at row-major cell ``index`` (0-15)."""
    if not 0 <= index < NUM_CELLS:
        raise IndexError(f"Cell index out of range: {index}")
    return (board >> (4 * index)) & CELL_MASK


def set_cell(board: int, index: int, tier: int) -> int:
    """Return a new board with cell ``index`` set to ``tier`` (0 clears the cell)."""
    if not 0 <= index < NUM_CELLS:
        raise IndexError(f"Cell index out of range: {index}")
    if not 0 <= tier <= MAX_TIER:
        raise ValueError(f"Tier out of range for a 4-bit cell: {tier}")
    shift = 4 * index
    cleared = board & (FULL_BOARD_MASK ^ (CELL_MASK << shift))
    return cleared | (tier << shift)


def spawn_tile(board: int, index: int, tier: int) -> int:
    """Place a newly spawned tile. Semantically identical to :func:`set_cell`."""
    return set_cell(board, index, tier)


def empty_cells(board: int) -> list[int]:
    """Return the row-major indices of every empty cell."""
    return [i for i in range(NUM_CELLS) if get_cell(board, i) == 0]


def count_empty(board: int) -> int:
    """Return how many cells are empty."""
    count = 0
    for i in range(NUM_CELLS):
        if get_cell(board, i) == 0:
            count += 1
    return count


def max_tile(board: int) -> int:
    """Return the highest tier exponent present on the board."""
    return max(get_cell(board, i) for i in range(NUM_CELLS))


def to_grid(board: int) -> list[list[int]]:
    """Convert to a 4x4 nested list of tier exponents, row-major."""
    return [
        [get_cell(board, row * BOARD_SIZE + col) for col in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE)
    ]


def from_grid(grid: list[list[int]]) -> int:
    """Inverse of :func:`to_grid`."""
    board = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            board = set_cell(board, row * BOARD_SIZE + col, grid[row][col])
    return board


def is_game_over(board: int) -> bool:
    """A board is game-over when no move in any direction changes it."""
    return not any(move(board, direction).moved for direction in Move)


def board_to_str(board: int) -> str:
    """Human-readable board dump for logs and debug output (tier exponents, not values)."""
    grid = to_grid(board)
    return "\n".join(" ".join(f"{cell:2d}" for cell in row) for row in grid)
