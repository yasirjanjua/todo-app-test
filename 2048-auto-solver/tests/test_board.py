"""Unit tests for the pure bitboard engine in core/board.py."""

from __future__ import annotations

from core.board import (
    Move,
    count_empty,
    empty_cells,
    from_grid,
    get_cell,
    is_game_over,
    max_tile,
    move,
    set_cell,
    to_grid,
)


def test_grid_roundtrip() -> None:
    grid = [
        [1, 0, 2, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 4],
        [1, 1, 0, 0],
    ]
    board = from_grid(grid)
    assert to_grid(board) == grid


def test_get_set_cell() -> None:
    board = 0
    board = set_cell(board, 5, 3)
    assert get_cell(board, 5) == 3
    for i in range(16):
        if i != 5:
            assert get_cell(board, i) == 0


def test_empty_cells_and_count() -> None:
    grid = [
        [1, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [1, 1, 0, 0],
    ]
    board = from_grid(grid)
    empties = empty_cells(board)
    assert count_empty(board) == len(empties)
    assert count_empty(board) == 11
    assert 1 in empties  # cell (0,1) is empty
    assert get_cell(board, 1) == 0


def test_move_left_merges_and_compacts() -> None:
    # Row: 2 2 4 4 (tiers 1 1 2 2) -> merges into 4 8 (tiers 2 3), padded with zeros.
    grid = [[1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.LEFT)
    assert result.moved is True
    new_grid = to_grid(result.board)
    assert new_grid[0] == [2, 3, 0, 0]
    # Score gained = 4 (from merging two 2s) + 8 (from merging two 4s) = 12
    assert result.score == 4 + 8


def test_move_right() -> None:
    grid = [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.RIGHT)
    assert to_grid(result.board)[0] == [0, 0, 0, 2]
    assert result.score == 4


def test_move_up_and_down() -> None:
    # Column 0: rows 0..3 = 1,1,0,0 -> moving up merges to tier 2 at row 0.
    grid = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)

    up_result = move(board, Move.UP)
    up_grid = to_grid(up_result.board)
    assert [up_grid[r][0] for r in range(4)] == [2, 0, 0, 0]
    assert up_result.score == 4

    down_result = move(board, Move.DOWN)
    down_grid = to_grid(down_result.board)
    assert [down_grid[r][0] for r in range(4)] == [0, 0, 0, 2]
    assert down_result.score == 4


def test_move_no_merge_across_gap_without_shift_is_still_moved() -> None:
    # 2 _ _ 2 moving left should merge (gap doesn't block sliding).
    grid = [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.LEFT)
    assert to_grid(result.board)[0] == [2, 0, 0, 0]


def test_move_does_not_double_merge() -> None:
    # 2 2 2 2 moving left should produce 4 4, not 8.
    grid = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.LEFT)
    assert to_grid(result.board)[0] == [2, 2, 0, 0]
    assert result.score == 4 + 4


def test_move_not_moved_flag_when_already_settled() -> None:
    grid = [[2, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.LEFT)
    assert result.moved is False
    assert result.board == board


def test_max_tile() -> None:
    grid = [[1, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    assert max_tile(board) == 5


def test_is_game_over_true_when_no_moves_change_board() -> None:
    # Full board, no adjacent equal tiers anywhere and no empty cells -> game over.
    grid = [
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 2, 1],
    ]
    board = from_grid(grid)
    assert count_empty(board) == 0
    assert is_game_over(board) is True


def test_is_game_over_false_when_a_move_is_available() -> None:
    grid = [
        [1, 2, 1, 2],
        [2, 1, 2, 1],
        [1, 2, 1, 2],
        [2, 1, 1, 2],  # two adjacent 1s at the end of the last row
    ]
    board = from_grid(grid)
    assert is_game_over(board) is False


def test_high_tier_cannot_merge_past_nibble_max() -> None:
    grid = [[15, 15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    board = from_grid(grid)
    result = move(board, Move.LEFT)
    # Tier 15 is the max a nibble can hold; two of them must not attempt to merge into 16.
    assert to_grid(result.board)[0] == [15, 15, 0, 0]
    assert result.moved is False
