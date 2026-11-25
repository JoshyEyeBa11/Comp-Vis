"""
test_move_detector.py - pytest tests for move_detector.py, run WITHOUT a camera.

Each test takes the empty-board photo, draws fake pieces on it, and plays whole
games: a fake hand picks up each piece, carries it, and puts it down, while the
picture also gets camera noise and small brightness changes. Then it checks that
move_detector.py found every move correctly.

The lesson moves are read from the JSON files in Chess_Openings/, so the tests
always check exactly what the lessons teach.

Run with pytest (from FINAL, or from mainChess without the prefix):

    pytest mainChess/

The fake pieces are simple drawings, so this proves the LOGIC works.
It does not prove that the settings suit your real pieces. For that, use your own video.
"""

import json
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from board_setup import find_corners_auto, flatten_board, BOARD_SIZE, SQUARE, FILES
from move_detector import MoveTracker, load_corners

HERE = Path(__file__).resolve().parent
BOARD_PHOTO = HERE / "Board_WhiteSide.jpg"
LESSON_DIR = HERE / "Chess_Openings"

FPS = 15
SETTLE = 17            # still frames at the end of each move (1.1 seconds)


# ---------- lesson data ----------

def load_lesson_lines() -> dict[str, list[str]]:
    """Read the opening name and the SAN move list from every lesson JSON file."""
    lines = {}
    for path in sorted(LESSON_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lines[data["name"]] = [move["san"] for move in data["moves"]]
    return lines


LESSON_LINES = load_lesson_lines()


# ---------- drawing the fake scene ----------

def centre(name: str) -> tuple[int, int]:
    """Pixel position of the middle of a square on the flat board."""
    col, row = FILES.index(name[0]), 8 - int(name[1])
    return col * SQUARE + SQUARE // 2, row * SQUARE + SQUARE // 2


def draw_piece(img: np.ndarray, x: int, y: int, white: bool) -> None:
    """Draw one chess piece as a simple round counter with a shadow."""
    cv2.ellipse(img, (x + 5, y + 9), (27, 19), 0, 0, 360, (45, 55, 55), -1)               # shadow
    body, line = ((225, 235, 242), (70, 80, 90)) if white else ((38, 32, 32), (150, 150, 150))
    cv2.circle(img, (x, y), 28, body, -1)
    cv2.circle(img, (x, y), 28, line, 3)
    cv2.circle(img, (x, y - 6), 12, line, 2)


def draw_hand(img: np.ndarray, x: int, y: int, big: bool = False) -> None:
    """Draw a kid's hand and arm reaching in from the bottom right."""
    skin = (125, 155, 210)
    cv2.line(img, (x + 120, BOARD_SIZE + 150), (x, y), skin, 80 if big else 55)           # arm
    cv2.ellipse(img, (x, y), (70, 48) if big else (48, 34), -30, 0, 360, skin, -1)         # hand


class Scene:
    """The fake world: a flat empty board, the pieces on it, and a way to make camera pictures."""

    def __init__(self, photo: np.ndarray):
        self.photo = photo
        self.h, self.w = photo.shape[:2]
        self.corners = find_corners_auto(photo)
        if self.corners is None:
            raise ValueError("I could not find the board in the photo. Try a photo with the whole board in view.")
        self.flat_empty = flatten_board(photo, self.corners)
        target = np.array([[0, 799], [799, 799], [799, 0], [0, 0]], dtype=np.float32)
        self.to_photo = np.linalg.inv(cv2.getPerspectiveTransform(self.corners, target))
        self.mask = cv2.warpPerspective(np.full((800, 800), 255, np.uint8), self.to_photo, (self.w, self.h))

    def picture(self, pieces: dict[str, bool], rng: np.random.Generator,
                hand: tuple[int, int] | None = None, carried: bool | None = None,
                big_hand: bool = False) -> np.ndarray:
        """pieces: {'e2': True (white) / False (black)}. hand: (x, y) or None."""
        flat = self.flat_empty.copy()
        for name, white in pieces.items():
            draw_piece(flat, *centre(name), white)
        if hand is not None:
            draw_hand(flat, *hand, big=big_hand)
            if carried is not None:
                draw_piece(flat, hand[0] - 8, hand[1] - 8, carried)
        flat = flat.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.normal(0, 4, flat.shape)   # light + noise
        flat = np.clip(flat, 0, 255).astype(np.uint8)
        warped = cv2.warpPerspective(flat, self.to_photo, (self.w, self.h))
        return np.where(self.mask[..., None] > 0, warped, self.photo)


# ---------- playing a game with a fake hand ----------

class Runner:
    """Feeds fake camera pictures of the scene to the move tracker, one by one."""

    def __init__(self, scene: Scene, seed: int = 7):
        self.scene = scene
        self.corners = scene.corners
        self.tracker = MoveTracker()
        self.frame_no = 0
        self.events: list[dict] = []
        self.rng = np.random.default_rng(seed)          # each test gets its own noise, so tests do not depend on order

    def feed(self, pieces: dict[str, bool], **kw) -> np.ndarray:
        """Show one frame to the tracker and remember any event it reports."""
        frame = self.scene.picture(pieces, self.rng, **kw)
        flat = flatten_board(frame, self.corners)
        event = self.tracker.update(flat, self.frame_no / FPS)
        self.frame_no += 1
        if event:
            self.events.append(event)
        return flat

    def still(self, pieces: dict[str, bool], frames: int = SETTLE) -> None:
        """Show the same quiet scene for a while."""
        for _ in range(frames):
            self.feed(pieces)

    def trip(self, pieces: dict[str, bool], src: str, dst: str,
             removed: tuple[str, ...] = (), settle: int = SETTLE) -> dict[str, bool]:
        """The hand picks the piece on src, carries it to dst, puts it down, and leaves."""
        pieces = dict(pieces)
        white = pieces[src]
        x0, y0 = centre(src)
        x1, y1 = centre(dst)
        for t in np.linspace(0, 1, 5):                                   # hand comes in
            self.feed(pieces, hand=(int(840 + (x0 - 840) * t), int(940 + (y0 - 940) * t)))
        del pieces[src]
        for r in removed:
            pieces.pop(r, None)
        for _ in range(2):                                               # lifts the piece
            self.feed(pieces, hand=(x0, y0), carried=white)
        for t in np.linspace(0, 1, 5):                                   # carries it
            self.feed(pieces, hand=(int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)), carried=white)
        pieces[dst] = white
        for _ in range(2):                                               # puts it down
            self.feed(pieces, hand=(x1 + 25, y1 + 25))
        for t in np.linspace(0, 1, 5):                                   # hand leaves
            self.feed(pieces, hand=(int(x1 + (840 - x1) * t), int(y1 + (940 - y1) * t)))
        self.still(pieces, settle)
        return pieces


def piece_colours(board: chess.Board) -> dict[str, bool]:
    """The pieces on the board as the Scene understands them: square name -> is it white?"""
    return {chess.square_name(s): (p.color == chess.WHITE) for s, p in board.piece_map().items()}


def play_move(runner: Runner, board: chess.Board, move: chess.Move,
              settle: int = SETTLE) -> dict[str, bool]:
    """Show one chess move being played by the fake hand. Returns the new pieces."""
    before = piece_colours(board)
    from_name, to_name = chess.square_name(move.from_square), chess.square_name(move.to_square)
    castling = board.is_castling(move)
    en_passant = board.is_en_passant(move)
    rank = chess.square_rank(move.from_square)
    rook_from = rook_to = None
    if castling:
        king_side = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from, rook_to = ((chess.square(7, rank), chess.square(5, rank)) if king_side
                              else (chess.square(0, rank), chess.square(3, rank)))
    removed = [to_name] if board.is_capture(move) and not en_passant else []
    if en_passant:
        removed = [chess.square_name(chess.square(chess.square_file(move.to_square), rank))]

    if castling and rook_from is not None and rook_to is not None:
        pieces = runner.trip(before, from_name, to_name, settle=4)
        pieces = runner.trip(pieces, chess.square_name(rook_from), chess.square_name(rook_to), settle=settle)
    else:
        pieces = runner.trip(before, from_name, to_name, removed=removed, settle=settle)
    board.push(move)
    return pieces


def start_game(runner: Runner, board: chess.Board) -> dict[str, bool]:
    """Set up the pieces, let the scene settle, and start the tracker watching."""
    pieces = piece_colours(board)
    runner.still(pieces)
    runner.tracker.start(runner.feed(pieces), chess.Board())
    return pieces


def play_line(runner: Runner, board: chess.Board, sans: list[str]) -> tuple[list[str], list[str]]:
    """Play a list of SAN moves on the fake board.
    Returns (expected moves, moves the tracker found), both as uci strings."""
    expected = []
    for san in sans:
        move = board.parse_san(san)
        expected.append(move.uci())
        play_move(runner, board, move)
    return expected, [m.uci() for m in runner.tracker.board.move_stack]


# ---------- the tests ----------

@pytest.fixture(scope="session")
def scene() -> Scene:
    """The fake world is expensive to build (corner finding), so build it once."""
    photo = cv2.imread(str(BOARD_PHOTO))
    if photo is None:
        pytest.fail(f"I could not open the board photo at {BOARD_PHOTO}.")
    photo = cv2.resize(photo, None, fx=1152 / photo.shape[1], fy=1152 / photo.shape[1],
                       interpolation=cv2.INTER_AREA)
    return Scene(photo)


@pytest.mark.parametrize("name", sorted(LESSON_LINES))
def test_lesson_is_detected_end_to_end(scene: Scene, name: str) -> None:
    """Every move of every lesson opening is detected, in order, with no extras."""
    runner = Runner(scene, seed=7)
    board = chess.Board()
    start_game(runner, board)
    expected, found = play_line(runner, board, LESSON_LINES[name])
    assert found == expected, (
        f"{name}: the detector lost track\n         expected {expected}\n         found    {found}")


def test_piece_lifted_and_put_back_is_not_a_move(scene: Scene) -> None:
    """A piece picked up and set down on the same square changes nothing."""
    runner = Runner(scene, seed=11)
    board = chess.Board()
    pieces = start_game(runner, board)
    runner.trip(pieces, "g1", "g1")
    assert len(runner.tracker.board.move_stack) == 0, "the empty trip was counted as a move"
    play_move(runner, board, board.parse_san("e4"))
    assert [m.uci() for m in runner.tracker.board.move_stack] == ["e2e4"], \
        "the next real move after the empty trip was not detected"


def test_hand_resting_on_the_board_is_not_a_move(scene: Scene) -> None:
    """A hand parked over the board for 3 seconds must not become a move."""
    runner = Runner(scene, seed=13)
    board = chess.Board()
    pieces = start_game(runner, board)
    for t in np.linspace(0, 1, 5):
        runner.feed(pieces, hand=(int(840 + (400 - 840) * t), int(940 + (400 - 940) * t)), big_hand=True)
    for _ in range(FPS * 3):
        runner.feed(pieces, hand=(400, 400), big_hand=True)
    for t in np.linspace(0, 1, 5):
        runner.feed(pieces, hand=(int(400 + 440 * t), int(400 + 540 * t)), big_hand=True)
    runner.still(pieces)
    types = {e["type"] for e in runner.events}
    assert len(runner.tracker.board.move_stack) == 0, "the resting hand was counted as a move"
    assert "hand" in types or "unclear" in types, f"the resting hand went unnoticed; events seen: {sorted(types)}"
    play_move(runner, board, board.parse_san("d4"))
    assert [m.uci() for m in runner.tracker.board.move_stack] == ["d2d4"], \
        "the next real move after the resting hand was not detected"


@pytest.mark.parametrize("seed", [1, 2])
def test_random_game_with_captures(scene: Scene, seed: int) -> None:
    """Random games with captures, to test moves that are not in the lessons."""
    game_rng = np.random.default_rng(seed)
    runner = Runner(scene, seed=100 + seed)
    board = chess.Board()
    start_game(runner, board)
    played, captures = [], 0
    for _ in range(14):
        legal = list(board.legal_moves)
        captures_available = [m for m in legal if board.is_capture(m)]
        move = (captures_available[0] if captures_available and game_rng.random() < 0.7
                else legal[int(game_rng.integers(len(legal)))])
        if move.promotion and move.promotion != chess.QUEEN:
            move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
        captures += board.is_capture(move)
        played.append(move.uci())
        play_move(runner, board, move)
    found = [m.uci() for m in runner.tracker.board.move_stack]
    assert found == played, (
        f"random game (seed {seed}, {captures} captures): the detector lost track\n"
        f"         expected {played}\n         found    {found}")


def test_load_corners_script_relative(scene: Scene) -> None:
    """Loading the corners from a script-relative path works."""
    photo = cv2.imread(str(BOARD_PHOTO))
    if photo is None:
        pytest.fail(f"I could not open the board photo at {BOARD_PHOTO}.")
    corners = load_corners(HERE / "setup_output" / "corners.json", photo.shape)
    assert corners is not None and len(corners) == 4, \
        "the loaded corners do not form a quadrilateral"
    for x, y in corners:
        assert 0 <= x < photo.shape[1] and 0 <= y < photo.shape[0], \
            "corner coordinates are out of bounds"


def test_missing_calibration_file(scene: Scene) -> None:
    """Starting the tracker without a calibration file works, and no moves are detected."""
    runner = Runner(scene, seed=17)
    board = chess.Board()
    pieces = piece_colours(board)
    runner.still(pieces)
    runner.tracker.start(runner.feed(pieces), chess.Board())
    assert len(runner.tracker.board.move_stack) == 0, "no moves should be detected without a calibration file"


def test_default_corners_load_relative_to_project(tmp_path: Path) -> None:
    """The detector default works even when launched outside mainChess."""
    photo = cv2.imread(str(BOARD_PHOTO))
    if photo is None:
        pytest.fail(f"I could not open the board photo at {BOARD_PHOTO}.")
    corners = load_corners(HERE / "setup_output" / "corners.json", photo.shape)
    assert corners.shape == (4, 2)


def test_missing_corners_has_clear_error(tmp_path: Path) -> None:
    """A missing corner file raises an actionable boundary error."""
    missing = tmp_path / "missing-corners.json"
    with pytest.raises(SystemExit, match="Run board_setup.py"):
        load_corners(missing, (600, 800, 3))
