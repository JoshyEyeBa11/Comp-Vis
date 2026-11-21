"""
test_move_detector.py - checks the move detector WITHOUT a camera.

It takes your empty-board photo, draws fake pieces on it, and plays whole games:
a fake hand picks up each piece, carries it, and puts it down, while the picture also gets
camera noise and small brightness changes. Then it checks that move_detector.py found
every move correctly.

Run (from the folder that has your board photo):
    python test_move_detector.py Board_WhiteSide.jpg

The fake pieces are simple drawings, so this proves the LOGIC works.
It does not prove that the settings suit your real pieces. For that, use your own video.
"""

import sys
import time

import chess
import cv2
import numpy as np

from board_setup import find_corners_auto, flatten_board, BOARD_SIZE, SQUARE, FILES
from move_detector import MoveTracker

FPS = 15
SETTLE = 17            # still frames at the end of each move (1.1 seconds)
rng = np.random.default_rng(7)

LESSONS = {
    "Ruy Lopez": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O",
    "Queen's Gambit": "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7",
    "French Defense": "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3 cxd4 Nxd4 Bc5",
}


# ---------- drawing the fake scene ----------

def centre(name):
    col, row = FILES.index(name[0]), 8 - int(name[1])
    return col * SQUARE + SQUARE // 2, row * SQUARE + SQUARE // 2


def draw_piece(img, x, y, white):
    cv2.ellipse(img, (x + 5, y + 9), (27, 19), 0, 0, 360, (45, 55, 55), -1)               # shadow
    body, line = ((225, 235, 242), (70, 80, 90)) if white else ((38, 32, 32), (150, 150, 150))
    cv2.circle(img, (x, y), 28, body, -1)
    cv2.circle(img, (x, y), 28, line, 3)
    cv2.circle(img, (x, y - 6), 12, line, 2)


def draw_hand(img, x, y, big=False):
    skin = (125, 155, 210)
    cv2.line(img, (x + 120, BOARD_SIZE + 150), (x, y), skin, 80 if big else 55)           # arm
    cv2.ellipse(img, (x, y), (70, 48) if big else (48, 34), -30, 0, 360, skin, -1)         # hand


class Scene:
    """The fake world: a flat empty board, the pieces on it, and a way to make camera pictures."""

    def __init__(self, photo):
        self.photo = photo
        self.h, self.w = photo.shape[:2]
        self.corners = find_corners_auto(photo)
        if self.corners is None:
            raise SystemExit("I could not find the board in the photo. Try a photo with the whole board in view.")
        self.flat_empty = flatten_board(photo, self.corners)
        target = np.array([[0, 799], [799, 799], [799, 0], [0, 0]], dtype=np.float32)
        self.to_photo = np.linalg.inv(cv2.getPerspectiveTransform(self.corners, target))
        self.mask = cv2.warpPerspective(np.full((800, 800), 255, np.uint8), self.to_photo, (self.w, self.h))

    def picture(self, pieces, hand=None, carried=None, big_hand=False):
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
    def __init__(self, scene):
        self.scene = scene
        self.corners = scene.corners
        self.tracker = MoveTracker()
        self.frame_no = 0
        self.events = []

    def feed(self, pieces, **kw):
        frame = self.scene.picture(pieces, **kw)
        flat = flatten_board(frame, self.corners)
        event = self.tracker.update(flat, self.frame_no / FPS)
        self.frame_no += 1
        if event:
            self.events.append(event)
        return flat

    def still(self, pieces, frames=SETTLE):
        for _ in range(frames):
            self.feed(pieces)

    def trip(self, pieces, src, dst, removed=(), settle=SETTLE):
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


def piece_colours(board):
    return {chess.square_name(s): (p.color == chess.WHITE) for s, p in board.piece_map().items()}


def play_move(runner, board, move, settle=SETTLE):
    """Show one chess move being played by the fake hand. Returns the new pieces."""
    before = piece_colours(board)
    expected = []
    from_name, to_name = chess.square_name(move.from_square), chess.square_name(move.to_square)
    castling = board.is_castling(move)
    en_passant = board.is_en_passant(move)
    rank = chess.square_rank(move.from_square)
    if castling:
        king_side = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook_from, rook_to = (chess.square(7, rank), chess.square(5, rank)) if king_side else \
                             (chess.square(0, rank), chess.square(3, rank))
    removed = [to_name] if board.is_capture(move) and not en_passant else []
    if en_passant:
        removed = [chess.square_name(chess.square(chess.square_file(move.to_square), rank))]

    if castling:
        pieces = runner.trip(before, from_name, to_name, settle=4)
        pieces = runner.trip(pieces, chess.square_name(rook_from), chess.square_name(rook_to), settle=settle)
    else:
        pieces = runner.trip(before, from_name, to_name, removed=removed, settle=settle)
    board.push(move)
    return pieces


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"   {detail}" if detail else ""))
    return ok


# ---------- the tests ----------

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python test_move_detector.py Board_WhiteSide.jpg")
    photo = cv2.imread(sys.argv[1])
    if photo is None:
        raise SystemExit(f"I could not open '{sys.argv[1]}'.")
    photo = cv2.resize(photo, None, fx=1152 / photo.shape[1], fy=1152 / photo.shape[1], interpolation=cv2.INTER_AREA)
    scene = Scene(photo)
    results = []
    started = time.time()

    # 1. the three lessons
    for name, line in LESSONS.items():
        runner = Runner(scene)
        board = chess.Board()
        pieces = piece_colours(board)
        runner.still(pieces)
        runner.tracker.start(runner.feed(pieces), chess.Board())
        expected = []
        for san in line.split():
            move = board.parse_san(san)
            expected.append(move.uci())
            pieces = play_move(runner, board, move)
        found = [m.uci() for m in runner.tracker.board.move_stack]
        results.append(check(f"Lesson: {name} ({len(expected)} moves)", found == expected,
                             "" if found == expected else f"\n         expected {expected}\n         found    {found}"))
        if name == "Ruy Lopez":
            sample_flat = None

    # 2. a piece picked up and put back on the same square is not a move
    runner = Runner(scene)
    board = chess.Board()
    pieces = piece_colours(board)
    runner.still(pieces)
    runner.tracker.start(runner.feed(pieces), chess.Board())
    runner.trip(pieces, "g1", "g1")
    results.append(check("Piece lifted and put back: no move reported", len(runner.tracker.board.move_stack) == 0))
    pieces = play_move(runner, board, board.parse_san("e4"))
    results.append(check("The next real move is still detected afterwards",
                         [m.uci() for m in runner.tracker.board.move_stack] == ["e2e4"]))

    # 3. a hand resting over the board for 3 seconds must not become a move
    runner = Runner(scene)
    board = chess.Board()
    pieces = piece_colours(board)
    runner.still(pieces)
    runner.tracker.start(runner.feed(pieces), chess.Board())
    for t in np.linspace(0, 1, 5):
        runner.feed(pieces, hand=(int(840 + (400 - 840) * t), int(940 + (400 - 940) * t)), big_hand=True)
    for _ in range(FPS * 3):
        runner.feed(pieces, hand=(400, 400), big_hand=True)
    for t in np.linspace(0, 1, 5):
        runner.feed(pieces, hand=(int(400 + 440 * t), int(400 + 540 * t)), big_hand=True)
    runner.still(pieces)
    types = [e["type"] for e in runner.events]
    results.append(check("Hand resting on the board: no move reported",
                         len(runner.tracker.board.move_stack) == 0 and ("hand" in types or "unclear" in types),
                         f"events seen: {sorted(set(types))}"))
    pieces = play_move(runner, board, board.parse_san("d4"))
    results.append(check("The next real move is still detected afterwards",
                         [m.uci() for m in runner.tracker.board.move_stack] == ["d2d4"]))

    # 4. random games with captures, to test moves that are not in the lessons
    for seed in (1, 2):
        game_rng = np.random.default_rng(seed)
        runner = Runner(scene)
        board = chess.Board()
        pieces = piece_colours(board)
        runner.still(pieces)
        runner.tracker.start(runner.feed(pieces), chess.Board())
        played, captures = [], 0
        for _ in range(14):
            legal = list(board.legal_moves)
            captures_available = [m for m in legal if board.is_capture(m)]
            move = captures_available[0] if captures_available and game_rng.random() < 0.7 else legal[int(game_rng.integers(len(legal)))]
            if move.promotion and move.promotion != chess.QUEEN:
                move = chess.Move(move.from_square, move.to_square, chess.QUEEN)
            captures += board.is_capture(move)
            played.append(move.uci())
            pieces = play_move(runner, board, move)
        found = [m.uci() for m in runner.tracker.board.move_stack]
        results.append(check(f"Random game {seed}: 14 moves, {captures} captures", found == played,
                             "" if found == played else f"\n         expected {played}\n         found    {found}"))

    print(f"\n{sum(results)} of {len(results)} checks passed  ({time.time() - started:.0f} seconds)")

    # a sample picture to look at
    runner = Runner(scene)
    board = chess.Board()
    pieces = piece_colours(board)
    x, y = centre("e2")
    cv2.imwrite("test_sample_frame.jpg", scene.picture(pieces, hand=(x + 60, y - 120), carried=True))


if __name__ == "__main__":
    main()
