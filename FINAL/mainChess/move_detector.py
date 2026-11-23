"""
move_detector.py - Steps 4 to 9 of the pipeline.

It watches the board through the camera and tells you which chess move was played:
  - flattens every camera picture into a top-down board (using corners.json)
  - waits until the board has LOOKED UNCHANGED for a moment (so a hand is not in
    the way, and so slow quiet moves are not missed)
  - finds the squares that changed since the last accepted picture
  - tries every legal move and picks the one that fits those squares (python-chess)

Run it directly (uses CAMERA_SOURCE defined at the top):

    python move_detector.py

Or override with a video file / other camera:

    python move_detector.py --source my_test_video.mp4

Keys in the window:
    S = pieces are set up, start watching      R = re-sync (accept what I see now as the truth)
    U = undo the last detected move            D = show / hide the debug numbers
    Q or ESC = quit

If a move is not detected, press D. Green squares are the ones the program thinks changed.
Change --color or --edge (smaller = more sensitive) until only the squares that really
changed turn green. --window is how different a square must look for the program to
count the board as "changed" while it waits for stillness.

Needs:  pip install opencv-python numpy chess
Keep board_setup.py in the same folder, because this file uses it.
"""

import argparse
import json
import logging
import time

import chess
import cv2
import numpy as np

from board_setup import flatten_board, SQUARE, FILES

logger = logging.getLogger(__name__)

CAMERA_SOURCE = "http://192.168.1.9:8080/video"

CORNER_NAMES = ["a1", "h1", "h8", "a8"]
MARGIN = int(SQUARE * 0.2)          # only the middle 60% of each square is compared
ALL_SQUARES = [f"{f}{r}" for f in FILES for r in range(1, 9)]


# ---------- small helpers ----------

def square_cell(name: str) -> tuple[int, int, int, int]:
    """Pixel box (y0, y1, x0, x1) of the middle part of a square on the flat board."""
    col = FILES.index(name[0])
    row = 8 - int(name[1])
    return (row * SQUARE + MARGIN, (row + 1) * SQUARE - MARGIN,
            col * SQUARE + MARGIN, (col + 1) * SQUARE - MARGIN)


def small_gray(flat: np.ndarray) -> np.ndarray:
    """A small, blurry grey copy of the board. Used only to tell if something is moving.
    Its brightness is evened out, so the camera's auto-exposure does not look like movement."""
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    return gray * (128.0 / max(float(gray.mean()), 1.0))      # cancels brightness flicker from the camera


def max_cell_diff(a: np.ndarray, b: np.ndarray) -> float:
    """How different are two small board pictures, per 8x8 block of squares?

    Returns the largest block difference. The camera noise alone gives about 1,
    a moved piece gives about 13 or more, so 4 is a safe "something changed" line."""
    block_h, block_w = a.shape[0] // 8, a.shape[1] // 8
    diff = np.abs(a - b).reshape(8, block_h, 8, block_w).mean(axis=(1, 3))
    return float(diff.max())


def match_light(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Match the brightness of the new picture to the old one, so a lamp or auto-exposure
    change is not mistaken for a move. Uses the middle value of each colour."""
    out = cur.astype(np.float32)
    for c in range(3):
        m = np.median(out[:, :, c])
        if m > 1:
            out[:, :, c] *= np.clip(np.median(ref[:, :, c]) / m, 0.7, 1.3)
    return np.clip(out, 0, 255)


def expected_squares(board: chess.Board, move: chess.Move) -> set[str]:
    """Which squares should look different after this move?"""
    squares = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            squares |= {chess.square(7, rank), chess.square(5, rank)}     # king side rook
        else:
            squares |= {chess.square(0, rank), chess.square(3, rank)}     # queen side rook
    if board.is_en_passant(move):
        squares.add(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    return {chess.square_name(s) for s in squares}


def match_move(board: chess.Board, scores: dict[str, float]) -> tuple[chess.Move, float, float] | None:
    """Pick the legal move that best fits the changed squares.
    Returns (move, cost, margin) or None if nothing fits well enough.
    cost   = how badly the best move fits (0 is perfect)
    margin = how much better it is than the next best move"""
    ranked = []
    for move in board.legal_moves:
        if move.promotion and move.promotion != chess.QUEEN:
            continue                                  # promotions look the same, so only try the queen
        expected = expected_squares(board, move)
        cost = 0.0
        for name in ALL_SQUARES:
            v = scores[name]
            if name in expected:
                cost += 1.0 if v < 0.5 else (0.4 if v < 1.0 else 0.0)    # a square that should change, but did not
            elif v >= 1.0:
                cost += 1.0                                              # a square that changed, but should not have
        ranked.append((cost, move))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    best_cost, best_move = ranked[0]
    margin = (ranked[1][0] - best_cost) if len(ranked) > 1 else 9.0
    if best_cost <= 0.85 and margin >= 0.5:
        return best_move, best_cost, margin
    return None


# ---------- the tracker ----------

class MoveTracker:
    """Keeps the game board and the 'last accepted picture', and looks for the next move."""

    def __init__(self, still_seconds: float = 1.0, motion_limit: float = 2.0,
                 color_limit: float = 14.0, edge_limit: float = 0.05,
                 max_changed: int = 5, window_limit: float = 4.0):
        self.still_seconds = still_seconds     # how long the board must look the same before I look
        self.motion_limit = motion_limit       # bigger number = ignores more movement
        self.color_limit = color_limit         # smaller number = more sensitive to colour changes
        self.edge_limit = edge_limit           # smaller number = more sensitive to new shapes
        self.max_changed = max_changed         # more changed squares than this means a hand, not a move
        self.window_limit = window_limit       # how different a square must look to count as "changed"
        self.board = chess.Board()
        self.ref = None
        self.history = []
        self.scores: dict[str, float] = {}
        self.status = "Press S when the pieces are set up"
        self.prev_gray: np.ndarray | None = None
        self.window_gray: np.ndarray | None = None   # first picture of the current quiet moment
        self.window_start: float | None = None       # when that quiet moment began
        self.evaluated = True

    # --- picture bookkeeping
    @staticmethod
    def _prepare(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blur = cv2.GaussianBlur(flat, (7, 7), 0)
        edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), 40, 120)
        return blur, edges

    def _reset_motion(self) -> None:
        self.prev_gray, self.window_gray = None, None
        self.window_start, self.evaluated = None, True
        self.scores = {}

    def _begin_window(self, gray: np.ndarray, now: float) -> None:
        """Start counting a new quiet moment from this picture."""
        self.window_gray, self.window_start, self.evaluated = gray, now, False

    def start(self, flat: np.ndarray, board: chess.Board | None = None) -> None:
        self.board = board or chess.Board()
        self.history = []
        self.ref = self._prepare(flat)
        self._reset_motion()
        self.status = "Watching"

    def resync(self, flat: np.ndarray) -> None:
        self.ref = self._prepare(flat)
        self._reset_motion()
        self.status = "Re-synced: I accept the board as it looks now"

    def undo(self) -> bool:
        if not self.history:
            return False
        self.board.pop()
        self.ref = self.history.pop()
        self._reset_motion()
        self.status = "Undid the last move"
        return True

    # --- the main step: call once for every camera picture
    def update(self, flat: np.ndarray, now: float) -> dict | None:
        if self.ref is None:
            return None
        gray = small_gray(flat)
        if self.prev_gray is None or self.window_gray is None:
            self.prev_gray = gray
            self._begin_window(gray, now)
            return None
        motion = float(np.abs(gray - self.prev_gray).mean())
        drift = max_cell_diff(gray, self.window_gray)
        self.prev_gray = gray

        if motion > self.motion_limit:                    # something is moving fast (a hand, probably)
            self._begin_window(gray, now)
            self.status = "Movement seen, waiting for the board to be still"
            return None
        if drift > self.window_limit:                     # the board quietly changed (a slow move)
            self._begin_window(gray, now)
            self.status = "The board changed, waiting for it to be still"
            return None
        if self.evaluated or now - self.window_start < self.still_seconds:
            return None

        self.evaluated = True                             # look only once for each quiet moment
        return self._evaluate(flat)

    def _score_squares(self, flat: np.ndarray) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:
        blur, edges = self._prepare(flat)
        ref_blur, ref_edges = self.ref
        cur = match_light(ref_blur, blur)
        diff = np.abs(ref_blur.astype(np.float32) - cur).mean(axis=2)
        scores = {}
        for name in ALL_SQUARES:
            y0, y1, x0, x1 = square_cell(name)
            color_change = diff[y0:y1, x0:x1].mean()
            edge_change = abs(edges[y0:y1, x0:x1].mean() / 255.0 - ref_edges[y0:y1, x0:x1].mean() / 255.0)
            scores[name] = max(color_change / self.color_limit, edge_change / self.edge_limit)
        return scores, (blur, edges)

    def _evaluate(self, flat: np.ndarray) -> dict:
        self.scores, new_ref = self._score_squares(flat)
        changed = [n for n in ALL_SQUARES if self.scores[n] >= 1.0]

        if not changed:
            self.ref = new_ref                            # nothing moved; soak up slow light changes
            self.status = "Watching (no change)"
            return {"type": "nothing"}
        if len(changed) > self.max_changed:
            self.status = f"Too many squares changed ({len(changed)}): a hand on the board?"
            return {"type": "hand", "changed": changed}

        found = match_move(self.board, self.scores)
        if found is None:
            self.status = "Changes seen on " + ", ".join(changed) + " but no legal move fits"
            return {"type": "unclear", "changed": changed}

        move, cost, margin = found
        number, side = self.board.fullmove_number, ("White" if self.board.turn == chess.WHITE else "Black")
        san = self.board.san(move)
        self.history.append(self.ref)
        self.board.push(move)
        self.ref = new_ref
        self.status = f"Move detected: {san}"
        return {"type": "move", "move": move, "san": san, "uci": move.uci(), "number": number,
                "side": side, "changed": changed, "cost": cost, "margin": margin}


# ---------- camera, corners, window ----------

def normalize_source(src: int | str) -> int | str:
    """Clean up camera source strings (auto-adds http:// or /video if omitted)."""
    if isinstance(src, int):
        return src
    src = str(src).strip()
    if src.isdigit():
        return int(src)
    if not (src.startswith("http://") or src.startswith("https://") or src.startswith("rtsp://")):
        if ":" in src or src.replace(".", "").isdigit():
            src = "http://" + src
    if src.startswith("http://") or src.startswith("https://"):
        if not (src.endswith("/video") or src.endswith("/video.mjpg") or src.endswith("/videofeed")):
            src = src.rstrip("/") + "/video"
    return src


def open_source(text: int | str) -> tuple[cv2.VideoCapture, bool]:
    """Open a camera, stream, or video file. Returns (capture, is_file)."""
    source = normalize_source(text)
    is_file = not (isinstance(source, int) or (isinstance(source, str) and (source.isdigit() or source.lower().startswith(("http://", "https://", "rtsp://")))))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"I could not open '{source}'.\n"
                         "For IP Webcam: check the Wi-Fi, the address, and that the server is started.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep the delay small for live cameras
    return cap, is_file


def load_corners(path: str, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Load corners.json and fit it to the size of the live picture."""
    with open(path) as f:
        data = json.load(f)
    corners = np.array([data[n] for n in CORNER_NAMES], dtype=np.float32)
    h, w = frame_shape[:2]
    size = data.get("image_size")
    if size:
        sw, sh = size
        corners *= np.array([w / sw, h / sh], dtype=np.float32)
    else:
        # Check if coordinates exceed the live frame dimensions
        max_x = corners[:, 0].max()
        max_y = corners[:, 1].max()
        if max_x > w or max_y > h:
            logger.warning("corners (%.0fx%.0f) exceed camera frame (%dx%d). Auto-scaling from 3456x4608.",
                           max_x, max_y, w, h)
            corners *= np.array([w / 3456.0, h / 4608.0], dtype=np.float32)
        else:
            logger.info("corners.json does not say which picture size it came from. Assuming 1:1.")
    return corners


# ---------- drawing the window ----------

def draw_camera_panel(frame: np.ndarray, corners: np.ndarray, panel_h: int) -> np.ndarray:
    """The live camera picture with the board outline drawn on it."""
    scale = panel_h / max(frame.shape[0], 1)
    cam_w = int(frame.shape[1] * scale)
    cam_view = cv2.resize(frame, (cam_w, panel_h))

    scaled_corners = corners * scale
    pts = scaled_corners.astype(np.int32)
    for i in range(4):
        cv2.line(cam_view, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
    for name, pt in zip(CORNER_NAMES, pts):
        cv2.circle(cam_view, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(cam_view, name, (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.rectangle(cam_view, (0, panel_h - 30), (cam_w, panel_h), (25, 25, 25), -1)
    cv2.putText(cam_view, "LIVE CAMERA FEED", (10, panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return cam_view


def draw_board_panel(flat: np.ndarray, tracker: MoveTracker, show_debug: bool,
                     panel_h: int) -> np.ndarray:
    """The flattened top-down board with change highlights and debug scores."""
    board_view = cv2.resize(flat, (panel_h, panel_h))
    cell = panel_h // 8
    overlay = board_view.copy()
    for name, value in tracker.scores.items():
        col, row = FILES.index(name[0]), 8 - int(name[1])
        x, y = col * cell, row * cell
        if value >= 1.0:
            cv2.rectangle(overlay, (x, y), (x + cell, y + cell), (0, 200, 0), -1)
        elif show_debug and value >= 0.5:
            cv2.rectangle(overlay, (x, y), (x + cell, y + cell), (0, 200, 220), -1)
        if show_debug:
            cv2.putText(board_view, f"{value:.1f}", (x + 4, y + cell - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    board_view = cv2.addWeighted(overlay, 0.35, board_view, 0.65, 0)

    cv2.rectangle(board_view, (0, panel_h - 30), (panel_h, panel_h), (25, 25, 25), -1)
    cv2.putText(board_view, "WARPED 8x8 BOARD", (10, panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return board_view


def draw_view(frame: np.ndarray, flat: np.ndarray, tracker: MoveTracker,
              corners: np.ndarray, show_debug: bool, banner: str) -> np.ndarray:
    """Build the whole program window: banner on top, camera and board side by side."""
    panel_h = 560
    cam_view = draw_camera_panel(frame, corners, panel_h)
    board_view = draw_board_panel(flat, tracker, show_debug, panel_h)
    combined_panels = np.hstack([cam_view, board_view])

    banner_bar = np.full((55, combined_panels.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(banner_bar, banner[:80], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(banner_bar, "S start  |  R re-sync  |  U undo  |  D debug  |  Q quit", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return np.vstack([banner_bar, combined_panels])


def say(event: dict) -> None:
    """Log what happened."""
    if event["type"] == "move":
        dots = "." if event["side"] == "White" else "..."
        logger.info("%s%s %-6s (%s, %s)   changed: %s", event["number"], dots, event["san"],
                    event["side"], event["uci"], ", ".join(event["changed"]))
    elif event["type"] == "hand":
        logger.info("   ... a hand or something else is on the board, waiting")
    elif event["type"] == "unclear":
        logger.info("   ... I saw changes on %s but no legal move fits. Fix the pieces or press R.",
                    ", ".join(event["changed"]))


# ---------- command line ----------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch the board and detect the chess moves.")
    parser.add_argument("--source", default=CAMERA_SOURCE,
                        help=f"phone address, webcam number, or video file (default: {CAMERA_SOURCE})")
    parser.add_argument("--corners", default="setup_output/corners.json", help="the corners.json made by board_setup.py")
    parser.add_argument("--still", type=float, default=1.0, help="seconds the board must look unchanged (default 1.0)")
    parser.add_argument("--motion", type=float, default=2.0, help="movement limit (bigger = ignores more movement)")
    parser.add_argument("--window", type=float, default=4.0,
                        help="how different a square must look to count as changed (default 4.0)")
    parser.add_argument("--color", type=float, default=14.0, help="colour change limit (smaller = more sensitive)")
    parser.add_argument("--edge", type=float, default=0.05, help="shape change limit (smaller = more sensitive)")
    parser.add_argument("--fen", default=None, help="start from this position instead of the normal start")
    parser.add_argument("--debug", action="store_true", help="show the score of every square")
    return parser


def setup_run(args: argparse.Namespace) -> tuple[cv2.VideoCapture, bool, np.ndarray, np.ndarray, MoveTracker, chess.Board | None]:
    """Open the camera, load the corners, and build the tracker."""
    logger.info("Connecting to camera: %s", args.source)
    cap, is_file = open_source(args.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("The camera did not send a picture.")
    h, w = frame.shape[:2]
    logger.info("Camera stream resolution: %d x %d", w, h)
    corners = load_corners(args.corners, frame.shape)
    tracker = MoveTracker(args.still, args.motion, args.color, args.edge,
                          window_limit=args.window)
    start_board = chess.Board(args.fen) if args.fen else None
    return cap, is_file, frame, corners, tracker, start_board


def apply_key(key: int, tracker: MoveTracker, flat: np.ndarray,
              start_board: chess.Board | None, show_debug: bool) -> bool:
    """Handle one key press. Returns the (possibly flipped) debug setting."""
    if key in (ord("s"), ord("S")):
        tracker.start(flat, start_board)
        logger.info("Started. Waiting for the first move.")
    elif key in (ord("r"), ord("R")):
        tracker.resync(flat)
        logger.info("Re-synced.")
    elif key in (ord("u"), ord("U")):
        logger.info("Undid the last move." if tracker.undo() else "Nothing to undo.")
    elif key in (ord("d"), ord("D")):
        show_debug = not show_debug
    return show_debug


def event_loop(cap: cv2.VideoCapture, is_file: bool, tracker: MoveTracker,
               corners: np.ndarray, start_board: chess.Board | None,
               show_debug: bool, window_name: str) -> None:
    """Read frames, look for moves, and show the window until the user quits."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = 0

    if is_file:                                   # a video file starts by itself
        ok, frame = cap.read()
        if not ok:
            return
        tracker.start(flatten_board(frame, corners), start_board)
        logger.info("Video mode: starting from the starting position at the first frame.")
    else:
        logger.info("Set up the pieces, then press S in the window.")

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:
                break
            logger.info("The camera stopped sending pictures.")
            break
        flat = flatten_board(frame, corners)
        now = (frame_number / fps) if is_file else time.monotonic()
        event = tracker.update(flat, now)
        if event:
            say(event)
        view = draw_view(frame, flat, tracker, corners, show_debug, tracker.status)
        cv2.imshow(window_name, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        show_debug = apply_key(key, tracker, flat, start_board, show_debug)
        frame_number += 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    cap, is_file, frame, corners, tracker, start_board = setup_run(args)

    window_name = "Chess Buddy: move detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    event_loop(cap, is_file, tracker, corners, start_board, args.debug, window_name)

    cap.release()
    cv2.destroyAllWindows()
    moves = tracker.board.move_stack
    if moves:
        logger.info("Game so far: %s", chess.Board().variation_san(moves))


if __name__ == "__main__":
    main()
