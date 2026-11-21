"""
board_setup.py - Steps 2 and 3 to 6 of the pipeline.

What it does with ONE photo of the empty board:
  1. Finds the 4 corners of the board (automatically, or you click them).
  2. Saves the corners to corners.json so you never have to find them again.
  3. Flattens the tilted board into a flat top-down square.
  4. Cuts the flat board into 64 squares named a1 to h8.
  5. Saves a "grid_check.png" so you can SEE if the squares line up.

How to run:
    python board_setup.py your_empty_board_photo.jpg            (tries automatic first)
    python board_setup.py your_empty_board_photo.jpg --click    (you click the corners)

Rules for the photo:
  - White must be at the BOTTOM of the picture (a1 at the bottom-left).
  - The camera must stay in the same place for every later photo and video.

Clicking order (only with --click): a1, then h1, then h8, then a8.
That is: bottom-left, bottom-right, top-right, top-left of the playing area.
Keys in the click window: R = start over, ENTER = accept, ESC = quit.

Needs:  pip install opencv-python numpy
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BOARD_SIZE = 800            # the flat board will be 800 x 800 pixels
SQUARE = BOARD_SIZE // 8    # so every square is 100 x 100 pixels
FILES = "abcdefgh"


# ---------- finding the corners ----------

def order_corners(points):
    """Put 4 points in this order: a1 (bottom-left), h1 (bottom-right),
    h8 (top-right), a8 (top-left). Assumes White is at the bottom."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)                 # x + y
    d = np.diff(pts, axis=1).ravel()    # y - x
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]
    return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)


def find_corners_auto(img):
    """Look for the 7 x 7 inner corners of the checkerboard, then work out
    where the 4 outer corners of the playing area must be."""
    h, w = img.shape[:2]
    scale = 1400 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    found, inner = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        found, inner = cv2.findChessboardCorners(gray, (7, 7),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None

    inner = inner.reshape(-1, 2) / scale                       # back to full-size pixels
    ideal = np.array([(c, r) for r in range(1, 8) for c in range(1, 8)], dtype=np.float32)
    H, _ = cv2.findHomography(ideal, inner)                    # ideal grid -> photo
    outer = np.array([[[0, 0], [8, 0], [8, 8], [0, 8]]], dtype=np.float32)
    outer_in_photo = cv2.perspectiveTransform(outer, H)[0]
    return order_corners(outer_in_photo)


def click_corners(img):
    """Show the photo and let the user click a1, h1, h8, a8."""
    h, w = img.shape[:2]
    scale = min(1000 / w, 800 / h, 1.0)
    shown = cv2.resize(img, None, fx=scale, fy=scale)
    labels = ["a1", "h1", "h8", "a8"]
    clicks = []

    def redraw():
        canvas = shown.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(canvas, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        text = f"Click {labels[len(clicks)]}" if len(clicks) < 4 else "ENTER = accept, R = redo"
        cv2.putText(canvas, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.imshow("Click the corners", canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            redraw()

    cv2.namedWindow("Click the corners")
    cv2.setMouseCallback("Click the corners", on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("r"), ord("R")):
            clicks.clear()
            redraw()
        elif key == 13 and len(clicks) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Stopped.")
    cv2.destroyAllWindows()
    return np.array(clicks, dtype=np.float32) / scale   # back to full-size pixels


# ---------- flattening and cutting ----------

def flatten_board(img, corners):
    """Turn the tilted board into a flat top-down square."""
    target = np.array([[0, BOARD_SIZE - 1], [BOARD_SIZE - 1, BOARD_SIZE - 1],
                       [BOARD_SIZE - 1, 0], [0, 0]], dtype=np.float32)   # a1, h1, h8, a8
    matrix = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), target)
    return cv2.warpPerspective(img, matrix, (BOARD_SIZE, BOARD_SIZE))


def square_name(col, row):
    """col 0..7 goes left to right (a..h). row 0..7 goes top to bottom (rank 8..1)."""
    return f"{FILES[col]}{8 - row}"


def cut_squares(flat):
    """Return a dictionary like {'a1': image, 'a2': image, ...} with 64 squares."""
    squares = {}
    for row in range(8):
        for col in range(8):
            y, x = row * SQUARE, col * SQUARE
            squares[square_name(col, row)] = flat[y:y + SQUARE, x:x + SQUARE]
    return squares


def draw_grid(flat):
    """Draw the grid and square names on a copy of the flat board, for checking."""
    out = flat.copy()
    for i in range(9):
        cv2.line(out, (i * SQUARE, 0), (i * SQUARE, BOARD_SIZE), (0, 0, 255), 1)
        cv2.line(out, (0, i * SQUARE), (BOARD_SIZE, i * SQUARE), (0, 0, 255), 1)
    for row in range(8):
        for col in range(8):
            cv2.putText(out, square_name(col, row), (col * SQUARE + 6, row * SQUARE + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def orientation_looks_right(squares):
    """a1 should be a dark square and b1 a light one. Returns True if that is so."""
    a1 = cv2.cvtColor(squares["a1"], cv2.COLOR_BGR2GRAY).mean()
    b1 = cv2.cvtColor(squares["b1"], cv2.COLOR_BGR2GRAY).mean()
    return a1 < b1


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Find the board, flatten it, and cut it into 64 squares.")
    parser.add_argument("photo", help="photo of the EMPTY board (White at the bottom)")
    parser.add_argument("--click", action="store_true", help="click the corners yourself")
    parser.add_argument("--out", default="setup_output", help="folder for the results (default: setup_output)")
    args = parser.parse_args()

    img = cv2.imread(args.photo)
    if img is None:
        raise SystemExit(f"I could not open '{args.photo}'.")

    corners = None
    if not args.click:
        corners = find_corners_auto(img)
        print("Automatic corner search:", "worked" if corners is not None else "did not work, switching to clicking")
    if corners is None:
        corners = click_corners(img)

    out = Path(args.out)
    (out / "squares").mkdir(parents=True, exist_ok=True)

    # save the corners so later programs can reuse them
    names = ["a1", "h1", "h8", "a8"]
    h, w = img.shape[:2]
    data = {n: [float(x), float(y)] for n, (x, y) in zip(names, corners)}
    data["image_size"] = [w, h]
    with open(out / "corners.json", "w") as f:
        json.dump(data, f, indent=2)

    # flatten, cut, and save
    flat = flatten_board(img, corners)
    squares = cut_squares(flat)
    cv2.imwrite(str(out / "flat_board.png"), flat)
    cv2.imwrite(str(out / "grid_check.png"), draw_grid(flat))
    for name, crop in squares.items():
        cv2.imwrite(str(out / "squares" / f"{name}.png"), crop)

    # a picture of the original with the corners marked
    marked = img.copy()
    for n, (x, y) in zip(names, corners):
        cv2.circle(marked, (int(x), int(y)), 25, (0, 0, 255), -1)
        cv2.putText(marked, n, (int(x) + 30, int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    cv2.imwrite(str(out / "corners_marked.jpg"), marked)

    print(f"Saved results in: {out.resolve()}")
    print("  corners.json, flat_board.png, grid_check.png, corners_marked.jpg, squares/a1.png ... h8.png")
    if not orientation_looks_right(squares):
        print("WARNING: a1 should be a dark square. The board may be turned around.")
        print("         Check that White is at the bottom, or use --click and click a1 first.")


if __name__ == "__main__":
    main()
