"""
grab_frame.py - save ONE picture from your phone camera (IP Webcam), a webcam, or a video file.

Why: the empty-board photo must come from the SAME camera and the SAME picture size as
the live video, otherwise the saved corners will not fit. So take it with this script.

How to use with IP Webcam (Android):
  1. Phone and computer must be on the same Wi-Fi.
  2. Open the IP Webcam app and tap "Start server". It shows an address like http://192.168.1.23:8080
  3. Put the phone on its stand, looking down at the board, and do not move it again.
  4. Run:    python grab_frame.py
  5. Press SPACE to save the picture (default name: empty_board.jpg). ESC quits.

Other sources (if you want to override the address in code):
    python grab_frame.py 0                  (webcam)
    python grab_frame.py some_video.mp4     (a video file)

If the address does not open, try /video.mjpg or /videofeed instead of /video.
Needs:  pip install opencv-python
"""

import argparse
import cv2

# ==============================================================================
# CAMERA CONFIGURATION: Change your phone/webcam IP address here!
# (Examples: "http://192.168.1.2:8080/video", "192.168.1.2:8080", or 0 for USB)
# ==============================================================================
CAMERA_SOURCE = "http://192.168.1.4:8080/video"
# ==============================================================================


def normalize_source(src):
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


def open_source(text):
    """A number means a webcam, anything else is a web address or a file."""
    source = normalize_source(text)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"I could not open '{source}'.\n"
                         "For IP Webcam: check the Wi-Fi, the address, and that the server is started.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Save one picture from a camera or stream.")
    parser.add_argument("source", nargs="?", default=CAMERA_SOURCE,
                        help=f"web address, webcam number, or video file (default: {CAMERA_SOURCE})")
    parser.add_argument("--out", default="empty_board.jpg", help="file name to save (default: empty_board.jpg)")
    args = parser.parse_args()

    print(f"Connecting to camera: {args.source}")
    cap = open_source(args.source)
    print("Press SPACE to save the picture, ESC to quit.")
    cv2.namedWindow("grab_frame", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("The camera stopped sending pictures.")
        h, w = frame.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        shown = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.putText(shown, "SPACE = save   ESC = quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("grab_frame", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.imwrite(args.out, frame)
            print(f"Saved {args.out}  (picture size: {w} x {h})")
            break
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
