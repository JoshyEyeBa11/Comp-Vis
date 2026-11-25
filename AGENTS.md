#instructions.md

## Project overview

Chess Buddy is a Python computer vision program that watches a real chess board through a camera and teaches beginners, especially kids, how to play. The camera detects the board, sees which piece moved, and checks the move against a beginner lesson. The program then gives short, friendly feedback on screen and by voice.

The vision layer is the "eyes" and a small lesson system is the "teacher." Classic OpenCV methods only. No YOLO, no trained CNN. Move legality is checked with python-chess. Read `docs/concept/project-plan.md` and the latest files in `docs/session/` before any work. The source DOCX informed the project plan, but is not stored in this repository.

## Repository

Public repo: https://github.com/JoshyEyeBa11/Comp-Vis
Branch: `main`. Working directory inside the repo: `FINAL`.

```
FINAL/
  mainChess/
    Chess_Openings/        lesson data and opening move lists
    setup_output/          board setup artifacts
    board_setup.py         board detection, perspective warp, 8x8 grid
    grab_frame.py          camera frame capture
    move_detector.py       frame differencing and move detection
    test_move_detector.py  tests for the move detector
    Board_BlackSide.jpg    calibration image
    Board_WhiteSide.jpg    calibration image
  readme.md
  docs/                    created by the agent if missing
    concept/               design and project plan docs (read first)
    session/               one log file per work session
      _TEMPLATE.md         session log template (create if missing)
```

## Project plan

This section is the plan from `Chess_Buddy_Project_Final.docx`, written for the agent. If `docs/concept/` has no project plan file yet, create `docs/concept/project-plan.md` as a copy of this section.

The author is a solo undergraduate (Computer Vision class mini project). Keep everything small, explainable, and finishable in about 4 weeks of part-time work.

### Goal and scope

Let a child learn on a real board instead of a screen. The camera watches the board, the program works out each move, and a small lesson system says if the move was right.

In scope:
- Board detection: find the board and flatten it into a top-down square.
- Move detection: find which squares changed between camera frames and match them to a legal move.
- Teaching layer: 3 beginner openings, each move checked against the lesson.
- Kid-friendly feedback: short messages, a highlighted hint square, stars, and a spoken voice.
- Hand-proof reading: wait until the hand leaves the board before reading a move.

Out of scope (do not build without explicit approval):
- A trained piece-recognition CNN. The plan lists it only as an optional upgrade if time allows. Rule: no trained detector without approval.
- Endgame lessons (only if time is left).
- Puzzle mode (see "Optional later" below).
- Support for several cameras, a moving camera, or glossy boards.

### Pipeline (13 steps)

Steps 1 to 3 happen once. Steps 4 to 13 repeat for every move.

| Step | What happens | Tool | Output | Code |
|------|--------------|------|--------|------|
| 1 | Write 3 opening lessons, one short sentence per move | JSON files | lesson files | `Chess_Openings/*.json` (done) |
| 2 | Take one photo of the empty board from the live camera | phone or webcam | reference image | `grab_frame.py` (done) |
| 3 | Find the 4 board corners (automatic, or click) and save them | OpenCV | `corners.json` | `board_setup.py` (done) |
| 4 | Read a frame from the live camera or a saved video | OpenCV | raw frame | `move_detector.py` (done) |
| 5 | Flatten the tilted board into a top-down square | perspective warp | flat board image | `board_setup.py` (done) |
| 6 | Cut the board into 64 squares named a1 to h8 | NumPy | 64 small images | `board_setup.py` (done) |
| 7 | Wait until the picture stays still, so a hand is not in the way | frame difference | a clean frame | `move_detector.py` (done) |
| 8 | Compare with the last clean frame to find squares that changed | OpenCV, NumPy | list of changed squares | `move_detector.py` (done) |
| 9 | Try each legal move and pick the one that matches the changed squares | python-chess | the move that was played | `move_detector.py` (done) |
| 10 | Compare with the lesson move | Python | right or wrong | not started |
| 11 | Right: star and voice. Wrong: hint (piece, then square, then answer) | pyttsx3, Tkinter or Pygame | feedback on screen and by voice | not started |
| 12 | If the lesson needs the other side's move, show it, tell the child which piece to move, and confirm it by camera | python-chess, OpenCV | board updated | not started |
| 13 | Go to the next move or next opening, and save the result | Python, CSV log | progress and test results | not started |

Details to handle:
- Starting setup: the child sets up the starting position by hand. The empty-board photo can be used to check which squares are occupied and to highlight a square that does not match. This checks if a square has a piece, not which piece it is.
- Special moves: captures, castling, and en passant change more than two squares, so step 9 compares every legal move instead of only looking at two squares.
- Mistakes: if no legal move matches, ask the child to put the piece back and try again.

### Design decisions already made

- White is at the bottom of the camera picture. Corners are stored in this order: a1 (bottom-left), h1, h8, a8. `board_setup.py` warns if a1 does not look like a dark square.
- `corners.json` also stores `image_size` (width, height). `move_detector.py` scales the corners if the live picture has the same shape but a different size, and stops with a clear message if the shape differs.
- Only the middle 60 percent of each square is compared. The practice mat is slightly curled, so the edges of squares are unreliable.
- A move is read only after the board has been still for about 1 second. Motion is measured on a small grey copy with brightness evened out, so auto-exposure flicker is not seen as movement.
- Each square gets a change score from colour difference and edge-density difference, whichever is larger. A score of 1.0 or more means changed.
- More than 5 changed squares means a hand or object is on the board, not a move.
- A move is accepted only if it is clearly the best legal fit (cost 0.85 or less, and at least 0.5 better than the runner-up). Promotions are treated as queen.
- Keys in the detector window: S start, R re-sync, U undo, D debug numbers, Q or ESC quit.
- Lesson files, one per opening. Top level fields: `id`, `name`, `child_plays` ("white" or "black"), `start_fen`, `intro`, `finish`, `total_moves` (counts half-moves), `moves`. Each move has: `number`, `ply`, `color`, `player` ("child" or "program"), `san`, `uci`, `from`, `to`, `capture`, `castling`, `sentence`, `fen_after`. Every lesson move was validated with python-chess.
- The 3 lessons:
  - Ruy Lopez: child plays White, 8 moves each side (16 half-moves). Includes castling for both sides.
  - Queen's Gambit (declined line): child plays White, 6 moves each side (12 half-moves). Includes Black castling.
  - French Defense: child plays Black, 8 moves each side (16 half-moves). Includes two captures. The program plays White's first move.
- Hint ladder for a wrong move: highlight the piece, then the square, then show the answer. Wording stays friendly, with no chess jargon and no scores.

Open decisions (not decided yet, note the choice in the session log):
- Screen toolkit: Tkinter or Pygame (pick one and stay with it).
- Voice: pyttsx3 (offline) or gTTS.
- Module layout for the teaching layer (steps 10 to 13).

### Camera setup

The author has no webcam and uses an Android phone with the IP Webcam app (by Pavel Khlebovich).
- Phone and computer must be on the same Wi-Fi. Start the server in the app. It shows an address like `http://192.168.1.23:8080`.
- Live stream for OpenCV: `http://PHONE_IP:8080/video`. If it does not open, try `/video.mjpg` or `/videofeed`. A single snapshot is at `/shot.jpg`.
- If frames drop, lower the resolution or frame rate in the app.
- The camera must stay fixed on a stand for every photo and video. Saved corners only work for that position.
- The empty-board photo must come from the same camera and picture size as the live stream, so take it with `grab_frame.py`.
- Board and lighting: use a matte surface on a hard flat table, tape the mat corners flat, use steady bright light, and avoid glare. White pieces on light squares and black pieces on dark squares are the hardest cases to detect.

### Phases and hours

| Phase | What to do | Hours | Output |
|-------|------------|-------|--------|
| 1. Setup | Install tools, set up the camera and board, record a few short videos once | 4 to 6 | working setup, 3 to 4 saved test videos |
| 2. Board detection | Find the board, straighten it, cut it into 64 squares | 8 to 10 | top-down board image with square grid |
| 3. Move detection | Compare frames, find changed squares, confirm with rules | 8 to 12 | program that reports each move |
| 4. Teaching layer | Add lessons, feedback, hints, voice, and stars | 6 to 10 | working lesson mode |
| 5. Testing and report | Test in different lighting, fix bugs, write the report | 4 to 6 | results table, demo video, final report |

Total: 30 to 44 hours over about 4 weeks. If time gets short, cut from 3 openings to 2 and skip extra polish.

Weekly targets:

| Week | Target | Check at the end of the week |
|------|--------|------------------------------|
| 1 | Phase 1 and most of Phase 2 | board is flattened and cut into 64 squares |
| 2 | Finish Phase 2 and do Phase 3 | program reads simple moves correctly |
| 3 | Phase 4 (teaching layer) | one full opening lesson works end to end |
| 4 | Phase 5 and polish | tests done, demo recorded, report written |

### Tools and libraries

| Tool | Use | Notes |
|------|-----|-------|
| Python 3.10+ | main language | |
| OpenCV | camera, edges, perspective warp, frame comparison | core of the vision part |
| NumPy | image and grid math | |
| python-chess | move rules, legal move check, FEN and PGN | handles all chess logic |
| pyttsx3 or gTTS | spoken feedback | pyttsx3 works offline |
| Tkinter or Pygame | screen for hints, stars, messages | pick one |
| Stockfish (optional) | rate moves for extra feedback | free chess engine |
| pytest | tests | |
| VS Code and Git | writing code, saving versions | commit after each phase |

### Risks and backup plans

| Risk | Why it matters | What to do |
|------|----------------|------------|
| Shaky board detection | glare or a moving camera breaks everything | fix the camera, use a matte board, save the corners once |
| Hands blocking the view | kids leave hands on the board | wait for the image to stay still before reading a move |
| Captures and castling | more than two squares change | use python-chess to work out which move fits the changes |
| Openings too long for kids | too many moves at once is confusing | keep each opening to the first 6 to 8 moves and use the hint steps |
| Running out of time | lesson writing takes longer than coding | cut from 3 openings to 2, skip extra polish |
| Missed moves in dim light | frame comparison is sensitive to light | steady lighting and thresholds that can be adjusted |

### How success is measured

- Move accuracy: percentage of moves read correctly across at least 10 short test games. Target 90 percent or better in steady light. Repeat in bright light, dim light, and a slightly different camera angle.
- Lesson check: a correct lesson move is marked right and a wrong one is marked wrong every time.
- Speed: a move is read within about 2 seconds after the hand leaves the board.
- Ease of use: 2 or 3 classmates or younger family members try it and the confusing spots are noted.

### Final deliverables

- Working Python program with a short README on how to run it.
- Lesson files.
- Demo video of a full lesson played on a real board.
- Final report with the method, test results, and limits.
- Optional upgrade if time allows (needs explicit approval): a small CNN that recognizes pieces, compared with the frame-differencing method.

### Optional later: puzzle mode

Removed from the core plan. Do not build it without approval. If it is added:
- Source: Lichess puzzle database (free CSV). Keep puzzles tagged mateIn3, mateIn4, mateIn5 with low ratings, about 5 per level (Level 1 = mate in 3, Level 2 = mate in 4, Level 3 = mate in 5).
- In the Lichess data the first move belongs to the computer's side. Play it first, then show the puzzle.
- Puzzles have several moves, so the program tells the child which piece to move for the computer, and the camera confirms it. On the last move, any move that gives checkmate counts as correct.
- Use the lowest-rated puzzles, the hint ladder, and optionally a few mate-in-one warm-ups, because mate in 3 to 5 is hard for young kids.

### Status (2026-11-22)

Done:
- Step 1: three lesson JSON files, validated with python-chess. `show_lessons.py` (in `Chess_Openings/`) prints them or builds `lessons_readable.html`.
- Steps 2 and 3: `grab_frame.py` and `board_setup.py`. Automatic corner detection worked on `Board_WhiteSide.jpg` and the grid check lined up. The edges are slightly wavy because the mat is curled.
- Steps 4 to 9: `move_detector.py`.
- `test_move_detector.py`: a simulation that draws fake pieces and a fake hand on the empty-board photo and plays the lessons plus random games.

Open issues (check these first):
- The simulation is not fully verified. In the last run, Queen's Gambit passed (12 of 12 moves, including Black castling), but Ruy Lopez stopped after 8 moves and did not detect White castling (e1g1). That run did not finish, so the French Defense lesson, the "piece put back", "hand resting on the board", and random game checks have no recorded result. The cause of the castling failure is not found yet.
- Nothing has been tested on real camera video yet. The thresholds (`--color 14`, `--edge 0.05`, `--motion 2.0`, `--still 1.0`) are untuned guesses for the real pieces.
- `test_move_detector.py` is a script with `main()`, not a pytest test, so `pytest mainChess/` collects nothing from it. Convert it into real tests.
- The scripts do not yet follow the coding standards below: no type hints, `print` instead of `logging`, several functions over 50 lines, and the lesson move lines are copied into the test file instead of being read from `Chess_Openings/`.

Not started: steps 10 to 13 (teaching layer), the report, the demo video, the real test videos.

## Build and test commands

- Create venv: `python -m venv venv && source venv/bin/activate`
- Install deps: `pip install opencv-python python-chess numpy pytest`
- Run tests: `pytest mainChess/`
- Run a module directly: `python mainChess/move_detector.py`
- Grab a calibration frame: `python mainChess/grab_frame.py`

Useful commands (run from `FINAL`, or drop the `mainChess/` prefix when inside `mainChess`):
- Grab the empty-board picture from the phone: `python mainChess/grab_frame.py http://PHONE_IP:8080/video --out mainChess/empty_board.jpg`
- Find corners and cut the squares: `python mainChess/board_setup.py mainChess/empty_board.jpg --out mainChess/setup_output` (add `--click` to click the corners yourself)
- Detect moves from the phone: `python mainChess/move_detector.py --source http://PHONE_IP:8080/video --corners mainChess/setup_output/corners.json`
- Detect moves from a saved video: `python mainChess/move_detector.py --source my_video.mp4 --corners mainChess/setup_output/corners.json`
- Simulation tests: `pytest mainChess/` (runs without a camera, uses `Board_WhiteSide.jpg`)
- Show the lessons: `python mainChess/Chess_Openings/show_lessons.py mainChess/Chess_Openings` (add `--html` to save a web page)

## Coding standards

- Python 3.10 or newer. Type-hint public functions. Keep functions under 50 lines.
- OpenCV only for vision work. Do not add YOLO or any trained detector without explicit approval.
- Use `logging`, not `print`, in library modules.
- Validate camera input and board corners at the boundary. Fail fast with a clear message.
- Keep lesson data in `Chess_Openings/` as data files, not hardcoded in Python.
- One concern per module. New tests go next to the code they cover.

## Rules and requirements (mandatory)

1. **Commit identity.** All commits and pushes use this git identity:
   - `user.name`: `JoshyEyeBa11`
   - `user.email`: `shuamararac@gmail.com`

   Set it before the first commit:
   ```
   git config user.name JoshyEyeBa11
   git config user.email shuamararac@gmail.com
   ```
   Never commit passwords, tokens, or any other secret.

2. **Commit messages must sound human.** Titles and descriptions must read like a person wrote them. No em dashes. No emojis. Keep them concise.
   - Good: `fix board corner detection in low light`
   - Bad: `Refactor: comprehensively enhance board detection pipeline for robustness` (too verbose, AI-sounding)

3. **Log every session in docs/session.** Before finishing a session, create a new file under `docs/session/` named `YYYY-MM-DD_short-topic.md` by copying `docs/session/_TEMPLATE.md`. Fill every field. If the template is missing, create it first with these fields:
   - Date
   - Session goal
   - Docs read (list files from `docs/concept/` and `docs/session/` that informed this session)
   - Files changed
   - Decisions made and why
   - Tests run and results
   - Outcome
   - Next steps

4. **Open a pull request every session.** When the session's work is done, push the branch and open a PR against `main` for verification. Do not merge it yourself. The PR description should summarize the session log.

5. **Stay grounded in the docs.** Before taking any action, read what is inside `docs/concept/` and the existing `docs/session/` files. You are not allowed to take uninformed actions or make decisions that divert from the concept. If a decision is not covered by the docs, stop and note it in the session log instead of guessing.

## Workflow per session

1. Read `docs/concept/` and recent `docs/session/` files.
2. Create a branch: `feat/`, `fix/`, or `chore/` plus a short topic.
3. Set the git identity (see rule 1).
4. Make changes. Run `pytest mainChess/` before committing.
5. Commit with a human-sounding, concise message. No em dashes, no emojis.
6. Fill in a new `docs/session/` log from the template.
7. Push the branch and open a PR against `main`. Do not merge.

## Do

- Read existing docs before writing any code.
- Run tests before every commit.
- Keep vision code in OpenCV only.
- Note any deviation from the concept in the session log.
- Clone the Public repo: https://github.com/JoshyEyeBa11/Comp-Vis
Branch: `main`. Working directory inside the repo: `FINAL`
- Set the git identity from rule 1 before the first commit.

## Don't

- Don't use YOLO or a trained CNN without explicit approval.
- Don't use em dashes or emojis in commit messages.
- Don't push directly to `main`.
- Don't skip the session log or the PR.
- Don't commit secrets, `.env`, or large binaries. Calibration images are an allowed exception.
