# Session log: plan status sync

Date: 2026-11-24

## What this session did

No code changes. Checked the state of the repo after PR 3 was merged and updated `docs/concept/project-plan.md` so the Status section matches reality.

## Findings

- PR 3 (fix quiet move detection, make the simulation real pytest tests) is merged into main. main is now at the commit that includes the castling fix, the pytest conversion, and the coding-standards pass on `move_detector.py` and `board_setup.py`.
- Steps 1 to 9 of the pipeline are done. The next step is 10 (compare with the lesson move), which starts the teaching layer.
- The old Status section still listed the Ruy Lopez castling failure, the missing pytest conversion, and the missing coding standards as open issues. All three are resolved, so the section was rewritten.

## Open issues carried forward

- Nothing is tested on real camera video yet. The thresholds (`--color 14`, `--edge 0.05`, `--motion 2.0`, `--window 4.0`) are untuned guesses for the real pieces.
- Steps 10 to 13 (teaching layer), the report, the demo video, and the real test videos are not started.

## Next steps

1. Tune the detector thresholds against a real camera video, then re-run the pytest suite.
2. Start step 10: compare the detected move against the lesson move.
