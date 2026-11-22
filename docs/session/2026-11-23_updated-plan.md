# Apply updated Chess Buddy plan

## Date
2026-11-23

## Session goal

Update the repository's Chess Buddy agent instructions from the user supplied final Markdown.

## Docs read

- `docs/concept/project-plan.md` (checked against the update; no plan-section changes)
- `docs/session/_TEMPLATE.md`
- `docs/session/2026-11-22_github-agent-setup.md`
- `C:\Users\User\Downloads\ChessBuddy_Plan(FINAL).md`

## Files changed

- `AGENTS.md`
- `docs/session/2026-11-23_updated-plan.md`

## Decisions made and why

- Updated the existing GitHub setup branch and PR #1 instead of opening another PR, because this is a correction to the agent guidance introduced by that still-open PR.
- Kept the repository-specific note that the source DOCX is not stored in this checkout, so future agents can follow the available `docs/concept/` and `docs/session/` files.
- Preserved the new allowance for calibration images and the new clone and Git identity reminders in the supplied instructions.
- The project-plan section matched the existing `docs/concept/project-plan.md`, so that file did not need changes.

## Tests run and results

- Not run. This session only updates project instructions and documentation.
- `git diff --check` passed.

## Outcome

- Repository instructions now include the final document's additions. The concept plan remains current because its source section did not change.

## Next steps

- Commit and push the update to `chore/github-agent-setup` so PR #1 includes the corrected instructions.
