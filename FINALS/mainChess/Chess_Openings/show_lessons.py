"""
show_lessons.py - make the chess lesson JSON files easy to read.

Put this file next to your "lessons" folder, then run:

    python show_lessons.py            -> prints every lesson in the terminal
    python show_lessons.py --html     -> also saves lessons_readable.html
    python show_lessons.py --html --open   -> saves it and opens it in your browser

You can also point it at another folder or a single file:

    python show_lessons.py path/to/lessons
    python show_lessons.py lessons/opening_1_ruy_lopez.json

It only uses Python's built-in tools, so nothing needs to be installed.
"""

import argparse
import html
import json
import textwrap
import webbrowser
from pathlib import Path


# ---------- loading ----------

def find_files(target):
    """Return a sorted list of lesson files from a folder or a single file."""
    path = Path(target)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise SystemExit(f"I could not find '{target}'. Check the folder or file name.")


def load_lesson(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- small helpers ----------

def move_label(move):
    """Turn a move into '1. e4' for White or '1... e5' for Black."""
    dots = "." if move["color"] == "white" else "..."
    return f"{move['number']}{dots} {move['san']}"


def notes_for(move):
    """Short tags for special moves."""
    notes = []
    if move.get("capture"):
        notes.append("capture")
    if move.get("castling"):
        notes.append("castling")
    return notes


def who_label(move):
    return "You" if move["player"] == "child" else "Program"


# ---------- console view ----------

def print_lesson(lesson):
    title = f"{lesson['name']}  (you play {lesson['child_plays'].title()})"
    line = "=" * max(len(title), 40)
    print(f"\n{line}\n{title}\n{line}")
    print(textwrap.fill(lesson["intro"], 70))
    print()
    print(f"{'Move':<10}{'Who':<9}{'What it means'}")
    print("-" * 70)

    for move in lesson["moves"]:
        sentence = move["sentence"]
        notes = notes_for(move)
        if notes:
            sentence += "  [" + ", ".join(notes) + "]"
        wrapped = textwrap.wrap(sentence, 50) or [""]
        print(f"{move_label(move):<10}{who_label(move):<9}{wrapped[0]}")
        for extra in wrapped[1:]:
            print(f"{'':<19}{extra}")

    print("-" * 70)
    print(textwrap.fill(lesson["finish"], 70))
    total_you = sum(1 for m in lesson["moves"] if m["player"] == "child")
    print(f"({len(lesson['moves'])} moves in total, {total_you} of them are yours)")


# ---------- HTML view ----------

CSS = """
:root { --bg:#f6f7f9; --card:#ffffff; --text:#1f2430; --muted:#5f6675; --line:#e3e6ec;
        --you:#e8f1fb; --accent:#2f5d8a; --tag:#fff1db; --tagtext:#8a5a12; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --card:#1d2128; --text:#e8eaee; --muted:#9aa1ae; --line:#2b313b;
          --you:#1e2d3d; --accent:#7fb0e0; --tag:#3a2f1c; --tagtext:#f0c27a; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--text);
       font-family:"Segoe UI", Calibri, Arial, sans-serif; line-height:1.5; }
main { max-width:760px; margin:0 auto; }
h1 { font-size:1.6rem; margin:0 0 1.5rem; color:var(--accent); }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px;
        padding:1.25rem 1.4rem; margin-bottom:1.5rem; }
.card h2 { margin:0 0 .25rem; font-size:1.3rem; }
.badge { display:inline-block; font-size:.8rem; padding:.1rem .6rem; border-radius:999px;
         background:var(--accent); color:#fff; margin-bottom:.6rem; }
.intro, .finish { color:var(--muted); margin:.4rem 0 1rem; }
.finish { margin:1rem 0 0; }
.tablewrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.95rem; }
th { text-align:left; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em;
     color:var(--muted); padding:.4rem .6rem; border-bottom:2px solid var(--line); }
td { padding:.5rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
tr.you td { background:var(--you); }
td.move { font-weight:600; white-space:nowrap; font-variant-numeric:tabular-nums; }
td.who { white-space:nowrap; color:var(--muted); }
.tag { display:inline-block; font-size:.75rem; padding:0 .45rem; margin-left:.4rem;
       border-radius:6px; background:var(--tag); color:var(--tagtext); }
.legend { font-size:.85rem; color:var(--muted); margin-top:.6rem; }
"""


def lesson_to_html(lesson):
    e = html.escape
    rows = []
    for move in lesson["moves"]:
        css = "you" if move["player"] == "child" else "program"
        tags = "".join(f'<span class="tag">{e(n)}</span>' for n in notes_for(move))
        rows.append(
            f'<tr class="{css}">'
            f'<td class="move">{e(move_label(move))}</td>'
            f'<td class="who">{e(who_label(move))}</td>'
            f'<td>{e(move["sentence"])}{tags}</td>'
            f"</tr>"
        )
    total_you = sum(1 for m in lesson["moves"] if m["player"] == "child")
    return f"""
<section class="card">
  <h2>{e(lesson['name'])}</h2>
  <span class="badge">You play {e(lesson['child_plays'].title())}</span>
  <p class="intro">{e(lesson['intro'])}</p>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Move</th><th>Who</th><th>What it means</th></tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  <p class="legend">{len(lesson['moves'])} moves in total, {total_you} of them are yours (shaded rows).</p>
  <p class="finish">{e(lesson['finish'])}</p>
</section>"""


def build_html(lessons):
    body = "".join(lesson_to_html(lesson) for lesson in lessons)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess Buddy Lessons</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Chess Buddy: Opening Lessons</h1>
{body}
</main>
</body>
</html>
"""


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Show the chess lesson JSON files in a readable way.")
    parser.add_argument("target", nargs="?", default=str(Path(__file__).parent / "lessons"),
                        help="a lessons folder or one .json file (default: the 'lessons' folder next to this script)")
    parser.add_argument("--html", action="store_true", help="also save lessons_readable.html")
    parser.add_argument("--open", action="store_true", help="open the HTML page in your browser (use with --html)")
    parser.add_argument("--quiet", action="store_true", help="do not print the lessons in the terminal")
    args = parser.parse_args()

    files = find_files(args.target)
    if not files:
        raise SystemExit("No .json lesson files were found there.")
    lessons = [load_lesson(f) for f in files]

    if not args.quiet:
        for lesson in lessons:
            print_lesson(lesson)
        print()

    if args.html:
        out = Path(__file__).parent / "lessons_readable.html"
        out.write_text(build_html(lessons), encoding="utf-8")
        print(f"Saved: {out}")
        if args.open:
            webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
