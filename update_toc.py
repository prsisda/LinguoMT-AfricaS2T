#!/usr/bin/env python3
"""
update_toc.py — Keep Table of Contents sections in markdown files up to date.

Usage:
    python update_toc.py              # update all papers/**/*.md once and exit
    python update_toc.py --watch      # watch for changes and update continuously
    python update_toc.py --watch --interval 3   # poll every 3 seconds (default: 2)
    python update_toc.py path/to/file.md        # update a single file

TOC markers — place these two comments in your markdown to opt a file in:
    <!-- TOC -->
    <!-- /TOC -->
Everything between them is replaced on each run.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Heading → anchor conversion (GitHub-flavoured Markdown rules)
# ---------------------------------------------------------------------------

_ANCHOR_STRIP = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_ANCHOR_SPACE = re.compile(r"\s+")


def _heading_to_anchor(text: str) -> str:
    """Convert a heading string to a GitHub-style anchor id."""
    text = text.lower()
    text = _ANCHOR_STRIP.sub("", text)
    text = _ANCHOR_SPACE.sub("-", text.strip())
    return text


# ---------------------------------------------------------------------------
# TOC generation
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_TOC_OPEN = "<!-- TOC -->"
_TOC_CLOSE = "<!-- /TOC -->"


def _extract_headings(lines: list[str]) -> list[tuple[int, str]]:
    """Return (level, text) for every heading outside the TOC block."""
    inside_toc = False
    inside_code = False
    headings = []
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            inside_code = not inside_code
        if inside_code:
            continue
        if stripped == _TOC_OPEN:
            inside_toc = True
            continue
        if stripped == _TOC_CLOSE:
            inside_toc = False
            continue
        if inside_toc:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))
    return headings


def _build_toc(headings: list[tuple[int, str]]) -> str:
    """Build the TOC block content (without the open/close markers)."""
    if not headings:
        return ""
    min_level = min(level for level, _ in headings)
    lines = []
    for level, text in headings:
        indent = "  " * (level - min_level)
        anchor = _heading_to_anchor(text)
        lines.append(f"{indent}- [{text}](#{anchor})")
    return "\n".join(lines)


def update_file(path: Path) -> bool:
    """
    Re-generate the TOC in *path* if it contains TOC markers.
    Returns True if the file was modified.
    """
    content = path.read_text(encoding="utf-8")
    if _TOC_OPEN not in content or _TOC_CLOSE not in content:
        return False

    lines = content.splitlines(keepends=True)
    headings = _extract_headings([l.rstrip("\n") for l in lines])
    toc_body = _build_toc(headings)

    new_toc_block = f"{_TOC_OPEN}\n{toc_body}\n{_TOC_CLOSE}"

    # Replace everything between (and including) the markers
    pattern = re.compile(
        re.escape(_TOC_OPEN) + r".*?" + re.escape(_TOC_CLOSE),
        re.DOTALL,
    )
    new_content, n_subs = pattern.subn(new_toc_block, content)
    if n_subs == 0:
        return False

    if new_content == content:
        return False

    path.write_text(new_content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

PAPERS_DIR = Path(__file__).parent / "papers"


def find_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------


def _mtimes(files: list[Path]) -> dict[Path, float]:
    mtimes = {}
    for f in files:
        try:
            mtimes[f] = f.stat().st_mtime
        except FileNotFoundError:
            pass
    return mtimes


def watch(root: Path, interval: float) -> None:
    files = find_markdown_files(root)
    known_mtimes = _mtimes(files)

    # Run once immediately on startup
    for f in files:
        changed = update_file(f)
        if changed:
            print(f"[TOC] updated  {f.relative_to(root.parent)}")

    print(f"[TOC] watching {root.relative_to(root.parent)}/**/*.md  (interval={interval}s)  Ctrl-C to stop")

    try:
        while True:
            time.sleep(interval)
            current_files = find_markdown_files(root)

            # Detect new files
            for f in current_files:
                if f not in known_mtimes:
                    print(f"[TOC] new file {f.relative_to(root.parent)}")
                    known_mtimes[f] = 0.0

            current_mtimes = _mtimes(current_files)
            for f, mtime in current_mtimes.items():
                if mtime != known_mtimes.get(f):
                    known_mtimes[f] = mtime
                    changed = update_file(f)
                    if changed:
                        print(f"[TOC] updated  {f.relative_to(root.parent)}")
    except KeyboardInterrupt:
        print("\n[TOC] stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-update <!-- TOC --> blocks in LinguoMT markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Specific markdown file(s) to update. Defaults to papers/**/*.md.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for changes and update continuously.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Poll interval in seconds when --watch is active (default: 2).",
    )
    args = parser.parse_args()

    if args.files:
        targets = [Path(f) for f in args.files]
        for t in targets:
            if not t.exists():
                print(f"[TOC] file not found: {t}", file=sys.stderr)
                continue
            changed = update_file(t)
            status = "updated" if changed else "no change"
            print(f"[TOC] {status}  {t}")
        return

    if args.watch:
        watch(PAPERS_DIR, args.interval)
    else:
        files = find_markdown_files(PAPERS_DIR)
        if not files:
            print(f"[TOC] no markdown files found under {PAPERS_DIR}")
            return
        any_changed = False
        for f in files:
            changed = update_file(f)
            if changed:
                print(f"[TOC] updated  {f.relative_to(PAPERS_DIR.parent)}")
                any_changed = True
        if not any_changed:
            print("[TOC] all TOC blocks already up to date.")


if __name__ == "__main__":
    main()
