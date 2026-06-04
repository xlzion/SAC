#!/usr/bin/env python3
"""Fail if release-unsafe strings appear in tracked public files."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


BLOCKLIST = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"OPENAI_API_KEY",
        r"sk-[A-Za-z0-9]",
        r"api[_-]?key\s*=",
        r"password\s*=",
        r"192\.168\.",
        r"/mnt/disk/",
        r"/home/xlz/",
        r"/Users/xlz/",
        r"System Override:\s*Alpha-7",
        r'"prompt"\s*:',
        r'"response"\s*:',
        r'"audit_raw"\s*:',
    ]
]

SKIP_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ai",
    ".tar",
    ".gz",
    ".zip",
}


def tracked_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [root / line for line in proc.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    failures: list[str] = []
    for path in tracked_files(root):
        if path.relative_to(root).as_posix() == "scripts/validate_release.py":
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        for pattern in BLOCKLIST:
            if pattern.search(text):
                failures.append(f"{path.relative_to(root)}: {pattern.pattern}")

    if failures:
        print("Release-safety scan failed:")
        print("\n".join(failures))
        raise SystemExit(1)
    print("Release-safety scan passed.")


if __name__ == "__main__":
    main()
