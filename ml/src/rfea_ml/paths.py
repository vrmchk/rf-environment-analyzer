"""Locations of datasets and experiment outputs.

Datasets live outside the repository so that the main checkout and every git
worktree share one copy. Override with the RFEA_DATA_DIR environment variable;
by default it is a `data` folder next to the main checkout.
"""

import os
import subprocess
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ML_DIR / "outputs"


def _main_checkout() -> Path:
    # The common git dir is the same for the main checkout and all its worktrees.
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ML_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).parent


def data_dir() -> Path:
    env = os.environ.get("RFEA_DATA_DIR")
    if env:
        return Path(env)
    try:
        return _main_checkout().parent / "data"
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("Cannot locate the data folder; set RFEA_DATA_DIR") from e


def git_commit() -> str:
    """Short hash of the current code commit, with a '-dirty' suffix for local changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ML_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", ".", "../packages", "../pyproject.toml")
        return commit + ("-dirty" if dirty else "")
    except OSError, subprocess.CalledProcessError:
        return "unknown"
