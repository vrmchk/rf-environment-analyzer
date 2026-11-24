"""Locations of datasets and experiment outputs.

Datasets and raw experiment outputs (models, logits, metrics, plots) live outside
the repository, so that the main checkout and every git worktree share one copy
and outputs survive removing a worktree. Override with the RFEA_DATA_DIR and
RFEA_OUTPUT_DIR environment variables; by default they are the `data` and
`outputs` folders next to the main checkout.
"""

import os
import subprocess
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[2]


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


def _shared_dir(env_var: str, name: str) -> Path:
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    try:
        return _main_checkout().parent / name
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"Cannot locate the {name} folder; set {env_var}") from e


def data_dir() -> Path:
    return _shared_dir("RFEA_DATA_DIR", "data")


def output_dir() -> Path:
    return _shared_dir("RFEA_OUTPUT_DIR", "outputs")


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
