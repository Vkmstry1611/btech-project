"""Dataset acquisition utilities for CatalogBank."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def clone_catalogbank(repo_url: str, repo_dir: Path) -> Path:
    """Clone the CatalogBank repository if it is not already present."""

    if repo_dir.exists():
        logger.info("CatalogBank repository already exists at %s", repo_dir)
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning CatalogBank from %s into %s", repo_url, repo_dir)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    return repo_dir


def pull_git_lfs(repo_dir: Path, include_patterns: list[str]) -> None:
    """Pull the requested Git LFS assets for the repository."""

    if not repo_dir.exists():
        raise FileNotFoundError(f"Repository directory does not exist: {repo_dir}")

    logger.info("Initializing Git LFS in %s", repo_dir)
    subprocess.run(["git", "lfs", "install"], cwd=repo_dir, check=True)

    include_spec = ",".join(include_patterns)
    logger.info("Pulling Git LFS files matching %s", include_spec)
    subprocess.run(["git", "lfs", "pull", "-I", include_spec], cwd=repo_dir, check=True)
