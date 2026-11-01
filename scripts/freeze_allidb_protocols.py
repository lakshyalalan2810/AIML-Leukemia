"""Regenerate approved ALL-IDB eligibility and split artifacts with the locked environment."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.allidb_protocol import freeze_protocols


if __name__ == "__main__":
    for output in freeze_protocols(repo_root):
        print(output.relative_to(repo_root))
