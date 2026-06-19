"""Launch the OpenTUI terminal app, replacing the old interactive CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def launch_tui() -> int:
    """Run the Bun-hosted OpenTUI app under ``tui/``.

    Returns the child process exit code, or 1 when Bun is unavailable.
    """
    root = Path(__file__).resolve().parent.parent
    bun = shutil.which("bun")
    if bun is None:
        print(
            "OpenTUI requires Bun. Install Bun from https://bun.sh, then run python main.py again."
        )
        return 1

    proc = subprocess.run(
        [bun, "run", "start"],
        cwd=root / "tui",
        env={**os.environ, "T_AIA_PYTHON": sys.executable, "T_AIA_ROOT": str(root)},
    )
    return proc.returncode
