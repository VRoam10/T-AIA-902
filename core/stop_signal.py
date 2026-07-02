"""Process-wide cooperative stop flag.

The TUI bridge (``core.tui_backend``) sets this when the user stops a run or
quits the app; the training / evaluation / human-play loops poll it and break at
a safe point so their ``finally`` blocks run (saving checkpoints, closing
BeamNG). Each backend command runs in its own process, so the flag always starts
clear — no reset needed between runs.
"""

from __future__ import annotations

import threading

_stop = threading.Event()


def request_stop() -> None:
    """Ask any running loop to stop at its next safe checkpoint."""
    _stop.set()


def stop_requested() -> bool:
    """True once a stop has been requested for this process."""
    return _stop.is_set()
