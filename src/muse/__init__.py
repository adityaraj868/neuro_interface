"""MUSE: reproducible EEG motor-imagery decoding."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _prepare_mne_runtime() -> None:
    """Give MNE a writable config location in restricted environments.

    MNE creates a small JSON preferences file during import.  The execution
    environment used for this repository can expose a read-only home
    directory, so use a temporary project-specific location only when the
    normal location cannot be created or written.
    """

    if os.environ.get("_MNE_FAKE_HOME_DIR"):
        return
    home = Path.home()
    mne_dir = home / ".mne"
    try:
        mne_dir.mkdir(exist_ok=True)
        probe = mne_dir / ".muse-write-probe"
        probe.touch(exist_ok=True)
        probe.unlink()
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "muse-mne-home"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["_MNE_FAKE_HOME_DIR"] = str(fallback)

    mpl_dir = Path(tempfile.gettempdir()) / "muse-matplotlib"
    try:
        mpl_dir.mkdir(parents=True, exist_ok=True)
        probe = mpl_dir / ".muse-write-probe"
        probe.touch(exist_ok=True)
        probe.unlink()
    except OSError:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


_prepare_mne_runtime()

__version__ = "0.1.0"

