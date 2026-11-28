import atexit
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, NoReturn

import nltk

NLTK_DATA = Path(__file__).parent.parent / "_vendor" / "nltk_data"
"""NLTK data shipped with the suite (English punkt_tab, stopwords, POS tagger)."""

G_PREPARED: list[str] = []
"""Private NLTK data directory, once prepared."""


def _refuse_download(*args: Any, **kwargs: Any) -> NoReturn:
    raise RuntimeError(
        "NLTK tried to download data at run time; the suite ships its NLTK data "
        f"in {NLTK_DATA}. Missing resource: {args!r} {kwargs!r}"
    )


def prepare_checkers() -> None:
    """Point NLTK at the shipped data and make the checkers deterministic.

    NLTK refuses data that other accounts can write, which depends on the
    checkout's umask, so the shipped data is copied into a fresh private
    temporary directory and used exclusively (no system NLTK paths). NLTK
    downloads are disabled, and `langdetect` is seeded.
    """
    if not G_PREPARED:
        private = tempfile.mkdtemp(prefix="custom-eval-nltk-")
        atexit.register(shutil.rmtree, private, ignore_errors=True)
        target = Path(private) / "nltk_data"
        shutil.copytree(NLTK_DATA, target)
        for root, dirs, files in os.walk(target):
            os.chmod(root, 0o700)
            for name in files:
                os.chmod(os.path.join(root, name), 0o600)
        G_PREPARED.append(str(target))
    nltk.data.path[:] = [G_PREPARED[0]]
    nltk.download = _refuse_download  # type: ignore[assignment]

    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
