"""Model asset download and verification helpers."""

import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import uuid

from .config import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL

_DOWNLOAD_LOCK = threading.Lock()


def ensure_model_asset(
    model_path: Path = DEFAULT_MODEL_PATH,
    model_url: str = DEFAULT_MODEL_URL,
    timeout_seconds: int = 60,
) -> Path:
    """Download the MediaPipe `.task` bundle locally if missing or empty."""
    target = Path(model_path)
    if target.exists() and target.stat().st_size > 0:
        return target

    with _DOWNLOAD_LOCK:
        if target.exists() and target.stat().st_size > 0:
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(
            f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )

        try:
            req = urllib.request.Request(
                model_url,
                headers={"User-Agent": "Signify-Hackathon/0.1"},
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                with tmp_path.open("wb") as out_file:
                    shutil.copyfileobj(resp, out_file)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded model file from {model_url} was empty."
                )

            # Retry briefly if Windows Defender / file indexer briefly locks the new file.
            for attempt in range(6):
                try:
                    tmp_path.replace(target)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.2)
        finally:
            tmp_path.unlink(missing_ok=True)

    return target
