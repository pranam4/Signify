"""Temporal gesture stability filtering and cooldown debouncing."""

from collections import Counter, deque
from dataclasses import dataclass
import time


@dataclass(slots=True)
class StabilityUpdate:
    """Result of feeding one frame's top gesture into the stabilizer."""

    candidate_label: str | None
    progress: float
    locked_label: str | None
    locked_score: float


class GestureStabilizer:
    """Smooths frame-level gesture predictions with majority vote and cooldown."""

    def __init__(
        self,
        window_size: int = 6,
        min_votes: int = 4,
        min_score: float = 0.55,
        cooldown_seconds: float = 1.5,
    ) -> None:
        if window_size < 1 or min_votes < 1 or min_votes > window_size:
            raise ValueError("Require 1 <= min_votes <= window_size.")
        self.window_size = window_size
        self.min_votes = min_votes
        self.min_score = min_score
        self.cooldown_seconds = cooldown_seconds
        self._history: deque[tuple[str | None, float]] = deque(maxlen=window_size)
        self._last_locked_label: str | None = None
        self._last_locked_at: float = 0.0

    def reset(self) -> None:
        """Clear buffered frame history."""
        self._history.clear()

    def update(
        self,
        label: str | None,
        score: float,
        now: float | None = None,
    ) -> StabilityUpdate:
        """Record a frame prediction and emit a locked event once stable."""
        ts = time.monotonic() if now is None else now
        valid_label = (
            label
            if label and label != "None" and score >= self.min_score
            else None
        )
        self._history.append((valid_label, score if valid_label else 0.0))

        counts = Counter(item[0] for item in self._history if item[0] is not None)
        if not counts:
            return StabilityUpdate(
                candidate_label=None,
                progress=0.0,
                locked_label=None,
                locked_score=0.0,
            )

        top_label, top_votes = counts.most_common(1)[0]
        progress = min(1.0, top_votes / self.min_votes)

        if top_votes < self.min_votes:
            return StabilityUpdate(
                candidate_label=top_label,
                progress=progress,
                locked_label=None,
                locked_score=0.0,
            )

        avg_score = (
            sum(s for lbl, s in self._history if lbl == top_label) / top_votes
        )

        # Prevent firing repeatedly while the signer holds the same pose.
        in_cooldown = (
            top_label == self._last_locked_label
            and (ts - self._last_locked_at) < self.cooldown_seconds
        )
        if in_cooldown:
            return StabilityUpdate(
                candidate_label=top_label,
                progress=1.0,
                locked_label=None,
                locked_score=avg_score,
            )

        self._last_locked_label = top_label
        self._last_locked_at = ts
        return StabilityUpdate(
            candidate_label=top_label,
            progress=1.0,
            locked_label=top_label,
            locked_score=avg_score,
        )
