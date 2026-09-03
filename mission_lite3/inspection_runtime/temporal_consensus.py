"""Confirm inspection results across a short video-frame window."""

from collections import deque


VALID_LETTERS = ("A", "B", "C", "D")
VALID_STATES = ("normal", "abnormal")


class TemporalConsensus:
    def __init__(self, window_size=5, min_votes=3, min_letter_confidence=0.70):
        if window_size < 1 or min_votes < 1 or min_votes > window_size:
            raise ValueError("require 1 <= min_votes <= window_size")
        self.window = deque(maxlen=int(window_size))
        self.min_votes = int(min_votes)
        self.min_letter_confidence = float(min_letter_confidence)

    def _observation(self, result):
        if not result or not result.get("ok", False):
            return None
        letter = result.get("letter_detection", {}).get("label")
        confidence = float(result.get("letter_detection", {}).get("confidence", 0.0))
        state = result.get("meter_detection", {}).get("state")
        if letter not in VALID_LETTERS or state not in VALID_STATES:
            return None
        if confidence < self.min_letter_confidence:
            return None
        description = result.get("meter_detection", {}).get("description", "")
        return letter, state, description

    def update(self, result):
        self.window.append(self._observation(result))
        return self.current()

    def current(self):
        valid = [observation for observation in self.window if observation is not None]
        if not valid:
            return None

        latest = self.window[-1] if self.window else None
        if latest is None:
            return None

        recent_valid = valid[-self.min_votes :]
        if len(recent_valid) < self.min_votes:
            return None

        if any(observation != latest for observation in recent_valid):
            return None

        letter, state, description = latest
        votes = sum(observation == latest for observation in valid)
        return {
            "ok": True,
            "stable": True,
            "letter": letter,
            "state": state,
            "description": description,
            "votes": votes,
            "window_size": len(self.window),
            "abnormal_areas": [letter] if state == "abnormal" else [],
            "unknown_areas": [],
        }

    def reset(self):
        self.window.clear()
