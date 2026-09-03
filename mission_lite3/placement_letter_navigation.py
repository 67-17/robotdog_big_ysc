from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


LETTERS = ("A", "B", "C", "D")
MOTION_MEASUREMENT_TOLERANCE_M = 0.03


class ActionKind(str, Enum):
    FORWARD = "forward"
    FINAL_APPROACH = "final_approach"
    STRAFE = "strafe"
    RETRY = "retry"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass(frozen=True)
class LetterCandidate:
    letter: str
    center_x_px: float
    confidence: float


@dataclass(frozen=True)
class NavigationObservation:
    frame_sequence: int
    frame_width: int
    candidates: tuple[LetterCandidate, ...]
    front_distance_m: float
    elapsed_s: float


@dataclass(frozen=True)
class NavigationAction:
    kind: ActionKind
    reason: str
    distance_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    centering: bool = False


@dataclass(frozen=True)
class PlacementLetterNavigationConfig:
    letter_order: tuple[str, ...] = LETTERS
    min_confidence: float = 0.60
    forward_speed_mps: float = 0.08
    lateral_speed_mps: float = 0.08
    front_stop_distance_m: float = 0.40
    forward_budget_m: float = 1.80
    forward_step_m: float = 0.10
    lateral_search_step_m: float = 0.20
    min_center_correction_m: float = 0.02
    max_center_correction_m: float = 0.08
    center_gain_m_per_fraction: float = 0.45
    max_lateral_search_m: float = 1.05
    bilateral_search_enabled: bool = False
    lateral_search_each_side_m: float = 1.00
    immediate_complete_on_target_detection: bool = False
    acquisition_center_band: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    center_tolerance_fraction: float = 0.05
    final_approach_distance_m: float = 0.0
    final_approach_step_m: float = 0.0
    letter_spacing_m: float = 0.50
    max_anchor_jump_m: float = 0.20
    target_vote_window: int = 3
    target_min_votes: int = 2
    target_memory_max_misses: int = 6
    target_memory_max_lateral_m: float = 0.15
    target_memory_max_forward_m: float = 0.40
    target_memory_fraction_per_m: float = 0.50
    required_center_frames: int = 5
    capture_retries: int = 3
    strafe_min_progress_m: float = 0.01
    strafe_zero_progress_reverse_count: int = 2
    image_timeout_s: float = 0.50
    # Kept for configuration compatibility. Zero disables the legacy mission
    # timeout; placement now ends only after the target is centered and placed.
    total_timeout_s: float = 0.0

    def __post_init__(self) -> None:
        if tuple(self.letter_order) != LETTERS:
            raise ValueError("letter_order must be A, B, C, D from left to right")
        positive_fields = (
            "forward_speed_mps",
            "lateral_speed_mps",
            "front_stop_distance_m",
            "forward_budget_m",
            "forward_step_m",
            "lateral_search_step_m",
            "min_center_correction_m",
            "max_center_correction_m",
            "center_gain_m_per_fraction",
            "max_lateral_search_m",
            "lateral_search_each_side_m",
            "letter_spacing_m",
            "max_anchor_jump_m",
            "target_memory_max_lateral_m",
            "target_memory_max_forward_m",
            "target_memory_fraction_per_m",
            "strafe_min_progress_m",
            "image_timeout_s",
        )
        for name in positive_fields:
            value = _finite_number(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        total_timeout_s = _finite_number("total_timeout_s", self.total_timeout_s)
        if total_timeout_s < 0.0:
            raise ValueError("total_timeout_s must be non-negative")

        if not 0.28 <= self.front_stop_distance_m <= 4.50:
            raise ValueError("front_stop_distance_m must be between 0.28 and 4.50")
        if self.min_center_correction_m > self.max_center_correction_m:
            raise ValueError("center correction range is inverted")
        for name, maximum in (
            ("min_confidence", 1.0),
            ("center_tolerance_fraction", 0.50),
        ):
            value = _finite_number(name, getattr(self, name))
            if not 0.0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0.0 and {maximum}")
        for name in (
            "required_center_frames",
            "capture_retries",
            "target_vote_window",
            "target_min_votes",
            "target_memory_max_misses",
            "strafe_zero_progress_reverse_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_min_votes > self.target_vote_window:
            raise ValueError("target_min_votes must be <= target_vote_window")
        if self.target_min_votes * 2 <= self.target_vote_window:
            raise ValueError("target_min_votes must form a strict majority")
        if self.final_approach_distance_m != 0.0:
            raise ValueError("final_approach_distance_m must be zero")
        if self.final_approach_step_m != 0.0:
            raise ValueError("final_approach_step_m must be zero")
        for name in (
            "bilateral_search_enabled",
            "immediate_complete_on_target_detection",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            not isinstance(self.acquisition_center_band, (list, tuple))
            or len(self.acquisition_center_band) != 2
        ):
            raise ValueError("acquisition_center_band must contain two values")
        center_left, center_right = (
            _finite_number(
                f"acquisition_center_band[{index}]",
                value,
            )
            for index, value in enumerate(self.acquisition_center_band)
        )
        if not 0.0 <= center_left < center_right <= 1.0:
            raise ValueError(
                "acquisition_center_band must satisfy 0 <= left < right <= 1"
            )
        if (
            self.bilateral_search_enabled
            and self.max_lateral_search_m
            < 3.0 * self.lateral_search_each_side_m
        ):
            raise ValueError(
                "max_lateral_search_m must cover left search and full right sweep"
            )


def _finite_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _add_distance(current_m: float, increment_m: float) -> float:
    return float(Decimal(str(current_m)) + Decimal(str(increment_m)))


def _remaining_budget(current_m: float, budget_m: float) -> float:
    return float(Decimal(str(budget_m)) - Decimal(str(current_m)))


def placement_strafe_completion_tolerance(
    requested_distance_m: float,
    configured_tolerance_m: float,
    minimum_progress_m: float,
) -> float:
    requested = abs(_finite_number("requested_distance_m", requested_distance_m))
    configured_tolerance = _finite_number(
        "configured_tolerance_m",
        configured_tolerance_m,
    )
    minimum_progress = _finite_number(
        "minimum_progress_m",
        minimum_progress_m,
    )
    if configured_tolerance < 0.0 or minimum_progress <= 0.0:
        raise ValueError("strafe tolerance must be non-negative and progress positive")
    return min(
        configured_tolerance,
        max(0.0, requested - minimum_progress),
    )


class PlacementLetterNavigator:
    def __init__(
        self,
        target_letter: str,
        config: PlacementLetterNavigationConfig,
        *,
        preferred_target_lateral_m: float | None = None,
        final_approach_completed: bool = False,
        final_approach_progress_m: float = 0.0,
    ) -> None:
        if target_letter not in config.letter_order:
            raise ValueError(f"invalid placement target: {target_letter!r}")
        self.target_letter = target_letter
        self.config = config
        self.last_frame_sequence: int | None = None
        self.centered_frames = 0
        self.capture_misses = 0
        self.saw_any_letter = False
        self.target_locked = False
        self.last_lateral_sign: int | None = None
        self.last_geometry_lateral_sign: int | None = None
        self.pending_recovery_lateral_sign: int | None = None
        self.zero_progress_strafe_count = 0
        self.forward_travel_m = 0.0
        self.lateral_travel_m = 0.0
        self.net_lateral_m = 0.0
        self.bilateral_search_phase = "left"
        if preferred_target_lateral_m is not None and not math.isfinite(
            preferred_target_lateral_m
        ):
            raise ValueError("preferred target lateral position must be finite")
        if not isinstance(final_approach_completed, bool):
            raise ValueError("final approach checkpoint must be a boolean")
        self.preferred_target_lateral_m = preferred_target_lateral_m
        self.preferred_geometry_exhausted = preferred_target_lateral_m is None
        final_approach_progress_m = _finite_number(
            "final_approach_progress_m",
            final_approach_progress_m,
        )
        if final_approach_progress_m != 0.0 or final_approach_completed:
            raise ValueError("final approach is disabled for placement")
        self.final_approach_requested = False
        self.final_approach_completed = False
        self.final_approach_travel_m = 0.0
        self.target_observations = deque(maxlen=config.target_vote_window)
        self.last_target_center_fraction: float | None = None
        self.last_target_forward_m = 0.0
        self.last_target_lateral_m = 0.0
        self.degraded_completion_reason: str | None = None

    def request_lateral_recovery(self, action: NavigationAction) -> None:
        if action.kind != ActionKind.STRAFE or action.distance_m == 0.0:
            return
        requested_sign = 1 if action.distance_m > 0.0 else -1
        recovery_sign = -requested_sign
        self.pending_recovery_lateral_sign = recovery_sign
        self.last_lateral_sign = recovery_sign
        if self.last_geometry_lateral_sign is not None:
            self.last_geometry_lateral_sign = recovery_sign
        else:
            self.bilateral_search_phase = (
                "left" if recovery_sign > 0 else "right"
            )
        self.zero_progress_strafe_count = 0

    def _retry(self, reason: str) -> NavigationAction:
        return NavigationAction(ActionKind.RETRY, reason)

    def _fail(self, reason: str) -> NavigationAction:
        return NavigationAction(ActionKind.FAIL, reason)

    def _complete_degraded(self, reason: str) -> NavigationAction:
        self.degraded_completion_reason = reason
        return NavigationAction(ActionKind.FAIL, reason)

    def _forward(self, reason: str) -> NavigationAction:
        remaining = _remaining_budget(
            self.forward_travel_m,
            self.config.forward_budget_m,
        )
        if remaining <= 1e-9:
            return self._complete_degraded(
                "degraded_forward_budget_exhausted"
            )
        distance = min(self.config.forward_step_m, remaining)
        return NavigationAction(
            ActionKind.FORWARD,
            reason,
            distance_m=distance,
            vx_mps=self.config.forward_speed_mps,
        )

    def _strafe(
        self,
        sign: int,
        reason: str,
        *,
        centering: bool = False,
        maximum_distance_m: float | None = None,
        requested_distance_m: float | None = None,
        geometry_derived: bool = False,
    ) -> NavigationAction:
        limit = (
            self.config.max_center_correction_m
            if centering
            else self.config.lateral_search_step_m
        )
        if requested_distance_m is not None and not centering:
            limit = max(0.0, float(requested_distance_m))
        if maximum_distance_m is not None:
            limit = min(limit, max(0.0, float(maximum_distance_m)))
        if requested_distance_m is not None and centering:
            limit = min(limit, max(0.0, float(requested_distance_m)))
        distance = sign * limit
        self.last_lateral_sign = sign
        if geometry_derived:
            self.last_geometry_lateral_sign = sign
        return NavigationAction(
            ActionKind.STRAFE,
            reason,
            distance_m=distance,
            vy_mps=sign * self.config.lateral_speed_mps,
            centering=centering,
        )

    def _bilateral_search(self) -> NavigationAction:
        each_side_m = self.config.lateral_search_each_side_m
        while True:
            if self.bilateral_search_phase == "left":
                remaining_left_m = each_side_m - self.net_lateral_m
                if remaining_left_m > 1e-9:
                    return self._strafe(
                        1,
                        "search_left_for_target_letter",
                        maximum_distance_m=remaining_left_m,
                    )
                self.bilateral_search_phase = "right"
                continue

            remaining_right_m = self.net_lateral_m + each_side_m
            if remaining_right_m > 1e-9:
                return self._strafe(
                    -1,
                    "search_right_for_target_letter",
                    maximum_distance_m=remaining_right_m,
                )
            self.bilateral_search_phase = "left"

    def _preferred_geometry_search(self) -> NavigationAction | None:
        if self.preferred_geometry_exhausted:
            return None
        assert self.preferred_target_lateral_m is not None
        remaining_m = self.preferred_target_lateral_m - self.net_lateral_m
        if abs(remaining_m) <= self.config.lateral_search_step_m / 2.0:
            self.preferred_geometry_exhausted = True
            return None
        return self._strafe(
            1 if remaining_m > 0.0 else -1,
            "cached_target_geometry",
            requested_distance_m=min(
                abs(remaining_m),
                self.config.max_anchor_jump_m,
            ),
        )

    def _anchor_geometry_search(
        self,
        visible: tuple[LetterCandidate, ...],
        frame_width: int,
    ) -> NavigationAction:
        anchor = min(
            visible,
            key=lambda item: abs(item.center_x_px - frame_width / 2.0),
        )
        target_index = self.config.letter_order.index(self.target_letter)
        anchor_index = self.config.letter_order.index(anchor.letter)
        box_delta = target_index - anchor_index
        sign = -1 if box_delta > 0 else 1
        requested_m = min(
            self.config.lateral_search_step_m,
            self.config.max_anchor_jump_m,
        )
        self.preferred_geometry_exhausted = True
        return self._strafe(
            sign,
            f"anchor={anchor.letter};box_delta={box_delta}",
            requested_distance_m=requested_m,
            geometry_derived=True,
        )

    def _remember_target(self, center_fraction: float) -> None:
        self.preferred_geometry_exhausted = True
        self.last_target_center_fraction = center_fraction
        self.last_target_forward_m = self.forward_travel_m
        self.last_target_lateral_m = self.net_lateral_m
        self.capture_misses = 0

    def _remembered_target_fraction(self) -> float | None:
        if self.last_target_center_fraction is None:
            return None
        lateral_delta = self.net_lateral_m - self.last_target_lateral_m
        forward_delta = self.forward_travel_m - self.last_target_forward_m
        if abs(lateral_delta) > self.config.target_memory_max_lateral_m:
            return None
        if abs(forward_delta) > self.config.target_memory_max_forward_m:
            return None
        if self.capture_misses > self.config.target_memory_max_misses:
            return None
        predicted = (
            self.last_target_center_fraction
            + lateral_delta * self.config.target_memory_fraction_per_m
        )
        return max(0.0, min(1.0, predicted))

    def _center_target(
        self,
        center_fraction: float,
        *,
        observed_now: bool,
    ) -> NavigationAction:
        error_fraction = center_fraction - 0.5
        target_is_centered = (
            abs(error_fraction) <= self.config.center_tolerance_fraction
        )
        confirmation_reason = "confirm_target_in_fine_center_band"

        if target_is_centered:
            if not observed_now:
                return self._retry("hold_center_from_target_memory")
            if self.centered_frames < self.config.required_center_frames:
                return self._retry(confirmation_reason)
            return NavigationAction(
                ActionKind.COMPLETE,
                "target_finely_centered",
            )

        self.centered_frames = 0
        requested_m = max(
            self.config.min_center_correction_m,
            abs(error_fraction) * self.config.center_gain_m_per_fraction,
        )
        if self.zero_progress_strafe_count > 0:
            requested_m = max(
                requested_m,
                min(
                    self.config.max_center_correction_m,
                    self.config.min_center_correction_m
                    * (self.zero_progress_strafe_count + 1),
                ),
            )
        reason = (
            f"center_target={self.target_letter}"
            if observed_now
            else f"center_target_memory={self.target_letter}"
        )
        return self._strafe(
            1 if error_fraction < 0.0 else -1,
            reason,
            centering=True,
            requested_distance_m=requested_m,
            geometry_derived=True,
        )

    def decide(self, observation: NavigationObservation) -> NavigationAction:
        if not math.isfinite(observation.elapsed_s) or observation.elapsed_s < 0.0:
            return self._retry("invalid_elapsed_time")
        if (
            not math.isfinite(observation.front_distance_m)
            or observation.front_distance_m <= 0.0
        ):
            return self._retry("invalid_front_ultrasound")
        if (
            isinstance(observation.frame_sequence, bool)
            or not isinstance(observation.frame_sequence, int)
            or observation.frame_sequence < 0
        ):
            return self._retry("invalid_frame_sequence")
        if (
            isinstance(observation.frame_width, bool)
            or not isinstance(observation.frame_width, int)
            or observation.frame_width <= 0
        ):
            return self._retry("invalid_frame_width")
        if self.last_frame_sequence is not None:
            if observation.frame_sequence == self.last_frame_sequence:
                return self._retry("duplicate_frame")
            if observation.frame_sequence < self.last_frame_sequence:
                return self._retry("stale_frame")
        self.last_frame_sequence = observation.frame_sequence

        visible = tuple(
            candidate
            for candidate in observation.candidates
            if candidate.letter in self.config.letter_order
            and candidate.confidence >= self.config.min_confidence
            and math.isfinite(candidate.confidence)
            and math.isfinite(candidate.center_x_px)
        )
        targets = tuple(
            candidate
            for candidate in visible
            if candidate.letter == self.target_letter
        )
        target = max(
            targets,
            key=lambda candidate: (
                candidate.confidence,
                -abs(candidate.center_x_px - observation.frame_width / 2.0),
            ),
            default=None,
        )
        current_target_fraction = (
            None
            if target is None
            else target.center_x_px / observation.frame_width
        )
        if current_target_fraction is None:
            self.centered_frames = 0
        else:
            if (
                abs(current_target_fraction - 0.5)
                <= self.config.center_tolerance_fraction
            ):
                self.centered_frames += 1
            else:
                self.centered_frames = 0
        self.target_observations.append(current_target_fraction)
        voted_target_fractions = tuple(
            value for value in self.target_observations if value is not None
        )
        target_is_confirmed = (
            len(voted_target_fractions) >= self.config.target_min_votes
        )
        if target is not None and (
            target_is_confirmed or self.target_locked
        ):
            self.saw_any_letter = True
            if target_is_confirmed:
                self.target_locked = True
            center_fraction = float(current_target_fraction)
            self._remember_target(center_fraction)
            if self.config.immediate_complete_on_target_detection:
                return NavigationAction(
                    ActionKind.COMPLETE,
                    "target_letter_detected",
                )
            return self._center_target(
                center_fraction,
                observed_now=True,
            )

        if target is not None:
            self.saw_any_letter = True
            self.capture_misses = 0
            return self._retry("confirm_target_detection_2_of_3")

        self.capture_misses += 1
        remembered_fraction = self._remembered_target_fraction()
        if remembered_fraction is not None:
            return self._center_target(
                remembered_fraction,
                observed_now=False,
            )
        if self.target_locked:
            return_delta_m = self.last_target_lateral_m - self.net_lateral_m
            return_tolerance_m = max(
                0.01,
                self.config.min_center_correction_m / 2.0,
            )
            if abs(return_delta_m) > return_tolerance_m:
                return self._strafe(
                    1 if return_delta_m > 0.0 else -1,
                    "return_to_last_locked_target_pose",
                    centering=True,
                    requested_distance_m=abs(return_delta_m),
                )
            return self._retry("hold_last_locked_target_pose")
        if any(value is not None for value in self.target_observations):
            return self._retry("confirm_recent_target_before_anchor_fallback")
        if visible:
            self.saw_any_letter = True
            self.capture_misses = 0
            return self._anchor_geometry_search(visible, observation.frame_width)

        preferred_action = self._preferred_geometry_search()
        if preferred_action is not None:
            return preferred_action
        if self.pending_recovery_lateral_sign is not None:
            return self._strafe(
                self.pending_recovery_lateral_sign,
                "reverse_after_repeated_zero_progress",
            )
        if self.last_geometry_lateral_sign is not None:
            if self.capture_misses <= self.config.capture_retries:
                return self._retry("retry_after_geometry_letter_loss")
            return self._strafe(
                self.last_geometry_lateral_sign,
                "continue_last_geometry_direction",
                geometry_derived=True,
            )
        if self.config.bilateral_search_enabled:
            return self._bilateral_search()
        if not self.saw_any_letter:
            return self._strafe(1, "search_left_without_letter")
        if self.capture_misses <= self.config.capture_retries:
            return self._retry("retry_after_letter_loss")
        if self.last_lateral_sign is None:
            return self._bilateral_search()
        return self._strafe(
            self.last_lateral_sign,
            "continue_last_valid_direction",
        )

    def record_motion(
        self,
        action: NavigationAction,
        measured_distance_m: float,
    ) -> None:
        if not math.isfinite(measured_distance_m):
            raise ValueError("measured motion must be finite")
        if action.kind not in (
            ActionKind.FORWARD,
            ActionKind.FINAL_APPROACH,
            ActionKind.STRAFE,
        ):
            return
        if action.kind == ActionKind.FINAL_APPROACH:
            raise ValueError("final approach is disabled for placement")
        elif action.kind == ActionKind.FORWARD:
            measured_distance_m = abs(measured_distance_m)
            self.forward_travel_m = _add_distance(
                self.forward_travel_m,
                measured_distance_m,
            )
            exceeds_budget = (
                Decimal(str(self.forward_travel_m))
                > Decimal(str(self.config.forward_budget_m))
            )
        else:
            requested_sign = 1 if action.distance_m > 0.0 else -1
            if abs(measured_distance_m) < self.config.strafe_min_progress_m:
                self.zero_progress_strafe_count += 1
                if (
                    not action.centering
                    and
                    self.zero_progress_strafe_count
                    >= self.config.strafe_zero_progress_reverse_count
                ):
                    self.request_lateral_recovery(action)
            else:
                self.zero_progress_strafe_count = 0
                self.pending_recovery_lateral_sign = None
            if (
                abs(measured_distance_m) >= self.config.strafe_min_progress_m
                and action.distance_m * measured_distance_m < 0.0
            ):
                raise ValueError("strafe moved opposite requested direction")
            self.lateral_travel_m = _add_distance(
                self.lateral_travel_m,
                abs(measured_distance_m),
            )
            self.net_lateral_m = _add_distance(
                self.net_lateral_m,
                measured_distance_m,
            )
            exceeds_budget = False

        if (
            abs(measured_distance_m)
            > abs(action.distance_m) + MOTION_MEASUREMENT_TOLERANCE_M
        ):
            raise ValueError("measured motion exceeds requested distance")
        if exceeds_budget:
            budget_name = (
                "forward budget"
                if action.kind in (ActionKind.FORWARD, ActionKind.FINAL_APPROACH)
                else "lateral budget"
            )
            raise ValueError(f"measured motion exceeds {budget_name}")

    def complete_final_approach_degraded(
        self,
        measured_distance_m: float,
        reason: str,
    ) -> None:
        measured_distance_m = _finite_number(
            "measured_distance_m",
            measured_distance_m,
        )
        self.final_approach_travel_m = min(
            self.config.final_approach_distance_m,
            _add_distance(
                self.final_approach_travel_m,
                max(0.0, measured_distance_m),
            ),
        )
        self.final_approach_requested = True
        self.final_approach_completed = True
        self.degraded_completion_reason = str(reason)
