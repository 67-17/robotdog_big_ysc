from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class ApproachConfig:
    target_distance_m: float = 0.28
    stop_latch_min_m: float = 0.26
    stop_latch_max_m: float = 0.30
    stop_confirm_max_m: float = 0.31
    resume_distance_m: float = 0.32
    far_threshold_m: float = 0.45
    near_threshold_m: float = 0.35
    far_speed_mps: float = 0.08
    middle_speed_mps: float = 0.05
    near_speed_mps: float = 0.05
    middle_pulse_s: float = 1.00
    near_pulse_s: float = 0.70
    settle_s: float = 0.00
    required_stop_samples: int = 5
    required_resume_samples: int = 3
    min_sensor_m: float = 0.03
    max_sensor_m: float = 4.50
    max_sample_age_s: float = 0.50
    max_runtime_s: float = 60.0


@dataclass(frozen=True)
class ApproachDecision:
    vx: float
    mode: str
    drive_duration_s: Optional[float]
    settle_duration_s: float
    reached: bool


class BoxApproachController:
    def __init__(self, config: ApproachConfig):
        self.config = config
        self._stop_latched = False
        self._stop_samples: Deque[float] = deque(
            maxlen=max(1, int(config.required_stop_samples))
        )
        self._resume_samples = 0

    def decide(self, distance_m: float) -> ApproachDecision:
        distance_m = float(distance_m)
        self._validate_distance(distance_m)
        if (
            not self._stop_latched
            and distance_m <= self.config.stop_latch_max_m
        ):
            self._stop_latched = True
            self._stop_samples.clear()
            self._resume_samples = 0

        if self._stop_latched:
            self._stop_samples.append(distance_m)
            if distance_m > self.config.resume_distance_m:
                self._resume_samples += 1
            else:
                self._resume_samples = 0

            if self._resume_samples >= self.config.required_resume_samples:
                self._stop_latched = False
                self._stop_samples.clear()
                self._resume_samples = 0
            else:
                reached = False
                if len(self._stop_samples) >= self.config.required_stop_samples:
                    median_distance_m = statistics.median(self._stop_samples)
                    reached = (
                        self.config.stop_latch_min_m
                        <= median_distance_m
                        <= self.config.stop_confirm_max_m
                    )
                mode = "stop" if reached else "stop_latched"
                if distance_m < self.config.stop_latch_min_m:
                    mode = "stop_too_close"
                elif distance_m > self.config.resume_distance_m:
                    mode = "stop_resume_confirm"
                return ApproachDecision(
                    vx=0.0,
                    mode=mode,
                    drive_duration_s=None,
                    settle_duration_s=0.0,
                    reached=reached,
                )

        if distance_m <= self.config.stop_latch_max_m:
            return ApproachDecision(
                vx=0.0,
                mode="stop_latched",
                drive_duration_s=None,
                settle_duration_s=0.0,
                reached=False,
            )

        if distance_m > self.config.far_threshold_m:
            speed_mps = self.config.far_speed_mps
        elif distance_m > self.config.near_threshold_m:
            speed_mps = self.config.middle_speed_mps
        else:
            speed_mps = self.config.near_speed_mps
        return ApproachDecision(
            vx=speed_mps,
            mode="continuous",
            drive_duration_s=None,
            settle_duration_s=0.0,
            reached=False,
        )

    def validate_runtime(self, sample_age_s: float, elapsed_s: float) -> None:
        if sample_age_s > self.config.max_sample_age_s:
            raise TimeoutError("front ultrasound data is stale")
        if elapsed_s > self.config.max_runtime_s:
            raise TimeoutError("box approach exceeded maximum runtime")

    def _validate_distance(self, distance_m: float) -> None:
        if not math.isfinite(distance_m):
            raise ValueError("front ultrasound distance must be finite")
        if not self.config.min_sensor_m <= distance_m <= self.config.max_sensor_m:
            raise ValueError(
                "front ultrasound distance outside "
                f"[{self.config.min_sensor_m:.2f}, {self.config.max_sensor_m:.2f}] m"
            )
