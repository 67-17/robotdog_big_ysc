import math

from .model import Decision, bbox_center_x


def is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_wz(initial_yaw, current_yaw, config):
    speed = config["speed"]
    correction = speed["heading_kp"] * wrap_angle(
        initial_yaw - current_yaw
    )
    limit = speed["max_heading_wz"]
    return max(-limit, min(limit, correction))


def lateral_position(odom_x, odom_y, initial_yaw):
    return (
        -math.sin(initial_yaw) * odom_x
        + math.cos(initial_yaw) * odom_y
    )


def forward_progress(
    odom_x, odom_y, initial_odom_x, initial_odom_y, initial_yaw
):
    return (
        math.cos(initial_yaw) * (odom_x - initial_odom_x)
        + math.sin(initial_yaw) * (odom_y - initial_odom_y)
    )


class AvoidanceController(object):
    def __init__(self, config):
        self.config = config
        self.state = "CRUISE"
        self.avoidance_count = 0
        self.direction = None
        self.active_track_id = None
        self.confirm_track_id = None
        self.confirm_resume_state = "CRUISE"
        self.confirm_frames = 0
        self.safe_frames = 0
        self.pass_origin_odom_x = None
        self.pass_origin_odom_y = None
        self.pass_progress_m = None
        self.return_line_target_m = None
        self.return_started_at = None
        self.return_stable_frames = 0
        self.return_line_error_m = None
        self.initial_yaw = None
        self.initial_odom_x = None
        self.initial_odom_y = None
        self.forward_progress_m = None
        self.start_time = None
        self.last_time = None
        self.hold_resume_state = None
        self.hold_reason = None
        self.hold_recovery_frames = 0

    def step(self, frame):
        safety_decision = self.check_safety(frame)
        if safety_decision is not None:
            return safety_decision

        if self.state == "AVOID":
            return self._avoid(frame)

        if self.state == "PASS":
            return self._pass(frame)

        if self.state == "RETURN_LINE":
            return self._return_line(frame)

        if self.state == "FINAL_CRUISE":
            return self._final_cruise(frame)

        if self.state == "HANDOFF":
            return self._handoff(frame)

        if self.state == "CONFIRM":
            target = self._find_track(frame.tracks, self.confirm_track_id)
            return self._confirm(frame, target)

        return self._start_confirmation_or_cruise(frame)

    def check_safety(self, frame):
        if self.state == "FINISHED":
            return self._finished_decision()

        hold_reason = self._invalid_frame_reason(frame)
        if hold_reason is None:
            if self.initial_yaw is None or self.start_time is None:
                self.initial_yaw = frame.yaw
                self.initial_odom_x = frame.odom_x
                self.initial_odom_y = frame.odom_y
                self.forward_progress_m = 0.0
                self.start_time = frame.now
                self.last_time = frame.now
            elif frame.now < self.last_time:
                hold_reason = "invalid time rollback"
            else:
                self.last_time = frame.now

        if hold_reason is None:
            self.forward_progress_m = forward_progress(
                frame.odom_x,
                frame.odom_y,
                self.initial_odom_x,
                self.initial_odom_y,
                self.initial_yaw,
            )
            if (
                is_finite_number(self.pass_origin_odom_x)
                and is_finite_number(self.pass_origin_odom_y)
            ):
                self.pass_progress_m = forward_progress(
                    frame.odom_x,
                    frame.odom_y,
                    self.pass_origin_odom_x,
                    self.pass_origin_odom_y,
                    self.initial_yaw,
                )
            hold_reason = self._safety_fault_reason(frame)

        if self.state == "HOLD":
            if hold_reason is not None:
                self.hold_reason = hold_reason
                self.hold_recovery_frames = 0
                return self._hold_decision()
            self.hold_recovery_frames += 1
            if self.hold_recovery_frames < self.config["decision"]["stable_frames"]:
                return self._hold_decision(recovering=True)
            resume_state = self.hold_resume_state or "CRUISE"
            self.state = resume_state
            self.hold_resume_state = None
            self.hold_reason = None
            self.hold_recovery_frames = 0
            return None

        if hold_reason is not None:
            return self.force_hold(hold_reason)
        return None

    def force_hold(self, reason):
        if self.state != "HOLD":
            self.hold_resume_state = self.state
        self.state = "HOLD"
        self.hold_reason = str(reason)
        self.hold_recovery_frames = 0
        return self._hold_decision()

    @staticmethod
    def _invalid_frame_reason(frame):
        fields = (
            ("now", frame.now),
            ("odom_x", frame.odom_x),
            ("odom_y", frame.odom_y),
            ("yaw", frame.yaw),
            ("image_age_s", frame.image_age_s),
            ("ultrasound_age_s", frame.ultrasound_age_s),
            ("odom_age_s", frame.odom_age_s),
            ("ultrasound_m", frame.ultrasound_m),
        )
        for name, value in fields:
            if not is_finite_number(value):
                return "invalid %s" % name

        nonnegative_fields = (
            ("image_age_s", frame.image_age_s),
            ("ultrasound_age_s", frame.ultrasound_age_s),
            ("odom_age_s", frame.odom_age_s),
            ("ultrasound_m", frame.ultrasound_m),
        )
        for name, value in nonnegative_fields:
            if value < 0.0:
                return "invalid %s" % name

        return None

    def _safety_fault_reason(self, frame):
        freshness = self.config["freshness"]
        if frame.image_age_s > freshness["image_s"]:
            return "stale image"
        if frame.ultrasound_age_s > freshness["ultrasound_s"]:
            return "stale ultrasound"
        if frame.odom_age_s > freshness["odom_s"]:
            return "stale odometry"
        if frame.ambiguous:
            return "ambiguous tracking"

        distance = self.config["distance"]
        if frame.ultrasound_m <= distance["emergency_stop_m"]:
            return "emergency distance floor"
        if (
            not any(track.missing_frames == 0 for track in frame.tracks)
            and frame.ultrasound_m <= distance["side_trigger_m"]
        ):
            return "unknown obstacle"
        return None

    def _hold_decision(self, recovering=False):
        reason = self.hold_reason or "fault hold"
        if recovering:
            reason = "%s; recovering %d/%d" % (
                reason,
                self.hold_recovery_frames,
                self.config["decision"]["stable_frames"],
            )
        return Decision(
            "HOLD", 0.0, 0.0, 0.0, reason, False, False
        )

    def _start_confirmation_or_cruise(self, frame):
        if self._obstacle_zone_crossed():
            self.state = "FINISHED"
            return self._finished_decision()

        target = self._select_target(frame.tracks)
        if target is not None and self._qualifies(target, frame.ultrasound_m):
            return self._start_confirmation(frame, target, "CRUISE")

        return self._cruise_decision(frame, "cruise")

    def _start_confirmation(self, frame, target, resume_state):
        self.state = "CONFIRM"
        self.confirm_track_id = target.track_id
        self.confirm_resume_state = resume_state
        self.confirm_frames = 1
        return self._confirmation_decision(frame, "confirm")

    def _confirm(self, frame, target):
        if (
            target is None
            or target.track_id != self.confirm_track_id
            or not self._qualifies(target, frame.ultrasound_m)
        ):
            resume_state = self.confirm_resume_state
            self.confirm_track_id = None
            self.confirm_resume_state = "CRUISE"
            self.confirm_frames = 0
            if resume_state == "PASS":
                self.state = "PASS"
                return self._pass(frame)
            self.state = "CRUISE"
            return self._start_confirmation_or_cruise(frame)

        self.confirm_frames += 1
        if self.confirm_frames < self.config["decision"]["stable_frames"]:
            return self._confirmation_decision(frame, "confirm")

        self.state = "AVOID"
        self.active_track_id = target.track_id
        self.direction = self._avoid_direction(target)
        if not is_finite_number(self.return_line_target_m):
            self.return_line_target_m = lateral_position(
                frame.odom_x, frame.odom_y, self.initial_yaw
            )
        self.avoidance_count += 1
        self.confirm_track_id = None
        self.confirm_resume_state = "CRUISE"
        self.confirm_frames = 0
        self.safe_frames = 0
        self._clear_pass_progress()
        return self._avoid_motion_decision(frame, "avoid_start")

    def _avoid(self, frame):
        if self.active_track_id in frame.cleared_ids:
            return self._begin_pass(frame, "pass_target_cleared")

        target = self._find_track(frame.tracks, self.active_track_id)
        expected_zone = (
            "safe_right" if self.direction == "left" else "safe_left"
        )
        if (
            target is not None
            and target.missing_frames == 0
            and target.zone == expected_zone
        ):
            self.safe_frames += 1
        else:
            self.safe_frames = 0

        if self.safe_frames >= self.config["decision"]["stable_frames"]:
            return self._begin_pass(frame, "pass")

        return self._avoid_motion_decision(
            frame, "avoid_%s" % self.direction
        )

    def _pass(self, frame):
        pass_forward_complete = (
            is_finite_number(self.pass_progress_m)
            and self.pass_progress_m
            >= self.config["decision"]["min_pass_forward_m"]
        )
        if pass_forward_complete:
            return self._begin_return_line(frame)

        if self.avoidance_count < self.config["decision"]["max_avoidances"]:
            target = self._select_target(frame.tracks)
            if target is not None and self._qualifies(target, frame.ultrasound_m):
                return self._start_confirmation(frame, target, "PASS")

        return self._pass_decision(frame, "pass_distance")

    def _begin_pass(self, frame, reason):
        self._clear_active()
        self.state = "PASS"
        self.pass_origin_odom_x = frame.odom_x
        self.pass_origin_odom_y = frame.odom_y
        self.pass_progress_m = 0.0
        return self._pass_decision(frame, reason)

    def _begin_return_line(self, frame):
        self._clear_active()
        self.state = "RETURN_LINE"
        self.return_started_at = frame.now
        self.return_stable_frames = 0
        self.return_line_error_m = self._return_error(frame)
        if not is_finite_number(self.return_line_error_m):
            return self.force_hold("invalid return line")
        return self._return_line(frame)

    def _return_line(self, frame):
        error = self._return_error(frame)
        if not is_finite_number(error):
            return self.force_hold("invalid return line")
        self.return_line_error_m = error

        tolerance = self.config["decision"]["return_tolerance_m"]
        if abs(error) <= tolerance:
            self.return_stable_frames += 1
            if self.return_stable_frames >= self.config["decision"][
                "stable_frames"
            ]:
                self._clear_return_line()
                if self._obstacle_zone_crossed():
                    self.state = "FINISHED"
                    return self._finished_decision()
                self.state = "FINAL_CRUISE"
                return self._cruise_decision(
                    frame, "finish_forward_after_return"
                )
            return self._motion_decision(
                frame, 0.0, 0.0, "return_stable"
            )

        self.return_stable_frames = 0
        lateral_speed = self.config["speed"]["avoid_vy"]
        if error < 0.0:
            lateral_speed = -lateral_speed
            reason = "return_right"
        else:
            reason = "return_left"
        return self._motion_decision(
            frame, 0.0, lateral_speed, reason
        )

    def _final_cruise(self, frame):
        if self._obstacle_zone_crossed():
            self.state = "FINISHED"
            return self._finished_decision()
        return self._cruise_decision(
            frame, "finish_forward_after_return"
        )

    def _return_error(self, frame):
        if not is_finite_number(self.return_line_target_m):
            return float("nan")
        current = lateral_position(
            frame.odom_x, frame.odom_y, self.initial_yaw
        )
        return self.return_line_target_m - current

    def _handoff(self, frame):
        self._clear_active()
        self.state = "CRUISE"
        return self._start_confirmation_or_cruise(frame)

    def _obstacle_zone_crossed(self):
        return (
            is_finite_number(self.forward_progress_m)
            and self.forward_progress_m
            >= self.config["decision"]["finish_forward_m"]
        )

    def _clear_active(self):
        self.active_track_id = None
        self.direction = None
        self.safe_frames = 0

    def _clear_pass_progress(self):
        self.pass_origin_odom_x = None
        self.pass_origin_odom_y = None
        self.pass_progress_m = None

    def _clear_return_line(self):
        self.return_line_target_m = None
        self.return_started_at = None
        self.return_stable_frames = 0
        self.return_line_error_m = None

    @staticmethod
    def _avoid_direction(target):
        return "left"

    @staticmethod
    def _find_track(tracks, track_id):
        return next(
            (track for track in tracks if track.track_id == track_id), None
        )

    def _select_target(self, tracks):
        candidates = [
            track
            for track in tracks
            if track.missing_frames == 0
            and track.zone not in ("safe_left", "safe_right")
        ]
        if not candidates:
            return None

        image_center = self.config["image"]["width"] / 2.0
        return min(
            candidates,
            key=lambda track: (
                -track.detection.contour_area,
                abs(bbox_center_x(track.detection.bbox) - image_center),
                track.track_id,
            ),
        )

    def _qualifies(self, target, ultrasound_m):
        if target.missing_frames != 0:
            return False

        distance = self.config["distance"]
        if target.zone == "front":
            return ultrasound_m <= distance["front_trigger_m"]
        if target.zone == "side":
            return ultrasound_m <= distance["side_trigger_m"]
        return False

    def _cruise_decision(self, frame, reason):
        return self._motion_decision(
            frame, self.config["speed"]["cruise_vx"], 0.0, reason
        )

    def _pass_decision(self, frame, reason):
        return self._motion_decision(
            frame, self.config["speed"]["pass_vx"], 0.0, reason
        )

    def _confirmation_decision(self, frame, reason):
        if self.confirm_resume_state == "PASS":
            vx = self.config["speed"]["pass_vx"]
        else:
            vx = self.config["speed"]["cruise_vx"]
        return self._motion_decision(frame, vx, 0.0, reason)

    def _avoid_motion_decision(self, frame, reason):
        lateral_speed = self.config["speed"]["avoid_vy"]
        if self.direction == "right":
            lateral_speed = -lateral_speed
        return self._motion_decision(frame, 0.0, lateral_speed, reason)

    def _finished_decision(self):
        return Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )

    def _motion_decision(self, frame, vx, vy, reason):
        wz = heading_wz(self.initial_yaw, frame.yaw, self.config)
        return Decision(self.state, vx, vy, wz, reason, False, False)
