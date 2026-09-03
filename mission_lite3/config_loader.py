from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


class ConfigError(ValueError):
    """Raised when mission configuration is missing or unsafe to execute."""


DEFAULT_FIELD: Dict[str, Any] = {
    "field": {
        "width_m": 5.0,
        "length_m": 6.0,
        "origin": "start_area_center",
        "layout_reference": "26比赛资料/比赛具体位置图.png",
        "effective_area_note": "5m x 6m is measured only between the top and bottom black horizontal boundary lines in the route image",
        "axis_note": "robot starts 0.40 m map-up from the lower-right start-area center, facing map-left toward the obstacle area; before pickup, forward is map-left and positive strafe is map-down; after the pickup turn, forward is map-up",
    },
    "waypoints": {
        "start": {"x": 0.0, "y": 0.40, "yaw": 3.1416},
        "obstacle_entry": {"x": -1.1, "y": 0.8, "yaw": 3.1416},
        "obstacle_exit": {"x": -1.8, "y": 1.45, "yaw": 3.1416},
        "inspect_lower": {"x": -2.35, "y": 1.9, "yaw": 3.1416},
        "inspect_upper": {"x": -2.35, "y": 3.55, "yaw": 3.1416},
        "pickup": {"x": -0.75, "y": 2.65, "yaw": 1.5708},
        "place": {"x": -0.65, "y": 4.9, "yaw": 1.5708},
    },
    "zones": {
        "random_obstacle": {"center": [-1.25, 1.0], "size": [2.2, 1.2]},
        "inspection_lower": {"center": [-2.45, 1.9], "size": [1.2, 1.1]},
        "inspection_upper": {"center": [-2.45, 3.55], "size": [1.2, 1.1]},
        "pickup": {"center": [-0.7, 2.65], "size": [1.5, 1.0]},
        "placement": {"center": [-0.65, 4.9], "size": [1.8, 0.5]},
    },
    "scripted_route": {
        "pass_obstacle": [
            {
                "action": "obstacle_forward",
                "steps": 8,
                "clear_step_m": 0.20,
                "avoid_strafe_m": 0.45,
                "avoid_forward_m": 0.50,
                "max_avoid_attempts": 1,
                "return_after_avoid": True,
                "note": "start facing map-left and cross the random obstacle area",
            },
        ],
        "inspect_stop_1_arrive": [
            {
                "action": "strafe",
                "distance_m": -0.65,
                "note": "move map-up toward inspection stop 1 from the start point shifted 0.40 m map-up",
            },
            {"action": "forward", "distance_m": 0.40, "note": "move map-left 2 clear steps to inspection stop 1"},
        ],
        "inspect_stop_2_arrive": [
            {
                "action": "turn",
                "yaw_rad": -3.1416,
                "note": "turn 180 degrees from inspection stop 3 to inspection stop 2",
            },
        ],
        "inspect_stop_3_arrive": [
            {
                "action": "strafe",
                "distance_m": -1.18,
                "note": "move map-up from inspection stop 1 toward inspection stop 3",
            },
            {"action": "forward", "distance_m": 1.00, "note": "move map-left to inspection stops 2 and 3"},
            {"action": "turn", "yaw_rad": -1.5708, "note": "turn right to face inspection stop 3"},
        ],
        "inspect_stop_4_arrive": [
            {"action": "turn", "yaw_rad": -1.5708, "note": "turn right toward inspection stop 4 route"},
            {"action": "backward", "distance_m": 1.00, "note": "move map-right to inspection stop 4 column"},
            {"action": "strafe", "distance_m": -0.95, "note": "move map-up to inspection stop 4"},
        ],
        "inspect_stop_4_depart": [],
        "pickup_from_upper_inspection": [
            {"action": "turn", "yaw_rad": 1.5708, "note": "face map-down for pickup and placement"},
            {
                "action": "forward",
                "distance_m": 1.35,
                "note": "move map-down from inspection stop 4 toward pickup row",
            },
            {"action": "strafe", "distance_m": 1.30, "note": "move map-right into pickup lane"},
        ],
        "place_from_pickup": [
            {"action": "turn", "yaw_rad": -3.1416},
            {"action": "placement_row_yaw_align"},
            {"action": "placement_letter_approach"},
        ],
        "pickup_from_place": [
            {"action": "turn", "yaw_rad": 3.1416},
            {
                "action": "forward",
                "distance_m": 1.38,
                "front_stop_is_completion": True,
            },
            {"action": "pickup_lane_restore"},
        ],
        "placement_letter_strafe_m": {"A": 0.15, "B": -0.15, "C": -0.45, "D": -0.75},
    },
}


DEFAULT_ROBOT: Dict[str, Any] = {
    "network": {
        "motion_ip": "192.168.1.120",
        "motion_port": 43893,
        "perception_ip": "192.168.1.103",
        "perception_port": 43899,
    },
    "ros2": {
        "enabled": True,
        "cmd_vel_topic": "/cmd_vel",
        "odom_topic": "/leg_odom2",
        "imu_topic": "/imu/data",
        "ultrasound_topic": "/us_publisher/front_distance",
        "rear_ultrasound_topic": "/us_publisher/rear_distance",
    },
    "camera": {
        "front": "rtsp://192.168.1.120:8554/test",
        "arm": "/dev/video4",
        "frame_width": 1280,
        "frame_height": 720,
        "flush_grab_frames": 4,
        "stale_frame_reconnect_count": 15,
        "digital_zoom": 1.0,
        "wide_calibration": "mission_lite3/config/wide_angle_camera_calibration.json",
        "open_timeout_ms": 3000,
        "read_timeout_ms": 2000,
        "startup_first_frame_timeout_s": 6.0,
        "reconnect_backoff_s": 0.25,
    },
    "motion": {
        "assume_standing": True,
        "command_hz": 20.0,
        "heartbeat_hz": 4.0,
        "max_vx": 0.30,
        "max_vy": 0.20,
        "max_wz": 0.55,
        "cruise_vx": 0.18,
        "strafe_vy": 0.12,
        "turn_wz": 0.35,
        "axis_full_scale": 30000,
        "axis_max_vx": 1.0,
        "axis_max_vy": 0.5,
        "axis_max_wz": 1.5,
    },
    "navigation": {
        "feedback_required": True,
        "distance_tolerance_m": 0.03,
        "yaw_tolerance_rad": 0.04,
        "action_timeout_scale": 2.0,
        "minimum_action_timeout_s": 2.0,
        "startup_sensor_timeout_s": 3.0,
        "translation_path_hold_enabled": True,
        "translation_cross_track_kp_s": 1.0,
        "translation_max_cross_track_correction_mps": 0.04,
        "translation_cross_track_deadband_m": 0.003,
        "translation_max_cross_track_drift_m": 0.15,
        "translation_yaw_hold_kp_s": 1.2,
        "translation_max_wz_correction_rad_s": 0.12,
        "translation_yaw_deadband_deg": 0.30,
        "translation_max_yaw_drift_deg": 5.0,
    },
    "safety": {
        "front_stop_distance_m": 0.35,
        "front_ultrasound_min_valid_m": 0.03,
        "front_caution_distance_m": 0.75,
        "placement_forward_speed_mps": 0.08,
        "placement_front_filter_window_s": 0.8,
        "use_ultrasound_obstacle": True,
        "use_vision_obstacle": False,
        "max_roll_deg": 18.0,
        "max_pitch_deg": 18.0,
        "low_battery_fraction": 0.35,
        "state_max_age_s": 0.75,
        "require_fresh_state": True,
    },
    "fault_hold": {
        "enabled": True,
        "poll_interval_s": 0.50,
        "recovery_stable_checks": 5,
        "resume_signal_path": "/tmp/lite3_fault_resume",
        "max_wait_s": 30.0,
        "max_retries_per_state": 2,
        "max_retries_per_placement_state": 8,
    },
    "startup_avoidance": {
        "enabled": True,
        "image": {
            "source": "/dev/video0",
            "width": 1280,
            "height": 720,
            "fps": 25,
        },
        "hsv": {
            "lower": [0, 110, 80],
            "upper": [18, 255, 255],
            "min_area": 1000,
            "kernel_size": 7,
        },
        "zones": {
            "safe_left_right_edge_max": 354,
            "safe_right_left_edge_min": 1054,
            "front_center_min": 512,
            "front_center_max": 768,
        },
        "distance": {
            "emergency_stop_m": 0.15,
            "front_trigger_m": 0.40,
            "side_trigger_m": 0.40,
            "ultrasound_window": 20,
        },
        "tracking": {
            "min_iou": 0.20,
            "max_center_distance_px": 200,
            "min_area_ratio": 0.40,
            "max_area_ratio": 2.50,
            "ambiguous_cost_fraction": 0.10,
            "max_missing_frames": 5,
        },
        "decision": {
            "stable_frames": 5,
            "finish_forward_m": 1.60,
            "min_pass_forward_m": 1.28,
            "max_avoidances": 1,
            "return_tolerance_m": 0.02,
        },
        "speed": {
            "cruise_vx": 0.08,
            "avoid_vy": 0.08,
            "pass_vx": 0.102,
            "max_heading_wz": 0.15,
            "heading_kp": 0.8,
        },
        "freshness": {
            "image_s": 0.30,
            "ultrasound_s": 0.20,
            "odom_s": 0.20,
        },
        "fault_hold_retry_s": 0.50,
        "fault_hold_max_s": 30.0,
        "log_dir": "startup_avoidance_runs",
    },
    "vision": {
        "inspection_backend": "runtime_meter_anchor",
        "runtime_min_letter_confidence": 0.70,
        "runtime_fast_accept_confidence": 0.84,
        "runtime_fast_accept_margin": 0.20,
        "runtime_best_candidate_confidence": 0.82,
        "runtime_fast_min_pointer_hit_ratio": 0.60,
        "runtime_fast_min_pointer_run_ratio": 0.45,
        "allow_legacy_fallback": False,
        "stable_window": 5,
        "stable_votes": 3,
        "cone_hsv": {"lower": [5, 80, 80], "upper": [25, 255, 255], "min_area": 1500},
        "red_hsv_1": {"lower": [0, 80, 70], "upper": [10, 255, 255], "min_area": 800},
        "red_hsv_2": {"lower": [170, 80, 70], "upper": [179, 255, 255], "min_area": 800},
        "green_hsv": {"lower": [35, 60, 50], "upper": [90, 255, 255], "min_area": 800},
        "letter_min_confidence": 0.48,
        "dashboard_min_radius": 45,
    },
    "inspection": {
        "round_result_path": "round_result.json",
        "latest_stop_result_path": "latest_stop_result.json",
        "gate_pickup_on_round_result": True,
        "reset_round_result_on_mission_start": True,
        "use_wide_undistortion": True,
        "front_stop_distance_m": 0.28,
        "stop_dwell_seconds": 8.0,
        "speak_at_inspection_stop": True,
        "evidence_dir": "evidence",
        "place_pause_seconds": 3.0,
        "tag_localization": {
            "enabled": True,
            "family": "DICT_APRILTAG_36h11",
            "marker_border_bits": 2,
            "tag_size_m": 0.08,
            "min_edge_px": 24.0,
            "mask_for_recognition": True,
            "mask_margin_px": 4,
            "station_tag_ids": {
                "inspection_stop_1": 0,
                "inspection_stop_3": 2,
            },
            "targets": {
                "0": {"center_x_px": 615.75, "center_y_px": 438.50, "edge_px": 37.27},
                "2": {"center_x_px": 710.25, "center_y_px": 428.50, "edge_px": 43.80},
            },
            "samples_per_attempt": 3,
            "sample_timeout_s": 0.8,
            "max_iterations": 2,
            "center_tolerance_px": 12.0,
            "edge_tolerance_px": 2.0,
            "max_forward_step_m": 0.10,
            "max_strafe_step_m": 0.10,
            "min_motion_step_m": 0.015,
            "forward_speed_mps": 0.08,
            "strafe_speed_mps": 0.06,
            "positive_error_strafe_sign": -1,
            "restore_route_anchor": True,
            "return_motion_threshold_m": 0.02,
            "return_tolerance_m": 0.06,
            "return_yaw_tolerance_deg": 3.0,
            "return_max_correction_passes": 2,
            "station_overrides": {
                "inspection_stop_3": {
                    "max_iterations": 5,
                    "max_forward_step_m": 0.20,
                    "max_strafe_step_m": 0.20,
                },
            },
        },
    },
    "arm": {
        "enabled": True,
        "backend": "runtime",
        "port": "/dev/ttyUSB0",
        "camera_device": "/dev/v4l/by-id/usb-SXW_USB_Camera_200901010001-video-index0",
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps": 25,
        "baud": 115200,
        "timeout": 2.0,
        "stow_command": "moving-pose",
        "runtime_config": "mission_lite3/arm/runtime/strip_detector_grasp_config.json",
        "calibration": "mission_lite3/arm/runtime/camera_calibration.json",
        "grasp_reference": "mission_lite3/arm/runtime/grasp_reference_square_face.json",
        "moving_pose": "mission_lite3/arm/runtime/moving_pose.json",
        "place_reference": "mission_lite3/arm/runtime/place_reference.json",
        "run_log_dir": "grasp_runs",
        "result_dir": "logs",
        "grasp_height_mm": 55,
        "release_height_mm": 85,
        "max_retries": 2,
    },
    "pregrasp_red_align": {
        "enabled": True,
        "roi": [0.42, 0.55, 0.58, 0.85],
        "reference_linear_size_px": 94.2391,
        "linear_size_tolerance": 0.30,
        "loose_motion_min_linear_size_ratio": 0.75,
        "strict_motion_min_linear_size_ratio": 0.70,
        "strict_tracking_min_linear_size_ratio": 0.50,
        "loose_motion_min_center_y_ratio": 0.55,
        "loose_motion_stable_frames": 3,
        "loose_motion_center_tolerance_px": 20.0,
        "loose_motion_size_ratio_tolerance": 0.20,
        "reconnect_camera_after_pulse": True,
        "strafe_speed_mps": 0.08,
        "pulse_seconds": 0.25,
        "min_pulse_seconds": 0.15,
        "max_pulse_seconds": 1.00,
        "horizontal_error_strafe_gain_m_per_px": 0.00080,
        "settle_seconds": 0.35,
        "post_stop_settle_seconds": 0.50,
        "success_stable_frames": 3,
        "no_red_frame_limit": 5,
        "target_not_found_retries": 3,
        "target_search_enabled": True,
        "target_search_speed_mps": 0.08,
        "target_search_step_seconds": 0.625,
        "target_search_settle_seconds": 0.00,
        "target_search_bilateral_enabled": True,
        "target_search_until_found": True,
        "target_search_each_side_m": 0.50,
        "target_search_max_distance_m": 3.00,
        "target_search_require_odom_progress": True,
        "target_search_min_progress_m": 0.015,
        "target_search_max_stalled_pulses": 3,
        "target_search_max_net_lateral_m": 0.50,
        "target_search_return_to_origin_on_failure": True,
        "target_search_center_band": [0.40, 0.60],
        "target_search_front_hold_enabled": True,
        "target_search_front_target_m": 0.28,
        "target_search_front_deadband_m": 0.015,
        "target_search_front_hold_kp_s": 0.8,
        "target_search_front_max_vx_mps": 0.025,
        "target_search_front_edge_far_m": 0.60,
        "target_search_front_edge_jump_m": 0.25,
        "target_search_front_edge_confirm_samples": 2,
        "acquire_fine_max_strafe_distance_m": 0.15,
        "acquired_target_lost_frame_limit": 2,
        "target_search_odometry_stall_recovery_attempts": 2,
        "target_search_odometry_stall_recovery_settle_seconds": 0.30,
        "target_search_odometry_stall_recovery_pulse_seconds": 1.25,
        "preapproach_search_min_distance_m": 0.00,
        "max_pulses": 30,
        "max_seconds": 90.0,
        "max_strafe_distance_m": 0.50,
        "strafe_pose_hold_enabled": True,
        "forward_hold_kp_s": 1.0,
        "max_vx_correction_mps": 0.04,
        "forward_deadband_m": 0.005,
        "max_forward_drift_m": 0.15,
        "yaw_hold_kp_s": 1.2,
        "max_wz_correction_rad_s": 0.12,
        "yaw_deadband_deg": 0.30,
        "max_yaw_drift_deg": 5.0,
        "wide_parallel": {
            "enabled": True,
            "frames_per_measurement": 12,
            "min_valid_frames": 8,
            "tolerance_deg": 1.5,
            "max_range_deg": 1.0,
            "correction_speed_rad_s": 0.10,
            "coarse_error_deg": 1.5,
            "coarse_pulse_seconds": 0.35,
            "fine_pulse_seconds": 0.15,
            "error_fraction_per_correction": 0.5,
            "motion_response_gain": 0.5,
            "min_pulse_seconds": 0.15,
            "max_pulse_seconds": 0.80,
            "settle_seconds": 2.0,
            "max_corrections": 6,
            "positive_error_wz_sign": -1,
            "run_log_dir": "wide_box_alignment_runs",
        },
        "ultrasound_gate_enabled": True,
        "ultrasound_min_m": 0.10,
        "ultrasound_max_m": 2.0,
        "final_distance_max_m": 0.30,
        "final_distance_min_m": 0.25,
        "final_distance_attempts": 3,
        "run_log_dir": "pregrasp_align_runs",
    },
    "box_center_alignment": {
        # Keep automatic motion gated until both manual scenes pass and the
        # saved placement annotations have been used to calibrate the ROI.
        "enabled": False,
        "frames_per_measurement": 7,
        "min_valid_frames": 4,
        "max_center_range_fraction": 0.03,
        "tolerance_fraction": 0.05,
        "max_corrections": 3,
        "strafe_speed_mps": 0.08,
        "max_single_strafe_m": 0.25,
        "max_total_strafe_m": 0.75,
        "adjacent_box_spacing_m": 0.30,
        "pickup_m_per_pixel": 0.001,
        "positive_error_strafe_sign": -1,
        "settle_seconds": 0.5,
        "placement_roi": [0.0, 0.28, 1.0, 1.0],
        "placement_min_span_fraction": 0.30,
        "placement_min_center_gap_fraction": 0.05,
        "placement_tracking_enabled": True,
        "placement_tracking_min_separators": 2,
        "placement_tracking_max_scale_change_fraction": 0.20,
        "placement_tracking_max_residual_fraction": 0.025,
        "placement_tracking_min_motion_gain": 0.20,
        "placement_tracking_max_motion_gain": 1.50,
        "strafe_pose_hold_enabled": True,
        "forward_hold_kp_s": 1.0,
        "max_vx_correction_mps": 0.04,
        "forward_deadband_m": 0.003,
        "max_forward_drift_m": 0.15,
        "yaw_hold_kp_s": 1.2,
        "max_wz_correction_rad_s": 0.12,
        "yaw_deadband_deg": 0.30,
        "max_yaw_drift_deg": 5.0,
        "placement_letter_min_confidence": 0.50,
        "placement_label_min_area_fraction": 0.00035,
        "placement_label_max_area_fraction": 0.12,
        "placement_white_max_saturation": 90,
        "placement_white_min_value": 165,
        "placement_glyph_fallback_enabled": True,
        "placement_glyph_roi": [0.0, 0.48, 1.0, 0.95],
        "placement_glyph_min_width_fraction": 0.012,
        "placement_glyph_max_width_fraction": 0.080,
        "placement_glyph_min_height_fraction": 0.030,
        "placement_glyph_max_height_fraction": 0.150,
        "placement_glyph_min_aspect": 0.30,
        "placement_glyph_max_aspect": 1.40,
        "placement_glyph_expand_x": 3.0,
        "placement_glyph_expand_y": 2.5,
        "placement_label_row_fallback_enabled": True,
        "placement_label_row_gray_min": 180,
        "placement_label_row_roi": [0.15, 0.58, 0.85, 0.80],
        "placement_label_row_min_area_fraction": 0.0015,
        "placement_label_row_max_area_fraction": 0.015,
        "placement_label_row_min_fill": 0.72,
        "placement_label_row_min_aspect": 0.75,
        "placement_label_row_max_aspect": 1.80,
        "placement_label_row_min_gap_fraction": 0.055,
        "placement_label_row_max_gap_fraction": 0.16,
        "placement_label_row_max_y_range_fraction": 0.045,
        "placement_label_row_min_size_ratio": 0.70,
        "placement_label_row_anchor_confidence": 0.60,
        "recognition_run_log_dir": "box_recognition_runs",
        "alignment_run_log_dir": "box_center_alignment_runs",
        "fallback_enabled": True,
        "fallback_offsets_m": {
            "A": 0.15,
            "B": -0.15,
            "C": -0.45,
            "D": -0.75,
        },
    },
    "placement_letter_navigation": {
        "enabled": True,
        "letter_order": ["A", "B", "C", "D"],
        "letter_min_confidence": 0.60,
        "forward_speed_mps": 0.08,
        "front_stop_distance_m": 0.28,
        "forward_budget_m": 1.80,
        "search_step_m": 0.20,
        "min_center_correction_m": 0.02,
        "max_center_correction_m": 0.08,
        "center_gain_m_per_fraction": 0.45,
        "lateral_speed_mps": 0.08,
        "fine_strafe_distance_tolerance_m": 0.015,
        "max_lateral_search_m": 3.10,
        "bilateral_search_enabled": True,
        "lateral_search_each_side_m": 1.00,
        "immediate_complete_on_target_detection": False,
        "acquisition_center_band": [1.0 / 3.0, 2.0 / 3.0],
        "center_tolerance_fraction": 0.05,
        "final_approach_distance_m": 0.0,
        "final_approach_step_m": 0.0,
        "letter_spacing_m": 0.50,
        "max_anchor_jump_m": 0.08,
        "target_vote_window": 3,
        "target_min_votes": 2,
        "target_memory_max_misses": 6,
        "target_memory_max_lateral_m": 0.15,
        "target_memory_max_forward_m": 0.40,
        "target_memory_fraction_per_m": 0.50,
        "ultrasound_filter_samples": 5,
        "ultrasound_stable_samples": 5,
        "final_ultrasound_min_m": 0.27,
        "final_ultrasound_max_m": 0.30,
        "ultrasound_jump_reject_m": 0.25,
        "ultrasound_jump_confirm_samples": 3,
        "ultrasound_odom_consistency_tolerance_m": 0.15,
        "ultrasound_stuck_value_m": 0.28,
        "ultrasound_stuck_tolerance_m": 0.01,
        "approach_filter_warmup_s": 0.30,
        "require_visual_row_before_forward": False,
        "visual_row_preflight_attempts": 1,
        "visual_row_preflight_trigger_m": 0.33,
        "visual_ultrasound_start_tolerance_m": 0.35,
        "strafe_forward_hold_kp_s": 0.0,
        "strafe_max_vx_correction_mps": 0.0,
        "strafe_forward_deadband_m": 0.02,
        "search_hold_capture_samples": 5,
        "search_hold_capture_timeout_s": 1.50,
        "search_hold_capture_max_spread_m": 0.06,
        "search_hold_capture_min_m": 0.20,
        "search_hold_capture_max_m": 1.20,
        "search_hold_boundary_delta_m": 0.20,
        "search_hold_restore_attempts": 1,
        "search_hold_restore_speed_mps": 0.03,
        "search_hold_restore_min_step_m": 0.04,
        "search_hold_restore_max_step_m": 0.10,
        "strafe_max_forward_drift_m": 0.15,
        "strafe_yaw_hold_kp_s": 1.2,
        "strafe_max_wz_correction_rad_s": 0.12,
        "strafe_yaw_deadband_deg": 0.30,
        "strafe_max_yaw_drift_deg": 5.0,
        "motion_stall_timeout_s": 2.0,
        "motion_stall_min_progress_m": 0.01,
        "strafe_zero_progress_reverse_count": 2,
        "motion_stall_retries": 3,
        "motion_recovery_pause_s": 0.30,
        "motion_recovery_speed_mps": 0.05,
        "approach_slow_distance_m": 0.40,
        "approach_creep_distance_m": 0.33,
        "approach_slow_speed_mps": 0.05,
        "approach_creep_speed_mps": 0.025,
        "cached_geometry_enabled": True,
        "required_center_frames": 5,
        "capture_retries": 3,
        "image_timeout_s": 0.50,
        "total_timeout_s": 0.0,
        "physical_left_strafe_sign": 1,
        "run_log_dir": "placement_letter_navigation_runs",
    },
    "pickup_transfer": {
        "enabled": True,
        "pre_retreat_yaw_alignment_enabled": False,
        "retreat_target_front_m": 0.80,
        "retreat_stop_threshold_m": 0.77,
        "retreat_max_front_m": 0.90,
        "retreat_speed_mps": 0.06,
        "retreat_timeout_s": 12.0,
        "retreat_max_odom_m": 0.55,
        "retreat_stuck_front_fallback_enabled": True,
        "retreat_stuck_front_value_m": 0.28,
        "retreat_stuck_front_tolerance_m": 0.01,
        "retreat_stuck_front_min_samples": 5,
        "retreat_odom_fallback_target_m": 0.44,
        "post_retreat_yaw_alignment_enabled": True,
        "retreat_lateral_hold_kp_s": 1.0,
        "retreat_max_vy_correction_mps": 0.04,
        "retreat_lateral_deadband_m": 0.003,
        "retreat_max_lateral_drift_m": 0.10,
        "yaw_hold_kp_s": 1.2,
        "max_wz_correction_rad_s": 0.12,
        "yaw_deadband_deg": 0.30,
        "max_yaw_drift_deg": 5.0,
        "departure_tolerance_fraction": 0.03,
        "arrival_tolerance_fraction": 0.06,
        "transfer_distance_m": 1.38,
        "lane_strafe_speed_mps": 0.08,
        "lane_forward_hold_kp_s": 1.0,
        "lane_max_vx_correction_mps": 0.04,
        "lane_forward_deadband_m": 0.003,
        "lane_max_forward_drift_m": 0.15,
        "max_recorded_lane_strafe_m": 1.05,
        "lane_offsets_m": {
            "A": 1.0,
            "B": 0.5,
            "C": 0.0,
            "D": -0.5,
        },
    },
    "placement_yaw_alignment": {
        "enabled": True,
        "frames_per_measurement": 12,
        "min_valid_frames": 8,
        "tolerance_deg": 1.5,
        "max_range_deg": 2.0,
        "correction_speed_rad_s": 0.10,
        "coarse_error_deg": 1.5,
        "coarse_pulse_seconds": 0.35,
        "fine_pulse_seconds": 0.15,
        "error_fraction_per_correction": 0.5,
        "motion_response_gain": 0.5,
        "min_pulse_seconds": 0.15,
        "max_pulse_seconds": 0.80,
        "settle_seconds": 2.0,
        "max_corrections": 3,
        "positive_error_wz_sign": -1,
        "min_row_span_fraction": 0.38,
        "run_log_dir": "placement_yaw_alignment_runs",
    },
    "audio": {
        "enabled": True,
        "mode": "remote_udp",
        "audio_dir": "mission_lite3/inspection_audio",
        "fallback_to_tts_on_audio_failure": False,
        "remote_host": "192.168.1.120",
        "remote_port": 43910,
        "remote_timeout_seconds": 8.0,
        "remote_retries": 1,
        "remote_gain_db": 3.0,
        "prewarm_enabled": True,
        "prewarm_duration_s": 0.8,
        "command": "spd-say",
        "args": ["-w"],
        "timeout_seconds": 12.0,
        "pulse_sink": "alsa_output.platform-sound.analog-stereo",
        "pulse_server": "unix:/run/user/1000/pulse/native",
        "pulse_volume": "60%",
        "prepare_pulse": True,
        "pulse_setup_timeout_seconds": 3.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_file(path: Path, *, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigError(f"required config file is missing: {path}")
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"failed to read config file {path}: {exc}") from exc
    try:
        import yaml  # type: ignore

        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{path} is not valid JSON and PyYAML is unavailable; install PyYAML to read YAML syntax"
            ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return data


def _number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return number


def _strict_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number, not a boolean")
    return _number(value, name, minimum=minimum, maximum=maximum)


def _strict_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return value


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing or invalid config section: {key}")
    return value


def _validate_startup_avoidance(config: Mapping[str, Any]) -> None:
    avoidance = _require_mapping(config, "startup_avoidance")
    if not isinstance(avoidance.get("enabled"), bool):
        raise ConfigError("startup_avoidance.enabled must be a boolean")

    section_names = (
        "image",
        "hsv",
        "zones",
        "distance",
        "tracking",
        "decision",
        "speed",
        "freshness",
    )
    sections: Dict[str, Mapping[str, Any]] = {}
    for section_name in section_names:
        section = avoidance.get(section_name)
        if not isinstance(section, Mapping):
            raise ConfigError(
                f"missing or invalid config section: startup_avoidance.{section_name}"
            )
        sections[section_name] = section

    image = sections["image"]
    source = image.get("source")
    if not (
        (isinstance(source, str) and bool(source.strip()))
        or (isinstance(source, int) and not isinstance(source, bool) and source >= 0)
    ):
        raise ConfigError(
            "startup_avoidance.image.source must be a non-empty path or nonnegative camera index"
        )
    width = _strict_integer(
        image.get("width"), "startup_avoidance.image.width", minimum=1
    )
    _strict_integer(
        image.get("height"), "startup_avoidance.image.height", minimum=1
    )
    _strict_integer(image.get("fps"), "startup_avoidance.image.fps", minimum=1)

    hsv = sections["hsv"]
    hsv_bounds: Dict[str, list[int]] = {}
    for bound_name in ("lower", "upper"):
        bound = hsv.get(bound_name)
        if not isinstance(bound, list) or len(bound) != 3:
            raise ConfigError(
                f"startup_avoidance.hsv.{bound_name} must contain three integers"
            )
        for index, value in enumerate(bound):
            _strict_integer(
                value,
                f"startup_avoidance.hsv.{bound_name}[{index}]",
                minimum=0,
                maximum=(179 if index == 0 else 255),
            )
        hsv_bounds[bound_name] = bound
    if any(
        lower > upper
        for lower, upper in zip(hsv_bounds["lower"], hsv_bounds["upper"])
    ):
        raise ConfigError(
            "startup_avoidance.hsv.lower must not exceed hsv.upper"
        )
    _strict_number(
        hsv.get("min_area"),
        "startup_avoidance.hsv.min_area",
        minimum=0.000001,
    )
    kernel_size = _strict_integer(
        hsv.get("kernel_size"),
        "startup_avoidance.hsv.kernel_size",
        minimum=1,
    )
    if kernel_size % 2 == 0:
        raise ConfigError("startup_avoidance.hsv.kernel_size must be odd")

    zones = sections["zones"]
    zone_values = {
        name: _strict_integer(
            zones.get(name),
            f"startup_avoidance.zones.{name}",
            minimum=0,
            maximum=width,
        )
        for name in (
            "safe_left_right_edge_max",
            "safe_right_left_edge_min",
            "front_center_min",
            "front_center_max",
        )
    }
    if not (
        zone_values["safe_left_right_edge_max"]
        < zone_values["front_center_min"]
        <= zone_values["front_center_max"]
        < zone_values["safe_right_left_edge_min"]
    ):
        raise ConfigError(
            "startup_avoidance zones must be ordered safe-left < front range < safe-right"
        )

    distance = sections["distance"]
    emergency_m = _strict_number(
        distance.get("emergency_stop_m"),
        "startup_avoidance.distance.emergency_stop_m",
        minimum=0.000001,
    )
    front_m = _strict_number(
        distance.get("front_trigger_m"),
        "startup_avoidance.distance.front_trigger_m",
        minimum=0.000001,
    )
    if front_m <= emergency_m:
        raise ConfigError(
            "startup_avoidance.distance.front_trigger_m must be greater than emergency_stop_m"
        )
    _strict_number(
        distance.get("side_trigger_m"),
        "startup_avoidance.distance.side_trigger_m",
        minimum=front_m,
    )
    _strict_integer(
        distance.get("ultrasound_window"),
        "startup_avoidance.distance.ultrasound_window",
        minimum=1,
    )

    tracking = sections["tracking"]
    _strict_number(
        tracking.get("min_iou"),
        "startup_avoidance.tracking.min_iou",
        minimum=0.0,
        maximum=1.0,
    )
    _strict_number(
        tracking.get("max_center_distance_px"),
        "startup_avoidance.tracking.max_center_distance_px",
        minimum=0.000001,
    )
    min_area_ratio = _strict_number(
        tracking.get("min_area_ratio"),
        "startup_avoidance.tracking.min_area_ratio",
        minimum=0.000001,
    )
    _strict_number(
        tracking.get("max_area_ratio"),
        "startup_avoidance.tracking.max_area_ratio",
        minimum=min_area_ratio,
    )
    _strict_number(
        tracking.get("ambiguous_cost_fraction"),
        "startup_avoidance.tracking.ambiguous_cost_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _strict_integer(
        tracking.get("max_missing_frames"),
        "startup_avoidance.tracking.max_missing_frames",
        minimum=1,
    )

    decision = sections["decision"]
    _strict_integer(
        decision.get("stable_frames"),
        "startup_avoidance.decision.stable_frames",
        minimum=1,
    )
    field = _require_mapping(config, "field")
    field_length_m = _strict_number(
        field.get("length_m"), "field.length_m", minimum=0.1
    )
    _strict_number(
        decision.get("finish_forward_m"),
        "startup_avoidance.decision.finish_forward_m",
        minimum=0.1,
        maximum=field_length_m,
    )
    _strict_number(
        decision.get("min_pass_forward_m"),
        "startup_avoidance.decision.min_pass_forward_m",
        minimum=0.1,
        maximum=field_length_m,
    )
    _strict_integer(
        decision.get("max_avoidances"),
        "startup_avoidance.decision.max_avoidances",
        minimum=0,
    )
    _strict_number(
        decision.get("return_tolerance_m"),
        "startup_avoidance.decision.return_tolerance_m",
        minimum=0.000001,
        maximum=0.10,
    )

    speed = sections["speed"]
    speed_limits = {
        "cruise_vx": float(config["motion"]["max_vx"]),
        "avoid_vy": float(config["motion"]["max_vy"]),
        "pass_vx": float(config["motion"]["max_vx"]),
        "max_heading_wz": float(config["motion"]["max_wz"]),
    }
    for name, maximum in speed_limits.items():
        _strict_number(
            speed.get(name),
            f"startup_avoidance.speed.{name}",
            minimum=0.000001,
            maximum=maximum,
        )
    _strict_number(
        speed.get("heading_kp"),
        "startup_avoidance.speed.heading_kp",
        minimum=0.000001,
    )

    freshness = sections["freshness"]
    for name in ("image_s", "ultrasound_s", "odom_s"):
        _strict_number(
            freshness.get(name),
            f"startup_avoidance.freshness.{name}",
            minimum=0.000001,
        )

    _strict_number(
        avoidance.get("fault_hold_retry_s"),
        "startup_avoidance.fault_hold_retry_s",
        minimum=0.0,
        maximum=60.0,
    )
    _strict_number(
        avoidance.get("fault_hold_max_s"),
        "startup_avoidance.fault_hold_max_s",
        minimum=0.1,
        maximum=3600.0,
    )

    log_dir = avoidance.get("log_dir")
    if not isinstance(log_dir, str) or not log_dir.strip():
        raise ConfigError("startup_avoidance.log_dir must be a non-empty string")


def _validate_route(config: Mapping[str, Any]) -> None:
    route = _require_mapping(config, "scripted_route")
    known_actions = {
        "forward",
        "backward",
        "strafe",
        "turn",
        "wait",
        "obstacle_forward",
        "placement_row_yaw_align",
        "placement_lane_strafe",
        "placement_letter_approach",
        "pickup_lane_restore",
    }
    for route_name, actions in route.items():
        if route_name == "placement_letter_strafe_m":
            offsets = actions
            if not isinstance(offsets, Mapping):
                raise ConfigError("scripted_route.placement_letter_strafe_m must be a mapping")
            for letter in ("A", "B", "C", "D"):
                _number(offsets.get(letter), f"scripted_route.placement_letter_strafe_m.{letter}")
            continue
        if not isinstance(actions, list):
            raise ConfigError(f"scripted_route.{route_name} must be a list")
        for index, action in enumerate(actions):
            prefix = f"scripted_route.{route_name}[{index}]"
            if not isinstance(action, Mapping):
                raise ConfigError(f"{prefix} must be a mapping")
            kind = str(action.get("action") or "")
            if kind not in known_actions:
                raise ConfigError(f"{prefix}.action is unknown: {kind!r}")
            if kind in {"forward", "backward", "strafe"}:
                _number(action.get("distance_m"), f"{prefix}.distance_m")
            elif kind == "turn":
                _number(action.get("yaw_rad"), f"{prefix}.yaw_rad")
            elif kind == "wait":
                _number(action.get("seconds"), f"{prefix}.seconds", minimum=0.0)
            elif kind == "obstacle_forward":
                _number(action.get("steps", 1), f"{prefix}.steps", minimum=0.0)
                _number(action.get("clear_step_m", 0.25), f"{prefix}.clear_step_m", minimum=0.0)
                _number(action.get("avoid_strafe_m", 0.35), f"{prefix}.avoid_strafe_m")
                _number(action.get("avoid_forward_m", 0.30), f"{prefix}.avoid_forward_m", minimum=0.0)
            if "front_stop_is_completion" in action:
                if kind != "forward":
                    raise ConfigError(
                        f"{prefix}.front_stop_is_completion is valid only for forward actions"
                    )
                if not isinstance(action["front_stop_is_completion"], bool):
                    raise ConfigError(
                        f"{prefix}.front_stop_is_completion must be a boolean"
                    )

    pickup_transfer = _require_mapping(config, "pickup_transfer")
    if pickup_transfer.get("enabled"):
        place_actions = route.get("place_from_pickup")
        action_kinds = (
            [str(action.get("action") or "") for action in place_actions]
            if isinstance(place_actions, list)
            else []
        )
        expected = [
            "turn",
            "placement_row_yaw_align",
            "placement_letter_approach",
        ]
        if action_kinds != expected:
            raise ConfigError(
                "scripted_route.place_from_pickup must be exactly turn, "
                "placement_row_yaw_align, placement_letter_approach when "
                "pickup_transfer.enabled is true"
            )


def validate_config(config: Mapping[str, Any]) -> None:
    for section in (
        "network",
        "ros2",
        "camera",
        "motion",
        "navigation",
        "safety",
        "fault_hold",
        "startup_avoidance",
        "vision",
        "inspection",
        "arm",
        "pregrasp_red_align",
        "box_center_alignment",
        "placement_letter_navigation",
        "pickup_transfer",
        "placement_yaw_alignment",
        "audio",
    ):
        _require_mapping(config, section)
    camera = _require_mapping(config, "camera")
    _number(
        camera.get("startup_first_frame_timeout_s", 6.0),
        "camera.startup_first_frame_timeout_s",
        minimum=0.1,
        maximum=30.0,
    )
    motion = _require_mapping(config, "motion")
    for key in ("command_hz", "heartbeat_hz", "max_vx", "max_vy", "max_wz", "cruise_vx", "strafe_vy", "turn_wz"):
        _number(motion.get(key), f"motion.{key}", minimum=0.000001)
    navigation = _require_mapping(config, "navigation")
    _number(navigation.get("distance_tolerance_m"), "navigation.distance_tolerance_m", minimum=0.0)
    _number(navigation.get("yaw_tolerance_rad"), "navigation.yaw_tolerance_rad", minimum=0.0)
    _number(navigation.get("action_timeout_scale"), "navigation.action_timeout_scale", minimum=1.0)
    _number(navigation.get("minimum_action_timeout_s"), "navigation.minimum_action_timeout_s", minimum=0.1)
    _number(navigation.get("startup_sensor_timeout_s"), "navigation.startup_sensor_timeout_s", minimum=0.1)
    if not isinstance(navigation.get("translation_path_hold_enabled"), bool):
        raise ConfigError("navigation.translation_path_hold_enabled must be a boolean")
    for key in (
        "translation_cross_track_kp_s",
        "translation_max_cross_track_correction_mps",
        "translation_cross_track_deadband_m",
        "translation_max_cross_track_drift_m",
        "translation_yaw_hold_kp_s",
        "translation_max_wz_correction_rad_s",
        "translation_yaw_deadband_deg",
        "translation_max_yaw_drift_deg",
    ):
        _number(navigation.get(key), f"navigation.{key}", minimum=0.0)
    if (
        navigation["translation_cross_track_deadband_m"]
        > navigation["translation_max_cross_track_drift_m"]
    ):
        raise ConfigError(
            "navigation.translation_cross_track_deadband_m must not exceed "
            "translation_max_cross_track_drift_m"
        )
    if (
        navigation["translation_yaw_deadband_deg"]
        > navigation["translation_max_yaw_drift_deg"]
    ):
        raise ConfigError(
            "navigation.translation_yaw_deadband_deg must not exceed "
            "translation_max_yaw_drift_deg"
        )
    if (
        navigation["translation_max_cross_track_correction_mps"]
        > min(motion["max_vx"], motion["max_vy"])
    ):
        raise ConfigError(
            "navigation.translation_max_cross_track_correction_mps exceeds "
            "the configured translation limits"
        )
    if (
        navigation["translation_max_wz_correction_rad_s"]
        > motion["max_wz"]
    ):
        raise ConfigError(
            "navigation.translation_max_wz_correction_rad_s exceeds motion.max_wz"
        )
    safety = _require_mapping(config, "safety")
    _number(safety.get("front_stop_distance_m"), "safety.front_stop_distance_m", minimum=0.01)
    _number(safety.get("state_max_age_s", 0.75), "safety.state_max_age_s", minimum=0.05)
    fault_hold = _require_mapping(config, "fault_hold")
    if not isinstance(fault_hold.get("enabled"), bool):
        raise ConfigError("fault_hold.enabled must be a boolean")
    _strict_number(
        fault_hold.get("poll_interval_s"),
        "fault_hold.poll_interval_s",
        minimum=0.0,
        maximum=60.0,
    )
    _strict_integer(
        fault_hold.get("recovery_stable_checks"),
        "fault_hold.recovery_stable_checks",
        minimum=1,
        maximum=1000,
    )
    resume_signal_path = fault_hold.get("resume_signal_path")
    if not isinstance(resume_signal_path, str) or not resume_signal_path.strip():
        raise ConfigError("fault_hold.resume_signal_path must be a non-empty string")
    _strict_number(
        fault_hold.get("max_wait_s"),
        "fault_hold.max_wait_s",
        minimum=0.1,
        maximum=3600.0,
    )
    _strict_integer(
        fault_hold.get("max_retries_per_state"),
        "fault_hold.max_retries_per_state",
        minimum=0,
        maximum=100,
    )
    _strict_integer(
        fault_hold.get("max_retries_per_placement_state"),
        "fault_hold.max_retries_per_placement_state",
        minimum=0,
        maximum=100,
    )
    _validate_startup_avoidance(config)
    vision = _require_mapping(config, "vision")
    for key, default in (
        ("runtime_min_letter_confidence", 0.70),
        ("runtime_fast_accept_confidence", 0.84),
        ("runtime_fast_accept_margin", 0.20),
        ("runtime_best_candidate_confidence", 0.82),
        ("runtime_fast_min_pointer_hit_ratio", 0.60),
        ("runtime_fast_min_pointer_run_ratio", 0.45),
    ):
        _number(vision.get(key, default), f"vision.{key}", minimum=0.0, maximum=1.0)
    window = int(_number(vision.get("stable_window"), "vision.stable_window", minimum=1))
    votes = int(_number(vision.get("stable_votes"), "vision.stable_votes", minimum=1))
    if votes > window:
        raise ConfigError("vision.stable_votes must be <= vision.stable_window")
    if votes * 2 <= window:
        raise ConfigError("vision.stable_votes must form a strict majority of vision.stable_window")
    inspection = _require_mapping(config, "inspection")
    _number(
        inspection.get("front_stop_distance_m", safety["front_stop_distance_m"]),
        "inspection.front_stop_distance_m",
        minimum=0.01,
    )
    if bool(inspection.get("use_wide_undistortion", False)):
        camera = _require_mapping(config, "camera")
        raw_path = camera.get("wide_calibration")
        if not raw_path:
            raise ConfigError("camera.wide_calibration is required for inspection undistortion")
        path = Path(str(raw_path)).expanduser()
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.is_file():
            raise ConfigError(f"camera.wide_calibration file is missing: {resolved}")
    tag_localization = inspection.get("tag_localization", {})
    if not isinstance(tag_localization, Mapping):
        raise ConfigError("inspection.tag_localization must be a mapping")
    if not isinstance(tag_localization.get("enabled", False), bool):
        raise ConfigError("inspection.tag_localization.enabled must be a boolean")
    if not isinstance(tag_localization.get("mask_for_recognition", True), bool):
        raise ConfigError(
            "inspection.tag_localization.mask_for_recognition must be a boolean"
        )
    if not isinstance(tag_localization.get("restore_route_anchor", True), bool):
        raise ConfigError(
            "inspection.tag_localization.restore_route_anchor must be a boolean"
        )
    if bool(tag_localization.get("enabled", False)):
        family = tag_localization.get("family", "")
        if not isinstance(family, str) or not family.strip():
            raise ConfigError("inspection.tag_localization.family must be a string")
        for key, minimum, maximum in (
            ("marker_border_bits", 1.0, 4.0),
            ("tag_size_m", 0.01, 0.50),
            ("min_edge_px", 4.0, 500.0),
            ("mask_margin_px", 0.0, 100.0),
            ("samples_per_attempt", 1.0, 20.0),
            ("sample_timeout_s", 0.05, 5.0),
            ("max_iterations", 1.0, 5.0),
            ("center_tolerance_px", 1.0, 200.0),
            ("edge_tolerance_px", 0.1, 100.0),
            ("max_forward_step_m", 0.0, 0.30),
            ("max_strafe_step_m", 0.0, 0.30),
            ("min_motion_step_m", 0.0, 0.10),
            ("forward_speed_mps", 0.01, 0.30),
            ("strafe_speed_mps", 0.01, 0.30),
            ("return_motion_threshold_m", 0.0, 0.10),
            ("return_tolerance_m", 0.01, 0.20),
            ("return_yaw_tolerance_deg", 0.1, 15.0),
            ("return_max_correction_passes", 0.0, 5.0),
        ):
            _number(
                tag_localization.get(key),
                f"inspection.tag_localization.{key}",
                minimum=minimum,
                maximum=maximum,
            )
        return_passes = float(
            tag_localization.get("return_max_correction_passes", 0)
        )
        if return_passes != int(return_passes):
            raise ConfigError(
                "inspection.tag_localization.return_max_correction_passes "
                "must be an integer"
            )
        if float(tag_localization.get("return_tolerance_m", 0.0)) < float(
            tag_localization.get("return_motion_threshold_m", 0.0)
        ):
            raise ConfigError(
                "inspection.tag_localization.return_tolerance_m must be >= "
                "return_motion_threshold_m"
            )
        station_ids = tag_localization.get("station_tag_ids")
        expected_station_ids = {
            "inspection_stop_1": 0,
            "inspection_stop_3": 2,
        }
        if station_ids != expected_station_ids:
            raise ConfigError(
                "inspection.tag_localization.station_tag_ids must map "
                "inspection stops 1 and 3 to tag ids 0 and 2"
            )
        targets = tag_localization.get("targets")
        if not isinstance(targets, Mapping):
            raise ConfigError("inspection.tag_localization.targets must be a mapping")
        for tag_id in expected_station_ids.values():
            target = targets.get(str(tag_id), targets.get(tag_id))
            if not isinstance(target, Mapping):
                raise ConfigError(
                    f"inspection.tag_localization.targets.{tag_id} must be a mapping"
                )
            for key in ("center_x_px", "center_y_px", "edge_px"):
                _number(
                    target.get(key),
                    f"inspection.tag_localization.targets.{tag_id}.{key}",
                    minimum=0.0 if key != "edge_px" else 1.0,
                )
        if int(tag_localization.get("positive_error_strafe_sign", 0)) not in {-1, 1}:
            raise ConfigError(
                "inspection.tag_localization.positive_error_strafe_sign must be -1 or 1"
            )
        station_overrides = tag_localization.get("station_overrides", {})
        if not isinstance(station_overrides, Mapping):
            raise ConfigError(
                "inspection.tag_localization.station_overrides must be a mapping"
            )
        override_limits = {
            "max_iterations": (1.0, 5.0),
            "max_forward_step_m": (0.0, 0.30),
            "max_strafe_step_m": (0.0, 0.30),
        }
        for stop_name, override in station_overrides.items():
            if stop_name not in expected_station_ids:
                raise ConfigError(
                    "inspection.tag_localization.station_overrides contains "
                    f"unknown stop: {stop_name}"
                )
            if not isinstance(override, Mapping):
                raise ConfigError(
                    "inspection.tag_localization.station_overrides."
                    f"{stop_name} must be a mapping"
                )
            unknown_keys = set(override) - set(override_limits)
            if unknown_keys:
                raise ConfigError(
                    "inspection.tag_localization.station_overrides."
                    f"{stop_name} has unknown keys: {sorted(unknown_keys)}"
                )
            for key, value in override.items():
                minimum, maximum = override_limits[key]
                _number(
                    value,
                    "inspection.tag_localization.station_overrides."
                    f"{stop_name}.{key}",
                    minimum=minimum,
                    maximum=maximum,
                )
    audio = _require_mapping(config, "audio")
    _number(audio.get("remote_gain_db", 3.0), "audio.remote_gain_db", minimum=-12.0, maximum=6.0)
    if not isinstance(audio.get("prewarm_enabled", True), bool):
        raise ConfigError("audio.prewarm_enabled must be a boolean")
    _number(
        audio.get("prewarm_duration_s", 0.8),
        "audio.prewarm_duration_s",
        minimum=0.1,
        maximum=3.0,
    )
    arm = _require_mapping(config, "arm")
    if bool(arm.get("enabled", True)) and str(arm.get("backend", "runtime")).lower() == "runtime":
        for key in ("runtime_config", "calibration", "grasp_reference", "moving_pose", "place_reference"):
            raw_path = arm.get(key)
            if not raw_path:
                raise ConfigError(f"arm.{key} is required for the runtime backend")
            path = Path(str(raw_path)).expanduser()
            resolved = path if path.is_absolute() else PROJECT_ROOT / path
            if not resolved.is_file():
                raise ConfigError(f"arm.{key} file is missing: {resolved}")
    pregrasp = _require_mapping(config, "pregrasp_red_align")
    ultrasound_min_m = _number(
        pregrasp.get("ultrasound_min_m", 0.10),
        "pregrasp_red_align.ultrasound_min_m",
        minimum=0.0,
    )
    ultrasound_max_m = _number(
        pregrasp.get("ultrasound_max_m", 2.0),
        "pregrasp_red_align.ultrasound_max_m",
        minimum=0.01,
    )
    if ultrasound_max_m <= ultrasound_min_m:
        raise ConfigError(
            "pregrasp_red_align.ultrasound_max_m must be greater than ultrasound_min_m"
        )
    final_distance_max_m = _number(
        pregrasp.get("final_distance_max_m", 0.30),
        "pregrasp_red_align.final_distance_max_m",
        minimum=0.01,
    )
    if not ultrasound_min_m < final_distance_max_m < ultrasound_max_m:
        raise ConfigError(
            "pregrasp_red_align.final_distance_max_m must be inside the "
            "configured ultrasound range"
        )
    final_distance_min_m = _number(
        pregrasp.get("final_distance_min_m", 0.25),
        "pregrasp_red_align.final_distance_min_m",
        minimum=0.01,
    )
    if not ultrasound_min_m < final_distance_min_m < final_distance_max_m:
        raise ConfigError(
            "pregrasp_red_align.final_distance_min_m must be above the configured "
            "ultrasound minimum and below final_distance_max_m"
        )
    final_distance_attempts = _number(
        pregrasp.get("final_distance_attempts", 3),
        "pregrasp_red_align.final_distance_attempts",
        minimum=1.0,
    )
    if final_distance_attempts != int(final_distance_attempts):
        raise ConfigError(
            "pregrasp_red_align.final_distance_attempts must be an integer"
        )
    legacy_pulse_seconds = _number(
        pregrasp.get("pulse_seconds", 0.25),
        "pregrasp_red_align.pulse_seconds",
        minimum=0.01,
    )
    min_pulse_seconds = _number(
        pregrasp.get("min_pulse_seconds", legacy_pulse_seconds),
        "pregrasp_red_align.min_pulse_seconds",
        minimum=0.01,
    )
    max_pulse_seconds = _number(
        pregrasp.get("max_pulse_seconds", legacy_pulse_seconds),
        "pregrasp_red_align.max_pulse_seconds",
        minimum=0.01,
    )
    if max_pulse_seconds < min_pulse_seconds:
        raise ConfigError(
            "pregrasp_red_align.max_pulse_seconds must be >= "
            "pregrasp_red_align.min_pulse_seconds"
        )
    _number(
        pregrasp.get("horizontal_error_strafe_gain_m_per_px", 0.00080),
        "pregrasp_red_align.horizontal_error_strafe_gain_m_per_px",
        minimum=0.000001,
    )
    _strict_integer(
        pregrasp.get("target_not_found_retries", 3),
        "pregrasp_red_align.target_not_found_retries",
        minimum=0,
    )
    if not isinstance(pregrasp.get("target_search_enabled", True), bool):
        raise ConfigError(
            "pregrasp_red_align.target_search_enabled must be a boolean"
        )
    if not isinstance(pregrasp.get("target_search_bilateral_enabled", False), bool):
        raise ConfigError(
            "pregrasp_red_align.target_search_bilateral_enabled must be a boolean"
        )
    if not isinstance(pregrasp.get("target_search_until_found", False), bool):
        raise ConfigError(
            "pregrasp_red_align.target_search_until_found must be a boolean"
        )
    for key, default in (
        ("target_search_require_odom_progress", True),
        ("target_search_return_to_origin_on_failure", True),
        ("target_search_front_hold_enabled", True),
    ):
        if not isinstance(pregrasp.get(key, default), bool):
            raise ConfigError(f"pregrasp_red_align.{key} must be a boolean")
    _strict_number(
        pregrasp.get("target_search_speed_mps", 0.08),
        "pregrasp_red_align.target_search_speed_mps",
        minimum=0.000001,
        maximum=float(config["motion"]["max_vy"]),
    )
    _strict_number(
        pregrasp.get("target_search_step_seconds", 1.00),
        "pregrasp_red_align.target_search_step_seconds",
        minimum=0.01,
        maximum=10.0,
    )
    _strict_number(
        pregrasp.get("target_search_settle_seconds", 0.00),
        "pregrasp_red_align.target_search_settle_seconds",
        minimum=0.0,
        maximum=10.0,
    )
    target_search_each_side_m = _strict_number(
        pregrasp.get("target_search_each_side_m", 1.00),
        "pregrasp_red_align.target_search_each_side_m",
        minimum=0.01,
        maximum=2.0,
    )
    target_search_max_distance_m = _strict_number(
        pregrasp.get("target_search_max_distance_m", 3.00),
        "pregrasp_red_align.target_search_max_distance_m",
        minimum=0.01,
        maximum=5.0,
    )
    if (
        pregrasp.get("target_search_bilateral_enabled", False)
        and not pregrasp.get("target_search_until_found", False)
        and target_search_max_distance_m < 3.0 * target_search_each_side_m
    ):
        raise ConfigError(
            "pregrasp_red_align.target_search_max_distance_m must cover "
            "left search and full right sweep"
        )
    target_search_max_net_lateral_m = _strict_number(
        pregrasp.get("target_search_max_net_lateral_m", 1.05),
        "pregrasp_red_align.target_search_max_net_lateral_m",
        minimum=0.01,
        maximum=2.0,
    )
    if target_search_max_net_lateral_m < target_search_each_side_m:
        raise ConfigError(
            "pregrasp_red_align.target_search_max_net_lateral_m must be >= "
            "target_search_each_side_m"
        )
    _strict_number(
        pregrasp.get("target_search_min_progress_m", 0.015),
        "pregrasp_red_align.target_search_min_progress_m",
        minimum=0.0001,
        maximum=0.20,
    )
    _strict_integer(
        pregrasp.get("target_search_max_stalled_pulses", 3),
        "pregrasp_red_align.target_search_max_stalled_pulses",
        minimum=1,
    )
    _strict_integer(
        pregrasp.get("target_search_odometry_stall_recovery_attempts", 2),
        "pregrasp_red_align.target_search_odometry_stall_recovery_attempts",
        minimum=0,
        maximum=10,
    )
    _strict_number(
        pregrasp.get("target_search_odometry_stall_recovery_settle_seconds", 0.30),
        "pregrasp_red_align.target_search_odometry_stall_recovery_settle_seconds",
        minimum=0.0,
        maximum=5.0,
    )
    _strict_number(
        pregrasp.get("target_search_odometry_stall_recovery_pulse_seconds", 1.25),
        "pregrasp_red_align.target_search_odometry_stall_recovery_pulse_seconds",
        minimum=0.01,
        maximum=5.0,
    )
    target_search_front_target_m = _strict_number(
        pregrasp.get("target_search_front_target_m", 0.28),
        "pregrasp_red_align.target_search_front_target_m",
        minimum=0.05,
        maximum=2.0,
    )
    _strict_number(
        pregrasp.get("target_search_front_deadband_m", 0.015),
        "pregrasp_red_align.target_search_front_deadband_m",
        minimum=0.0,
        maximum=0.20,
    )
    _strict_number(
        pregrasp.get("target_search_front_hold_kp_s", 0.8),
        "pregrasp_red_align.target_search_front_hold_kp_s",
        minimum=0.0,
        maximum=5.0,
    )
    _strict_number(
        pregrasp.get("target_search_front_max_vx_mps", 0.025),
        "pregrasp_red_align.target_search_front_max_vx_mps",
        minimum=0.0,
        maximum=float(config["motion"]["max_vx"]),
    )
    target_search_front_edge_far_m = _strict_number(
        pregrasp.get("target_search_front_edge_far_m", 0.60),
        "pregrasp_red_align.target_search_front_edge_far_m",
        minimum=0.05,
        maximum=5.0,
    )
    if target_search_front_edge_far_m <= target_search_front_target_m:
        raise ConfigError(
            "pregrasp_red_align.target_search_front_edge_far_m must be "
            "greater than target_search_front_target_m"
        )
    _strict_number(
        pregrasp.get("target_search_front_edge_jump_m", 0.25),
        "pregrasp_red_align.target_search_front_edge_jump_m",
        minimum=0.01,
        maximum=5.0,
    )
    _strict_integer(
        pregrasp.get("target_search_front_edge_confirm_samples", 2),
        "pregrasp_red_align.target_search_front_edge_confirm_samples",
        minimum=1,
    )
    _strict_number(
        pregrasp.get("acquire_fine_max_strafe_distance_m", 0.15),
        "pregrasp_red_align.acquire_fine_max_strafe_distance_m",
        minimum=0.001,
        maximum=0.20,
    )
    _strict_integer(
        pregrasp.get("acquired_target_lost_frame_limit", 2),
        "pregrasp_red_align.acquired_target_lost_frame_limit",
        minimum=0,
        maximum=10,
    )
    preapproach_search_min_distance_m = _strict_number(
        pregrasp.get("preapproach_search_min_distance_m", 0.0),
        "pregrasp_red_align.preapproach_search_min_distance_m",
        minimum=0.0,
        maximum=5.0,
    )
    if preapproach_search_min_distance_m > target_search_max_distance_m:
        raise ConfigError(
            "pregrasp_red_align.preapproach_search_min_distance_m must not "
            "exceed target_search_max_distance_m"
        )
    center_band = pregrasp.get(
        "target_search_center_band",
        [0.40, 0.60],
    )
    if not isinstance(center_band, (list, tuple)) or len(center_band) != 2:
        raise ConfigError(
            "pregrasp_red_align.target_search_center_band must contain two numbers"
        )
    center_left = _strict_number(
        center_band[0],
        "pregrasp_red_align.target_search_center_band[0]",
        minimum=0.0,
        maximum=1.0,
    )
    center_right = _strict_number(
        center_band[1],
        "pregrasp_red_align.target_search_center_band[1]",
        minimum=0.0,
        maximum=1.0,
    )
    if center_left >= center_right:
        raise ConfigError(
            "pregrasp_red_align.target_search_center_band must be ordered left < right"
        )
    for key in (
        "forward_hold_kp_s",
        "max_vx_correction_mps",
        "forward_deadband_m",
        "max_forward_drift_m",
        "yaw_hold_kp_s",
        "max_wz_correction_rad_s",
        "yaw_deadband_deg",
        "max_yaw_drift_deg",
    ):
        _number(
            pregrasp.get(key),
            f"pregrasp_red_align.{key}",
            minimum=0.0,
        )
    wide_parallel = pregrasp.get("wide_parallel")
    if not isinstance(wide_parallel, Mapping):
        raise ConfigError("pregrasp_red_align.wide_parallel must be a mapping")
    for key in (
        "frames_per_measurement",
        "min_valid_frames",
        "tolerance_deg",
        "max_range_deg",
        "correction_speed_rad_s",
        "coarse_error_deg",
        "coarse_pulse_seconds",
        "fine_pulse_seconds",
        "settle_seconds",
        "max_corrections",
    ):
        _number(
            wide_parallel.get(key),
            f"pregrasp_red_align.wide_parallel.{key}",
            minimum=0.0,
        )
    if int(wide_parallel.get("positive_error_wz_sign", 0)) not in {-1, 1}:
        raise ConfigError(
            "pregrasp_red_align.wide_parallel.positive_error_wz_sign must be -1 or 1"
        )
    box_center = _require_mapping(config, "box_center_alignment")
    if not isinstance(
        box_center.get("placement_glyph_fallback_enabled"),
        bool,
    ):
        raise ConfigError(
            "box_center_alignment.placement_glyph_fallback_enabled must be a boolean"
        )
    frames_value = _number(
        box_center.get("frames_per_measurement"),
        "box_center_alignment.frames_per_measurement",
        minimum=1.0,
    )
    valid_frames_value = _number(
        box_center.get("min_valid_frames"),
        "box_center_alignment.min_valid_frames",
        minimum=1.0,
    )
    corrections_value = _number(
        box_center.get("max_corrections"),
        "box_center_alignment.max_corrections",
        minimum=0.0,
    )
    if any(
        value != int(value)
        for value in (frames_value, valid_frames_value, corrections_value)
    ):
        raise ConfigError(
            "box_center_alignment frame counts and max_corrections must be integers"
        )
    frames = int(frames_value)
    valid_frames = int(valid_frames_value)
    if valid_frames > frames:
        raise ConfigError(
            "box_center_alignment.min_valid_frames must be <= frames_per_measurement"
        )
    tracking_min_separators = _number(
        box_center.get("placement_tracking_min_separators"),
        "box_center_alignment.placement_tracking_min_separators",
        minimum=2.0,
        maximum=3.0,
    )
    if tracking_min_separators != int(tracking_min_separators):
        raise ConfigError(
            "box_center_alignment.placement_tracking_min_separators must be an integer"
        )
    for key, minimum, maximum in (
        ("max_center_range_fraction", 0.0, 1.0),
        ("tolerance_fraction", 0.0, 1.0),
        ("strafe_speed_mps", 0.000001, None),
        ("max_single_strafe_m", 0.000001, None),
        ("max_total_strafe_m", 0.000001, None),
        ("adjacent_box_spacing_m", 0.000001, None),
        ("pickup_m_per_pixel", 0.000001, None),
        ("settle_seconds", 0.0, None),
        ("placement_min_span_fraction", 0.0, 1.0),
        ("placement_min_center_gap_fraction", 0.0, 1.0),
        ("placement_tracking_max_scale_change_fraction", 0.0, 1.0),
        ("placement_tracking_max_residual_fraction", 0.0, 1.0),
        ("placement_tracking_min_motion_gain", 0.0, None),
        ("placement_tracking_max_motion_gain", 0.0, None),
        ("forward_hold_kp_s", 0.0, None),
        ("max_vx_correction_mps", 0.0, None),
        ("forward_deadband_m", 0.0, None),
        ("max_forward_drift_m", 0.0, None),
        ("yaw_hold_kp_s", 0.0, None),
        ("max_wz_correction_rad_s", 0.0, None),
        ("yaw_deadband_deg", 0.0, None),
        ("max_yaw_drift_deg", 0.0, None),
        ("placement_letter_min_confidence", 0.0, 1.0),
        ("placement_white_max_saturation", 0.0, 255.0),
        ("placement_white_min_value", 0.0, 255.0),
        ("placement_glyph_min_width_fraction", 0.0, 1.0),
        ("placement_glyph_max_width_fraction", 0.0, 1.0),
        ("placement_glyph_min_height_fraction", 0.0, 1.0),
        ("placement_glyph_max_height_fraction", 0.0, 1.0),
        ("placement_glyph_min_aspect", 0.000001, None),
        ("placement_glyph_max_aspect", 0.000001, None),
        ("placement_glyph_expand_x", 1.0, None),
        ("placement_glyph_expand_y", 1.0, None),
    ):
        _number(
            box_center.get(key),
            f"box_center_alignment.{key}",
            minimum=minimum,
            maximum=maximum,
        )
    if float(box_center["max_total_strafe_m"]) < float(
        box_center["max_single_strafe_m"]
    ):
        raise ConfigError(
            "box_center_alignment.max_total_strafe_m must be >= max_single_strafe_m"
        )
    if int(box_center.get("positive_error_strafe_sign", 0)) not in {-1, 1}:
        raise ConfigError(
            "box_center_alignment.positive_error_strafe_sign must be -1 or 1"
        )
    if float(box_center["placement_tracking_max_motion_gain"]) < float(
        box_center["placement_tracking_min_motion_gain"]
    ):
        raise ConfigError(
            "box_center_alignment placement tracking motion gain range is invalid"
        )
    placement_label_min_area = _strict_number(
        box_center.get("placement_label_min_area_fraction"),
        "box_center_alignment.placement_label_min_area_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    placement_label_max_area = _strict_number(
        box_center.get("placement_label_max_area_fraction"),
        "box_center_alignment.placement_label_max_area_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        placement_label_min_area <= 0.0
        or placement_label_min_area >= placement_label_max_area
    ):
        raise ConfigError(
            "box_center_alignment placement_label area fractions must satisfy "
            "0 < minimum < maximum <= 1"
        )
    for minimum_key, maximum_key in (
        (
            "placement_glyph_min_width_fraction",
            "placement_glyph_max_width_fraction",
        ),
        (
            "placement_glyph_min_height_fraction",
            "placement_glyph_max_height_fraction",
        ),
        ("placement_glyph_min_aspect", "placement_glyph_max_aspect"),
    ):
        if float(box_center[minimum_key]) >= float(box_center[maximum_key]):
            raise ConfigError(
                "box_center_alignment glyph range is invalid: "
                f"{minimum_key} must be less than {maximum_key}"
            )
    roi = box_center.get("placement_roi")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ConfigError("box_center_alignment.placement_roi must contain four values")
    roi_values = [
        _number(value, f"box_center_alignment.placement_roi[{index}]", minimum=0.0, maximum=1.0)
        for index, value in enumerate(roi)
    ]
    if roi_values[0] >= roi_values[2] or roi_values[1] >= roi_values[3]:
        raise ConfigError("box_center_alignment.placement_roi must have positive area")
    glyph_roi = box_center.get("placement_glyph_roi")
    if not isinstance(glyph_roi, (list, tuple)) or len(glyph_roi) != 4:
        raise ConfigError(
            "box_center_alignment.placement_glyph_roi must contain four values"
        )
    glyph_roi_values = [
        _number(
            value,
            f"box_center_alignment.placement_glyph_roi[{index}]",
            minimum=0.0,
            maximum=1.0,
        )
        for index, value in enumerate(glyph_roi)
    ]
    if (
        glyph_roi_values[0] >= glyph_roi_values[2]
        or glyph_roi_values[1] >= glyph_roi_values[3]
    ):
        raise ConfigError(
            "box_center_alignment.placement_glyph_roi must have positive area"
        )
    fallback = box_center.get("fallback_offsets_m")
    if not isinstance(fallback, Mapping):
        raise ConfigError("box_center_alignment.fallback_offsets_m must be a mapping")
    for letter in ("A", "B", "C", "D"):
        _number(
            fallback.get(letter),
            f"box_center_alignment.fallback_offsets_m.{letter}",
        )
    placement_navigation = _require_mapping(config, "placement_letter_navigation")
    if not isinstance(placement_navigation.get("enabled"), bool):
        raise ConfigError("placement_letter_navigation.enabled must be a boolean")
    letter_order = placement_navigation.get("letter_order")
    if not isinstance(letter_order, (list, tuple)) or list(letter_order) != [
        "A",
        "B",
        "C",
        "D",
    ]:
        raise ConfigError(
            "placement_letter_navigation.letter_order must be A,B,C,D"
        )
    for key, minimum, maximum in (
        ("letter_min_confidence", 0.0, 1.0),
        ("forward_speed_mps", 0.000001, None),
        ("front_stop_distance_m", 0.28, 4.50),
        ("forward_budget_m", 0.000001, None),
        ("search_step_m", 0.000001, None),
        ("min_center_correction_m", 0.000001, None),
        ("max_center_correction_m", 0.000001, None),
        ("center_gain_m_per_fraction", 0.000001, None),
        ("lateral_speed_mps", 0.000001, None),
        ("fine_strafe_distance_tolerance_m", 0.000001, 0.029999),
        ("max_lateral_search_m", 0.000001, None),
        ("lateral_search_each_side_m", 0.000001, None),
        ("center_tolerance_fraction", 0.0, 0.50),
        ("final_approach_distance_m", 0.0, 0.0),
        ("final_approach_step_m", 0.0, 0.0),
        ("letter_spacing_m", 0.000001, 2.0),
        ("max_anchor_jump_m", 0.000001, 2.0),
        ("target_memory_max_lateral_m", 0.000001, 2.0),
        ("target_memory_max_forward_m", 0.000001, 2.0),
        ("target_memory_fraction_per_m", 0.000001, 10.0),
        ("ultrasound_jump_reject_m", 0.000001, 2.0),
        ("final_ultrasound_min_m", 0.03, 4.50),
        ("final_ultrasound_max_m", 0.03, 4.50),
        ("ultrasound_odom_consistency_tolerance_m", 0.0, 1.0),
        ("ultrasound_stuck_value_m", 0.03, 4.5),
        ("ultrasound_stuck_tolerance_m", 0.0, 0.20),
        ("approach_filter_warmup_s", 0.0, 2.0),
        ("visual_row_preflight_trigger_m", 0.28, 4.50),
        ("visual_ultrasound_start_tolerance_m", 0.0, 2.0),
        ("strafe_forward_hold_kp_s", 0.0, None),
        ("strafe_max_vx_correction_mps", 0.0, 0.05),
        ("strafe_forward_deadband_m", 0.0, None),
        ("search_hold_capture_timeout_s", 0.1, 10.0),
        ("search_hold_capture_max_spread_m", 0.0, 1.0),
        ("search_hold_capture_min_m", 0.03, 4.50),
        ("search_hold_capture_max_m", 0.03, 4.50),
        ("search_hold_boundary_delta_m", 0.01, 2.0),
        ("search_hold_restore_speed_mps", 0.000001, 0.10),
        ("search_hold_restore_min_step_m", 0.0, 0.10),
        ("search_hold_restore_max_step_m", 0.000001, 0.30),
        ("strafe_max_forward_drift_m", 0.0, 1.0),
        ("strafe_yaw_hold_kp_s", 0.0, None),
        ("strafe_max_wz_correction_rad_s", 0.0, 1.0),
        ("strafe_yaw_deadband_deg", 0.0, 180.0),
        ("strafe_max_yaw_drift_deg", 0.0, 180.0),
        ("motion_stall_timeout_s", 0.1, 30.0),
        ("motion_stall_min_progress_m", 0.000001, 0.50),
        ("motion_recovery_pause_s", 0.0, 10.0),
        ("motion_recovery_speed_mps", 0.000001, 0.30),
        ("approach_slow_distance_m", 0.28, 4.50),
        ("approach_creep_distance_m", 0.28, 4.50),
        ("approach_slow_speed_mps", 0.000001, 0.30),
        ("approach_creep_speed_mps", 0.000001, 0.30),
        ("image_timeout_s", 0.000001, None),
        ("total_timeout_s", 0.0, None),
    ):
        _strict_number(
            placement_navigation.get(key),
            f"placement_letter_navigation.{key}",
            minimum=minimum,
            maximum=maximum,
        )
    if float(placement_navigation["min_center_correction_m"]) > float(
        placement_navigation["max_center_correction_m"]
    ):
        raise ConfigError(
            "placement_letter_navigation center correction range is inverted"
        )
    if float(placement_navigation["approach_creep_distance_m"]) > float(
        placement_navigation["approach_slow_distance_m"]
    ):
        raise ConfigError(
            "placement_letter_navigation approach creep distance must not "
            "exceed slow distance"
        )
    if not (
        float(placement_navigation["approach_creep_speed_mps"])
        <= float(placement_navigation["approach_slow_speed_mps"])
        <= float(placement_navigation["forward_speed_mps"])
    ):
        raise ConfigError(
            "placement_letter_navigation approach speeds must satisfy "
            "creep <= slow <= forward"
        )
    for key in (
        "bilateral_search_enabled",
        "immediate_complete_on_target_detection",
        "cached_geometry_enabled",
    ):
        if not isinstance(placement_navigation.get(key), bool):
            raise ConfigError(
                f"placement_letter_navigation.{key} must be a boolean"
            )
    acquisition_center_band = placement_navigation.get("acquisition_center_band")
    if (
        not isinstance(acquisition_center_band, (list, tuple))
        or len(acquisition_center_band) != 2
    ):
        raise ConfigError(
            "placement_letter_navigation.acquisition_center_band must contain two numbers"
        )
    zero_progress_reverse_count = placement_navigation.get(
        "strafe_zero_progress_reverse_count"
    )
    if (
        isinstance(zero_progress_reverse_count, bool)
        or not isinstance(zero_progress_reverse_count, int)
        or zero_progress_reverse_count < 1
    ):
        raise ConfigError(
            "placement_letter_navigation.strafe_zero_progress_reverse_count "
            "must be a positive integer"
        )
    search_hold_capture_samples = placement_navigation.get(
        "search_hold_capture_samples"
    )
    if (
        isinstance(search_hold_capture_samples, bool)
        or not isinstance(search_hold_capture_samples, int)
        or search_hold_capture_samples < 3
    ):
        raise ConfigError(
            "placement_letter_navigation.search_hold_capture_samples "
            "must be an integer of at least 3"
        )
    search_hold_restore_attempts = placement_navigation.get(
        "search_hold_restore_attempts"
    )
    if (
        isinstance(search_hold_restore_attempts, bool)
        or not isinstance(search_hold_restore_attempts, int)
        or search_hold_restore_attempts < 1
    ):
        raise ConfigError(
            "placement_letter_navigation.search_hold_restore_attempts "
            "must be a positive integer"
        )
    final_min_m = float(placement_navigation["final_ultrasound_min_m"])
    final_max_m = float(placement_navigation["final_ultrasound_max_m"])
    if not final_min_m <= float(placement_navigation["front_stop_distance_m"]) <= final_max_m:
        raise ConfigError(
            "placement final ultrasound window must contain front_stop_distance_m"
        )
    if float(placement_navigation["search_hold_capture_min_m"]) >= float(
        placement_navigation["search_hold_capture_max_m"]
    ):
        raise ConfigError(
            "placement search hold capture range must have positive width"
        )
    acquisition_left = _strict_number(
        acquisition_center_band[0],
        "placement_letter_navigation.acquisition_center_band[0]",
        minimum=0.0,
        maximum=1.0,
    )
    acquisition_right = _strict_number(
        acquisition_center_band[1],
        "placement_letter_navigation.acquisition_center_band[1]",
        minimum=0.0,
        maximum=1.0,
    )
    if acquisition_left >= acquisition_right:
        raise ConfigError(
            "placement_letter_navigation.acquisition_center_band must be ordered left < right"
        )
    if (
        placement_navigation["bilateral_search_enabled"]
        and float(placement_navigation["max_lateral_search_m"])
        < 3.0 * float(placement_navigation["lateral_search_each_side_m"])
    ):
        raise ConfigError(
            "placement_letter_navigation.max_lateral_search_m must cover "
            "left search and full right sweep"
        )
    for key in (
        "required_center_frames",
        "capture_retries",
        "target_vote_window",
        "target_min_votes",
        "target_memory_max_misses",
        "ultrasound_filter_samples",
        "ultrasound_stable_samples",
        "ultrasound_jump_confirm_samples",
        "visual_row_preflight_attempts",
        "motion_stall_retries",
    ):
        value = placement_navigation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"placement_letter_navigation.{key} must be an integer >= 1"
            )
    if placement_navigation["target_min_votes"] > placement_navigation["target_vote_window"]:
        raise ConfigError(
            "placement_letter_navigation.target_min_votes must be <= target_vote_window"
        )
    if (
        placement_navigation["target_min_votes"] * 2
        <= placement_navigation["target_vote_window"]
    ):
        raise ConfigError(
            "placement_letter_navigation.target_min_votes must form a strict majority"
        )
    physical_left_sign = placement_navigation.get("physical_left_strafe_sign")
    if (
        isinstance(physical_left_sign, bool)
        or not isinstance(physical_left_sign, int)
        or physical_left_sign not in {-1, 1}
    ):
        raise ConfigError(
            "placement_letter_navigation.physical_left_strafe_sign must be -1 or 1"
        )
    run_log_dir = placement_navigation.get("run_log_dir")
    if not isinstance(run_log_dir, str) or not run_log_dir.strip():
        raise ConfigError(
            "placement_letter_navigation.run_log_dir must be a non-empty string"
        )
    pickup_transfer = _require_mapping(config, "pickup_transfer")
    if not isinstance(pickup_transfer.get("enabled"), bool):
        raise ConfigError("pickup_transfer.enabled must be a boolean")
    if pickup_transfer["enabled"] and not placement_navigation["enabled"]:
        raise ConfigError(
            "pickup_transfer.enabled requires "
            "placement_letter_navigation.enabled"
        )
    transfer_values: dict[str, float] = {}
    for key, minimum, maximum in (
        ("retreat_target_front_m", 0.01, None),
        ("retreat_stop_threshold_m", 0.01, None),
        ("retreat_max_front_m", 0.01, None),
        ("retreat_speed_mps", 0.000001, None),
        ("retreat_timeout_s", 0.1, None),
        ("retreat_max_odom_m", 0.01, None),
        ("retreat_stuck_front_value_m", 0.01, None),
        ("retreat_stuck_front_tolerance_m", 0.0, None),
        ("retreat_stuck_front_min_samples", 1.0, None),
        ("retreat_odom_fallback_target_m", 0.01, None),
        ("retreat_lateral_hold_kp_s", 0.0, None),
        ("retreat_max_vy_correction_mps", 0.0, None),
        ("retreat_lateral_deadband_m", 0.0, None),
        ("retreat_max_lateral_drift_m", 0.0, None),
        ("yaw_hold_kp_s", 0.0, None),
        ("max_wz_correction_rad_s", 0.0, None),
        ("yaw_deadband_deg", 0.0, None),
        ("max_yaw_drift_deg", 0.0, None),
        ("departure_tolerance_fraction", 0.0, 1.0),
        ("arrival_tolerance_fraction", 0.0, 1.0),
        ("transfer_distance_m", 0.01, None),
        ("lane_strafe_speed_mps", 0.000001, None),
        ("lane_forward_hold_kp_s", 0.0, None),
        ("lane_max_vx_correction_mps", 0.0, None),
        ("lane_forward_deadband_m", 0.0, None),
        ("lane_max_forward_drift_m", 0.0, None),
        ("max_recorded_lane_strafe_m", 0.0, None),
    ):
        transfer_values[key] = _number(
            pickup_transfer.get(key),
            f"pickup_transfer.{key}",
            minimum=minimum,
            maximum=maximum,
        )
    if not (
        transfer_values["retreat_stop_threshold_m"]
        <= transfer_values["retreat_target_front_m"]
        <= transfer_values["retreat_max_front_m"]
    ):
        raise ConfigError(
            "pickup_transfer retreat distances must satisfy "
            "stop_threshold <= target <= maximum"
        )
    if transfer_values["retreat_stuck_front_min_samples"] != int(
        transfer_values["retreat_stuck_front_min_samples"]
    ):
        raise ConfigError(
            "pickup_transfer.retreat_stuck_front_min_samples must be an integer"
        )
    if transfer_values["retreat_odom_fallback_target_m"] >= transfer_values[
        "retreat_max_odom_m"
    ]:
        raise ConfigError(
            "pickup_transfer.retreat_odom_fallback_target_m must be below retreat_max_odom_m"
        )
    lane_offsets = pickup_transfer.get("lane_offsets_m")
    if not isinstance(lane_offsets, Mapping):
        raise ConfigError("pickup_transfer.lane_offsets_m must be a mapping")
    for letter in ("A", "B", "C", "D"):
        offset = _number(
            lane_offsets.get(letter),
            f"pickup_transfer.lane_offsets_m.{letter}",
        )
        if abs(offset) > transfer_values["max_recorded_lane_strafe_m"]:
            raise ConfigError(
                f"pickup_transfer.lane_offsets_m.{letter} exceeds recorded lane limit"
            )
    placement_yaw = _require_mapping(config, "placement_yaw_alignment")
    placement_yaw_values: dict[str, float] = {}
    for key, minimum, maximum in (
        ("frames_per_measurement", 1.0, None),
        ("min_valid_frames", 1.0, None),
        ("tolerance_deg", 0.0, None),
        ("max_range_deg", 0.0, None),
        ("correction_speed_rad_s", 0.000001, None),
        ("coarse_error_deg", 0.0, None),
        ("coarse_pulse_seconds", 0.0, None),
        ("fine_pulse_seconds", 0.0, None),
        ("error_fraction_per_correction", 0.0, 1.0),
        ("motion_response_gain", 0.000001, None),
        ("min_pulse_seconds", 0.0, None),
        ("max_pulse_seconds", 0.0, None),
        ("settle_seconds", 0.0, None),
        ("max_corrections", 0.0, None),
        ("min_row_span_fraction", 0.0, 1.0),
    ):
        placement_yaw_values[key] = _number(
            placement_yaw.get(key),
            f"placement_yaw_alignment.{key}",
            minimum=minimum,
            maximum=maximum,
        )
    for key in ("frames_per_measurement", "min_valid_frames", "max_corrections"):
        if placement_yaw_values[key] != int(placement_yaw_values[key]):
            raise ConfigError(f"placement_yaw_alignment.{key} must be an integer")
    if placement_yaw_values["min_valid_frames"] > placement_yaw_values["frames_per_measurement"]:
        raise ConfigError(
            "placement_yaw_alignment.min_valid_frames must be <= frames_per_measurement"
        )
    if placement_yaw_values["max_pulse_seconds"] < placement_yaw_values["min_pulse_seconds"]:
        raise ConfigError(
            "placement_yaw_alignment.max_pulse_seconds must be >= min_pulse_seconds"
        )
    if int(placement_yaw.get("positive_error_wz_sign", 0)) not in {-1, 1}:
        raise ConfigError(
            "placement_yaw_alignment.positive_error_wz_sign must be -1 or 1"
        )
    _validate_route(config)


def load_config(config_dir: Path | None = None, *, strict: bool = True) -> Dict[str, Any]:
    config_dir = config_dir or ROOT / "config"
    config_dir = Path(config_dir)
    config = _deep_merge(DEFAULT_FIELD, DEFAULT_ROBOT)
    field_data = _load_config_file(config_dir / "field.yaml", required=strict)
    robot_data = _load_config_file(config_dir / "robot.yaml", required=strict)
    if strict and "scripted_route" not in field_data:
        raise ConfigError(f"field config must define scripted_route: {config_dir / 'field.yaml'}")
    for name in (
        "network",
        "ros2",
        "camera",
        "motion",
        "navigation",
        "safety",
        "fault_hold",
        "startup_avoidance",
        "vision",
        "inspection",
        "arm",
        "box_center_alignment",
        "placement_letter_navigation",
        "pickup_transfer",
        "audio",
    ):
        if strict and name not in robot_data:
            raise ConfigError(f"robot config must define {name}: {config_dir / 'robot.yaml'}")
    config = _deep_merge(config, field_data)
    config = _deep_merge(config, robot_data)
    if strict:
        validate_config(config)
    return config
