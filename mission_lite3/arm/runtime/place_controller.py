import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


VALID_SLOTS = ("A", "B", "C", "D")
JOINT_KEYS = ("b", "s", "e", "w")
DEFAULT_SPEED = 40.0
DEFAULT_ACCELERATION = 40.0
# On this arm, decreasing e raises the elbow.
DEFAULT_PLACE_PRE_LIFT_E_DELTA_DEG = -30.0
DEFAULT_PLACE_LIFT_E_DEG = -70.0
DEFAULT_PLACE_S_FORWARD_DELTA_DEG = 20.0
DEFAULT_PLACE_RELEASE_S_DEG = 70.0
DEFAULT_PLACE_MAX_S_DEG = 60.0
DEFAULT_PLACE_MIN_S_PROGRESS_DEG = 0.05
DEFAULT_RETREAT_CLEARANCE_S_DEG = 50.0
DEFAULT_RETREAT_SHOULDER_FRACTION_BEFORE_ELBOW = 0.5
DEFAULT_RETREAT_CLEARANCE_SPEED = 30.0
DEFAULT_RETREAT_CLEARANCE_ACCELERATION = 30.0
DEFAULT_CARDBOARD_CENTER_RATIO = 0.43
DEFAULT_CARDBOARD_LOWER_CENTER_RATIO = 0.65
DEFAULT_RETREAT_JOINTS_DEG = {
    "b": -1.4941406024222599,
    "s": -89.9121093745983,
    "e": 87.53906251523503,
    "w": -1.5820312395755176,
}


@dataclass
class PlaceResult:
    ok: bool
    stage: str
    reason: str = ""
    slot: Optional[str] = None
    object_held: bool = True
    released: bool = False
    plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _validate_joints(value: Any, name: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a joint mapping")
    missing = [joint for joint in JOINT_KEYS if joint not in value]
    if missing:
        raise ValueError(f"{name}缂哄皯鍏宠妭: {', '.join(missing)}")
    return {joint: _require_number(value[joint], f"{name}.{joint}") for joint in JOINT_KEYS}


def validate_place_reference(reference: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(reference, Mapping):
        raise ValueError("place reference must be a mapping")
    if int(reference.get("schema_version", 0)) != 1:
        raise ValueError("place reference schema_version must be 1")
    slots = reference.get("slots")
    if not isinstance(slots, Mapping):
        raise ValueError("place reference is missing slots")
    if set(slots) != set(VALID_SLOTS):
        raise ValueError("place reference must contain A/B/C/D slots")

    normalized = {"schema_version": 1, "slots": {}}
    for slot in VALID_SLOTS:
        item = slots[slot]
        if not isinstance(item, Mapping):
            raise ValueError(f"slot {slot} config must be a mapping")
        normalized["slots"][slot] = {
            "pre_place_joints_deg": _validate_joints(
                item.get("pre_place_joints_deg"),
                f"slots.{slot}.pre_place_joints_deg",
            ),
            "release_gripper_h": _require_number(
                item.get("release_gripper_h"),
                f"slots.{slot}.release_gripper_h",
            ),
            "retreat_joints_deg": _validate_joints(
                item.get("retreat_joints_deg"),
                f"slots.{slot}.retreat_joints_deg",
            ),
        }
    return normalized


def load_place_reference(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_place_reference(json.load(handle))


def default_place_reference() -> Dict[str, Any]:
    slots = {}
    for slot, base in zip(VALID_SLOTS, (0.0, 8.0, -8.0, 0.0)):
        slots[slot] = {
            "pre_place_joints_deg": {"b": base, "s": -12.0, "e": 20.0, "w": 0.0},
            "release_gripper_h": -45.0,
            "retreat_joints_deg": dict(DEFAULT_RETREAT_JOINTS_DEG),
        }
    slots["D"]["pre_place_joints_deg"]["s"] = -18.0
    return validate_place_reference({"schema_version": 1, "slots": slots})


class PlaceController:
    def __init__(
        self,
        reference: Mapping[str, Any],
        motion: Any,
        *,
        place_vision: Optional[Any] = None,
        spd: float = DEFAULT_SPEED,
        acc: float = DEFAULT_ACCELERATION,
    ):
        self.reference = validate_place_reference(reference)
        self.motion = motion
        self.place_vision = place_vision
        self.spd = float(spd)
        self.acc = float(acc)

    def _reference_slot(self, slot: str) -> str:
        return slot if slot in VALID_SLOTS else VALID_SLOTS[0]

    def _current_pose_or_pre_place(self, slot: str) -> Dict[str, float]:
        item = self.reference["slots"][self._reference_slot(slot)]
        current_pose_fn = getattr(self.motion, "current_pose_degrees", None)
        if callable(current_pose_fn):
            try:
                pose = current_pose_fn()
                return {
                    joint: float(pose.get(joint, item["pre_place_joints_deg"][joint]))
                    for joint in JOINT_KEYS
                }
            except (KeyError, TypeError, ValueError):
                pass
        return dict(item["pre_place_joints_deg"])

    def _release_pose(self, slot: str) -> Dict[str, float]:
        pose = self._current_pose_or_pre_place(slot)
        pose.update(
            {
                "s": DEFAULT_PLACE_RELEASE_S_DEG,
                "e": DEFAULT_PLACE_LIFT_E_DEG,
            }
        )
        return pose

    def _pre_lift_pose(self, slot: str) -> Dict[str, float]:
        pose = self._current_pose_or_pre_place(slot)
        pose["e"] = pose["e"] + DEFAULT_PLACE_PRE_LIFT_E_DELTA_DEG
        return pose

    def _retreat_shoulder_midpoint(self, slot: str) -> float:
        target = self.reference["slots"][self._reference_slot(slot)][
            "retreat_joints_deg"
        ]["s"]
        return DEFAULT_RETREAT_CLEARANCE_S_DEG + (
            target - DEFAULT_RETREAT_CLEARANCE_S_DEG
        ) * DEFAULT_RETREAT_SHOULDER_FRACTION_BEFORE_ELBOW

    def _cardboard_seen(self, sample: Mapping[str, Any]) -> bool:
        return (
            float(sample.get("center_ratio", 0.0)) >= DEFAULT_CARDBOARD_CENTER_RATIO
            and float(sample.get("lower_center_ratio", 0.0))
            >= DEFAULT_CARDBOARD_LOWER_CENTER_RATIO
        )

    def _detect_cardboard(self) -> Dict[str, Any]:
        if self.place_vision is None:
            return {
                "ok": False,
                "reason": "place cardboard vision is not configured",
                "center_ratio": 0.0,
                "lower_center_ratio": 0.0,
            }
        sample = dict(self.place_vision.detect_cardboard())
        sample["ok"] = self._cardboard_seen(sample)
        sample.setdefault("center_threshold", DEFAULT_CARDBOARD_CENTER_RATIO)
        sample.setdefault("lower_center_threshold", DEFAULT_CARDBOARD_LOWER_CENTER_RATIO)
        return sample

    def _build_plan(self, slot: str) -> List[Dict[str, Any]]:
        item = self.reference["slots"][self._reference_slot(slot)]
        return [
            {
                "stage": "LIFT_E_BEFORE_PLACE_RELEASE",
                "joints_deg": self._pre_lift_pose(slot),
                "reason": "lift elbow before moving shoulder and elbow together",
            },
            {
                "stage": "MOVE_TO_PLACE_RELEASE",
                "joints_deg": self._release_pose(slot),
                "reason": "move shoulder and elbow together to fixed release pose",
            },
            {
                "stage": "OPEN_GRIPPER",
                "release_gripper_h": item["release_gripper_h"],
            },
            {
                "stage": "MOVE_S_TO_RETREAT_CLEARANCE",
                "joints_deg": {"s": DEFAULT_RETREAT_CLEARANCE_S_DEG},
                "spd": DEFAULT_RETREAT_CLEARANCE_SPEED,
                "acc": DEFAULT_RETREAT_CLEARANCE_ACCELERATION,
                "reason": "move shoulder to clearance angle before ordered retreat",
            },
            {
                "stage": "RETRACT_SHOULDER_TO_HALF",
                "joints_deg": {"s": self._retreat_shoulder_midpoint(slot)},
                "spd": DEFAULT_RETREAT_CLEARANCE_SPEED,
                "acc": DEFAULT_RETREAT_CLEARANCE_ACCELERATION,
                "reason": "retract shoulder halfway before moving elbow",
            },
            {
                "stage": "RETRACT_SHOULDER_AND_ELBOW_TO_MOVING_POSE",
                "joints_deg": {
                    "s": item["retreat_joints_deg"]["s"],
                    "e": item["retreat_joints_deg"]["e"],
                },
                "spd": DEFAULT_RETREAT_CLEARANCE_SPEED,
                "acc": DEFAULT_RETREAT_CLEARANCE_ACCELERATION,
                "reason": "finish shoulder and elbow together after shoulder reaches halfway",
            },
            {
                "stage": "COMPLETE_RETREAT_TO_MOVING_POSE",
                "joints_deg": {
                    "b": item["retreat_joints_deg"]["b"],
                    "w": item["retreat_joints_deg"]["w"],
                },
                "reason": "finish base and wrist after shoulder and elbow are safe",
            },
        ]

    def _is_nonfatal_pre_lift_error(self, exc: Exception) -> bool:
        message = str(exc)
        return (
            "e关节已停止但未到达目标" in message
            or "e joint stopped but did not reach target" in message
        )

    def place(self, slot: str, *, object_held: bool, dry_run: bool = False) -> PlaceResult:
        slot = str(slot).upper()
        plan = self._build_plan(slot)
        if dry_run:
            return PlaceResult(True, "DRY_RUN", slot=slot, object_held=True, plan=plan)

        item = self.reference["slots"][self._reference_slot(slot)]
        (
            pre_lift_step,
            place_step,
            release_step,
            retreat_clearance_step,
            retract_shoulder_half_step,
            retract_shoulder_elbow_step,
            complete_retreat_step,
        ) = plan
        executed_plan: List[Dict[str, Any]] = []

        for step in (pre_lift_step, place_step):
            try:
                self.motion.move_joints(
                    step["joints_deg"],
                    spd=self.spd,
                    acc=self.acc,
                )
            except RuntimeError as exc:
                if not self._is_nonfatal_pre_lift_error(exc):
                    return PlaceResult(
                        False,
                        step["stage"],
                        reason=f"{step['stage']} failed: {exc}",
                        slot=slot,
                        object_held=bool(object_held),
                        released=False,
                        plan=executed_plan,
                    )
                step = dict(step)
                step["warning"] = str(exc)
                step["continued_after_stable_stop"] = True
            except Exception as exc:
                return PlaceResult(
                    False,
                    step["stage"],
                    reason=f"{step['stage']} failed: {exc}",
                    slot=slot,
                    object_held=bool(object_held),
                    released=False,
                    plan=executed_plan,
                )
            executed_plan.append(step)

        try:
            self.motion.open_gripper(
                angle=item["release_gripper_h"],
                spd=self.spd,
                acc=self.acc,
            )
        except Exception as exc:
            return PlaceResult(
                False,
                release_step["stage"],
                reason=f"{release_step['stage']} failed: {exc}",
                slot=slot,
                object_held=bool(object_held),
                released=False,
                plan=executed_plan,
            )
        executed_plan.append(release_step)

        for step in (
            retreat_clearance_step,
            retract_shoulder_half_step,
            retract_shoulder_elbow_step,
            complete_retreat_step,
        ):
            try:
                self.motion.move_joints(
                    step["joints_deg"],
                    spd=float(step.get("spd", self.spd)),
                    acc=float(step.get("acc", self.acc)),
                )
            except Exception as exc:
                return PlaceResult(
                    False,
                    step["stage"],
                    reason=f"{step['stage']} failed: {exc}",
                    slot=slot,
                    object_held=False,
                    released=True,
                    plan=executed_plan,
                )
            executed_plan.append(step)
        return PlaceResult(
            True,
            "DONE",
            slot=slot,
            object_held=False,
            released=True,
            plan=executed_plan,
        )
