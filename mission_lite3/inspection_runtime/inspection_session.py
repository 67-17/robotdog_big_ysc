"""Accumulate stable single-area observations into one four-area inspection."""


VALID_AREAS = ("A", "B", "C", "D")
VALID_STATES = ("normal", "abnormal")


class InspectionSession:
    def __init__(self):
        self._areas = {}

    def update(self, stable_result):
        letter = str(stable_result.get("letter", "")).upper()
        state = stable_result.get("state")
        if letter not in VALID_AREAS or state not in VALID_STATES:
            raise ValueError("stable result must contain a valid letter and state")
        area_result = {
            "area": letter,
            "state": state,
            "description": stable_result.get("description", ""),
        }
        if stable_result.get("source_image"):
            area_result["source_image"] = str(stable_result["source_image"])
        self._areas[letter] = area_result
        return self.result()

    def result(self):
        observed = [self._areas[letter] for letter in VALID_AREAS if letter in self._areas]
        abnormal_areas = [
            item["area"] for item in observed if item["state"] == "abnormal"
        ]
        unknown_areas = [letter for letter in VALID_AREAS if letter not in self._areas]
        if unknown_areas:
            count_check = "incomplete"
        else:
            normal_count = sum(item["state"] == "normal" for item in observed)
            abnormal_count = sum(item["state"] == "abnormal" for item in observed)
            count_check = "pass" if normal_count == 2 and abnormal_count == 2 else "fail"
        return {
            "ok": True,
            "areas": observed,
            "abnormal_areas": abnormal_areas,
            "unknown_areas": unknown_areas,
            "count_check": count_check,
            "ready": not unknown_areas and count_check == "pass",
        }

    def reset(self):
        self._areas.clear()
