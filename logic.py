"""Strict frame contracts and circular, confidence-gated per-Run tracking."""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import re
import time
import uuid

DEFAULTS = {"allowed_root": "", "source_id": "camera", "tier": "fast", "device": "cpu",
    "abstention": "standard", "min_confidence": 0.65, "smoothing_window": 5, "change_threshold_deg": 10,
    "min_interval_ms": 1000, "max_age_ms": 10000, "reset_after_sec": 60, "max_sources": 8,
    "max_bytes": 10485760, "max_pixels": 16777216, "timeout_sec": 30,
    "download_missing": True, "download_timeout_sec": 180, "threads": 2}


class OrientationError(ValueError):
    def __init__(self, message, code="error"):
        super().__init__(message)
        self.code = code


class Cancelled(OrientationError):
    pass


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw) - set(DEFAULTS) - {"execution", "runtime_path", "runtime_path_label"}:
        raise OrientationError("Unknown RightWayUp setting.")
    cfg = {**DEFAULTS, **{key: value for key, value in raw.items() if key in DEFAULTS}}
    if not isinstance(cfg["allowed_root"], str) or len(cfg["allowed_root"]) > 4096 or any(ord(c)<32 for c in cfg["allowed_root"]):
        raise OrientationError("Choose a valid permitted image directory.")
    if not identifier(cfg["source_id"]):
        raise OrientationError("Source ID must contain 1–128 ASCII letters, digits, underscores or hyphens.")
    for key, options in (("tier", ("nano", "fast", "max")), ("device", ("cpu", "cuda")), ("abstention", ("standard", "strict"))):
        if cfg[key] not in options:
            raise OrientationError("Choose a supported tier, device and abstention policy.")
    if type(cfg["download_missing"]) is not bool:
        raise OrientationError("Download choice must be a boolean.")
    for key, low, high in (("smoothing_window", 1, 30), ("min_interval_ms", 0, 60000), ("max_age_ms", 100, 600000),
        ("reset_after_sec", 1, 3600), ("max_sources", 1, 32), ("max_bytes", 1024, 20971520),
        ("max_pixels", 65536, 16777216), ("timeout_sec", 1, 120), ("download_timeout_sec", 10, 600), ("threads", 1, 8)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise OrientationError(key + " is outside its integer limits.")
    for key, low, high in (("min_confidence", 0, 1), ("change_threshold_deg", 0.1, 180)):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or not low <= cfg[key] <= high:
            raise OrientationError(key + " is outside its numeric limits.")
    return cfg


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise OrientationError("Duplicate JSON properties are not accepted.")
            result[key] = value
        return result
    try:
        if len(raw) > 16384:
            raise ValueError()
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise OrientationError("Expected a bounded frame JSON object.") from None


def frame(value, cfg, now=None):
    now = time.time() if now is None else now
    if isinstance(value, str) and value.lstrip().startswith("{"):
        value = document(value)
    if isinstance(value, str):
        value = {"path": value}
    if not isinstance(value, Mapping) or set(value) - {"path", "source_id", "timestamp", "frame_id", "sha256"}:
        raise OrientationError("Frame fields: path, source_id, timestamp, frame_id and optional sha256.")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip() or len(path) > 4096 or "://" in path or path.startswith("data:") or any(ord(c)<32 for c in path):
        raise OrientationError("A local image path is required, not a URL or encoded image.")
    source, ident = value.get("source_id", cfg["source_id"]), value.get("frame_id", uuid.uuid4().hex)
    if not identifier(source) or not identifier(ident):
        raise OrientationError("Invalid source or frame ID.")
    stamp = value.get("timestamp", datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="microseconds"))
    try:
        if not isinstance(stamp, str) or len(stamp) > 64:
            raise ValueError()
        date = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError()
        epoch = date.timestamp()
        if epoch > now + 5:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise OrientationError("timestamp must be an ISO-8601 date with timezone, not over five seconds in the future.") from None
    digest = value.get("sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise OrientationError("sha256 must be a lowercase SHA-256 digest.")
    result = {"path": path, "source_id": source, "frame_id": ident, "timestamp": date.astimezone(timezone.utc).isoformat()}
    if digest is not None:
        result["sha256"] = digest
    return result


def epoch(item):
    return datetime.fromisoformat(item["timestamp"]).timestamp()


def signed(angle):
    return (angle + 180) % 360 - 180


class Tracker:
    """One fixed baseline after warm-up, circular mean, hysteresis and source isolation."""
    def __init__(self, config):
        self.config, self.sources = config, {}

    def apply(self, item, prediction, now=None):
        now = time.time() if now is None else now
        cfg = self.config
        source, timestamp = item["source_id"], epoch(item)
        base = {"source_id": source, "frame_id": item["frame_id"], "timestamp": item["timestamp"],
            "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "angle_cw": None, "correction_ccw": None, "smoothed_angle_cw": None, "accepted": False}
        old = self.sources.get(source)
        if now - timestamp > cfg["max_age_ms"]/1000:
            return {"orientation": {**base, "state": "stale"}}
        if old and timestamp <= old["timestamp"]:
            return {"orientation": {**base, "state": "out_of_order"}}
        if old is None and len(self.sources) >= cfg["max_sources"]:
            return {"orientation": {**base, "state": "source_limit"}}
        angle, confidence = prediction.get("angle_cw"), prediction.get("confidence")
        if (type(angle) not in (int, float) or not math.isfinite(angle) or not 0 <= angle < 360 or
            type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1 or
            type(prediction.get("abstain")) is not bool):
            raise OrientationError("Invalid model prediction.", "invalid_prediction")
        if old is None or timestamp - old["last_accepted"] > cfg["reset_after_sec"]:
            old = {"samples": [], "baseline": None, "drifted": False, "last_accepted": timestamp}
            self.sources[source] = old
        old["timestamp"] = timestamp
        base.update({"raw_angle_cw": angle, "confidence": confidence, "model_abstained": prediction["abstain"],
            "abstention_threshold": prediction.get("abstention_threshold"), "model": prediction.get("model"),
            "source": prediction.get("source"), "elapsed_ms": prediction.get("elapsed_ms")})
        if prediction["abstain"] or confidence < cfg["min_confidence"]:
            old["samples"] = []
            return {"orientation": {**base, "state": "uncertain"}}
        old["samples"] = (old["samples"] + [angle])[-cfg["smoothing_window"]:]
        x = sum(math.cos(math.radians(value)) for value in old["samples"]) / len(old["samples"])
        y = sum(math.sin(math.radians(value)) for value in old["samples"]) / len(old["samples"])
        if math.hypot(x, y) < .5:
            return {"orientation": {**base, "state": "unstable"}}
        old["last_accepted"] = timestamp
        mean = math.degrees(math.atan2(y, x)) % 360
        base.update({"angle_cw": angle, "correction_ccw": mean, "smoothed_angle_cw": mean, "accepted": True,
            "sample_count": len(old["samples"]), "baseline_angle_cw": old["baseline"], "delta_cw": None})
        if len(old["samples"]) < cfg["smoothing_window"]:
            return {"orientation": {**base, "state": "warming"}}
        if old["baseline"] is None:
            old["baseline"] = mean
            base["baseline_angle_cw"] = mean
        delta = signed(mean-old["baseline"])
        event = None
        if not old["drifted"] and abs(delta) >= cfg["change_threshold_deg"]:
            old["drifted"] = True; event = "rotated"
        elif old["drifted"] and abs(delta) <= cfg["change_threshold_deg"]/2:
            old["drifted"] = False; event = "returned"
        base.update({"delta_cw": delta, "state": "drifted" if old["drifted"] else "stable"})
        output = {"orientation": base}
        if event:
            output["change"] = {**base, "event": event, "threshold_deg": cfg["change_threshold_deg"]}
        return output
