"""FB1/FB2/FB3/FB4: real ONNX, circular math, bounded latest-frame queue and cancellation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"tests"), str(Path(__file__).parent)]
from blocs.rightwayup import engine, logic
from blocs.rightwayup.assets import AssetError, Redirects, manifest, verified_bytes
from blocs.rightwayup.worker import decode, letterbox
from rightwayup_fixture import install_cache, picture


def refused(action):
    try:
        action()
    except (logic.OrientationError, AssetError, OSError):
        return
    raise AssertionError("Invalid data was accepted")


def event(stamp, source="camera", ident=None):
    return {"path": "frame.png", "source_id": source, "frame_id": ident or str(stamp).replace(".", "-"),
        "timestamp": datetime.fromtimestamp(stamp, timezone.utc).isoformat()}


def prediction(angle, confidence=.95, abstain=False):
    return {"angle_cw": angle, "confidence": confidence, "abstain": abstain}


def tracking():
    cfg = logic.configuration({"smoothing_window": 2, "max_age_ms": 600000, "reset_after_sec": 20})
    track = logic.Tracker(cfg)
    first = track.apply(event(100), prediction(359), 100)
    assert first["orientation"]["state"] == "warming"
    second = track.apply(event(101), prediction(1), 101)
    assert abs(logic.signed(second["orientation"]["smoothed_angle_cw"])) < .001
    assert second["orientation"]["state"] == "stable" and "change" not in second
    moved = track.apply(event(102), prediction(30), 102)
    assert moved["change"]["event"] == "rotated" and moved["change"]["delta_cw"] > 10
    assert "change" not in track.apply(event(103), prediction(30), 103)
    assert track.apply(event(104), prediction(30, .1), 104)["orientation"]["state"] == "uncertain"
    assert track.apply(event(105), prediction(0), 105)["orientation"]["state"] == "warming"
    assert track.apply(event(106), prediction(0), 106)["change"]["event"] == "returned"
    assert track.apply(event(106), prediction(0), 107)["orientation"]["state"] == "out_of_order"
    assert track.apply(event(107), prediction(0), 999)["orientation"]["state"] == "stale"
    before = json.dumps(track.sources)
    refused(lambda: track.apply(event(108), prediction(float("nan")), 108))
    assert json.dumps(track.sources) == before, "Malformed model output must not mutate tracking"
    assert track.apply(event(108), prediction(20, abstain=True), 108)["orientation"]["accepted"] is False
    assert track.apply(event(109, "other"), prediction(200), 109)["orientation"]["state"] == "warming"
    assert track.apply(event(130), prediction(20), 130)["orientation"]["state"] == "warming"
    assert track.apply(event(131), prediction(20), 131)["orientation"]["baseline_angle_cw"] == 20
    unstable = logic.Tracker(cfg)
    unstable.apply(event(1), prediction(0), 1)
    assert unstable.apply(event(2), prediction(180), 2)["orientation"]["state"] == "unstable"
    limit = logic.Tracker({**cfg, "max_sources": 1})
    limit.apply(event(1), prediction(0), 1)
    assert limit.apply(event(2, "other"), prediction(0), 2)["orientation"]["state"] == "source_limit"


def scheduling():
    class Controlled:
        def __init__(self, *args):
            self.started, self.release = threading.Event(), threading.Event()
            self.calls, self.closed = [], False
        def predict(self, item, cancel):
            self.calls.append(item["frame_id"]); self.started.set()
            while not self.release.wait(.01):
                if cancel():
                    raise logic.Cancelled("Cancelled")
            if item["frame_id"] == "bad":
                raise logic.OrientationError("Bad frame")
            return prediction(10)
        def close(self):
            self.closed = True
    cfg = logic.configuration({"min_interval_ms": 0, "smoothing_window": 1, "max_sources": 2})
    outputs = []
    listener = engine.Listener(cfg, ".", ".", outputs.append, factory=Controlled)
    now = time.time()
    def offer(index, source="camera", ident=None):
        return listener.offer(event(now+index*.001, source, ident or str(index)))
    try:
        assert offer(0); listener.tick()
        assert listener.session.started.wait(1)
        for number in range(1, 101):
            assert offer(number)
        assert len(listener.pending) == 1
        assert offer(101, "other") and not offer(102, "third")
        assert not offer(99)
        listener.session.release.set()
        deadline = time.monotonic()+2
        while len(outputs) < 3 and time.monotonic() < deadline:
            listener.tick(); time.sleep(.01)
        assert listener.session.calls == ["0", "100", "101"], listener.session.calls
        assert len(outputs) == 3 and listener.dropped == 101
        offer(103, ident="bad")
        deadline = time.monotonic()+1
        while len(outputs) < 4 and time.monotonic() < deadline:
            listener.tick(); time.sleep(.01)
        assert outputs[-1]["orientation"]["state"] == "error"
        listener.session.started.clear(); listener.session.release.clear()
        offer(104); listener.tick()
        assert listener.session.started.wait(1)
        started = time.monotonic(); listener.close()
        assert time.monotonic()-started < .5 and len(outputs) == 4
        assert listener.session.closed
    finally:
        listener.close()
    # Rate limit retains newest work without starting it before the interval.
    listener = engine.Listener({**cfg, "min_interval_ms": 1000}, ".", ".", lambda _: None, factory=Controlled)
    listener.session.release.set()
    try:
        listener.offer(event(now, ident="first")); listener.tick(100)
        assert listener.session.started.wait(1)
        while not listener.future.done():
            time.sleep(.005)
        listener.offer(event(now+.001, ident="latest")); listener.tick(100.5)
        assert listener.future is None and len(listener.pending) == 1
        listener.tick(101)
        assert listener.future is not None
    finally:
        listener.close()


def main():
    import numpy as np
    from PIL import Image
    for config in ({"tier": "remote"}, {"device": "auto"}, {"max_sources": True}, {"min_confidence": float("nan")},
        {"source_id": "bad/source"}, {"download_missing": "yes"}, {"smoothing_window": 0}, {"url": "no"}):
        refused(lambda: logic.configuration(config))
    cfg = logic.configuration({})
    for raw in ("https://invalid/image.png", "../forbidden\n", {}, {"path": "x", "timestamp": "2025-01-01"},
        '{"path":"x","path":"y"}', {"path": "x", "sha256": "invalid"}, {"path": "x", "source_id": []}):
        refused(lambda: logic.frame(raw, cfg))
    probabilities = np.zeros((1, 360), np.float32); probabilities[0, 359] = .7; probabilities[0, 1] = .3
    angle, confidence = decode(probabilities)
    assert abs(logic.signed(angle)-(-.4)) < .0001 and confidence == 1
    refused(lambda: decode(np.zeros((1, 360))))
    pixels = letterbox(Image.new("RGB", (40, 20), (255, 0, 0)), 32)
    assert pixels.shape == (1, 3, 32, 32) and pixels.dtype == np.float32
    assert abs(pixels[0, 0, 16, 16] - (1-.485)/.229) < .00001
    for url in ("http://huggingface.co/a", "https://evil.invalid/a", "https://huggingface.co.evil.invalid/a"):
        refused(lambda: Redirects().redirect_request(Request("https://huggingface.co/a"), None, 302, "", {}, url))
    tracking(); scheduling()
    with TemporaryDirectory(prefix="rightwayup-real-") as temporary:
        root = Path(temporary); storage = root/"storage"; storage.mkdir()
        install_cache(storage, ("nano", "fast", "max"))
        source = picture(root); original = source.read_bytes()
        cfg = logic.configuration({"allowed_root": str(root), "download_missing": False})
        for tier in ("nano", "fast", "max"):
            job = engine.Session({**cfg, "tier": tier}, root, storage)
            try:
                result = job.predict(logic.frame(str(source), cfg))
                assert 0 <= result["angle_cw"] < 360 and 0 <= result["confidence"] <= 1
                assert result["model"]["tier"] == tier and result["model"]["device"] == "cpu"
                assert result["model"]["revision"] == manifest()["revision"]
                assert result["source"]["sha256"] == hashlib.sha256(original).hexdigest()
                pid = job.process.pid
                repeated = job.predict(logic.frame(str(source), cfg))
                assert job.process.pid == pid and abs(repeated["angle_cw"]-result["angle_cw"]) < .00001
                print("[ok] Real ONNX " + tier + " CPU, confidence=" + str(round(result["confidence"], 4)), flush=True)
            finally:
                job.close()
            assert not list(storage.glob("rightwayup-job-*"))
        assert source.read_bytes() == original
        # Allowing first-use downloads must not retain staging after a cached first load.
        cached_online = engine.Session({**cfg, "download_missing": True}, root, storage)
        try:
            cached_online.predict(logic.frame(str(source), cfg))
            assert not list(storage.glob("rightwayup-job-*"))
            cached_online.predict(logic.frame(str(source), cfg))
            assert not list(storage.glob("rightwayup-job-*"))
        finally:
            cached_online.close()
        # A host crash must not orphan its confirmed live native model process.
        child_script = """import json,os,sys
from blocs.rightwayup import engine,logic
cfg,root,storage,path=json.loads(sys.argv[1])
job=engine.Session(cfg,root,storage)
job.predict(logic.frame(path,cfg))
print(job.process.pid,flush=True)
os._exit(0)
"""
        parent = subprocess.run([sys.executable, "-B", "-c", child_script,
            json.dumps([cfg, str(root), str(storage), str(source)])], capture_output=True, text=True, timeout=20)
        assert parent.returncode == 0, parent.stderr
        native_pid = int(parent.stdout.strip())
        deadline = time.monotonic()+2
        def running():
            try:
                status = Path("/proc", str(native_pid), "stat").read_text().split(") ", 1)[1].split()[0]
                return status != "Z"
            except FileNotFoundError:
                return False
        while running() and time.monotonic() < deadline:
            time.sleep(.01)
        assert not running(), "Native model survived the death of its owning process"
        # The crashed test host cannot run Python finally: clean only its known empty job dirs.
        for directory in storage.glob("rightwayup-job-*"):
            directory.rmdir()
        flat = root/"blank.png"; Image.new("RGB", (640, 480), (100, 150, 220)).save(flat)
        job = engine.Session(cfg, root, storage)
        try:
            assert job.predict(logic.frame(str(flat), cfg))["abstain"] is True
            refused(lambda: job.predict(logic.frame({"path": str(source), "sha256": "0"*64}, cfg)))
            assert job.process is None
            refused(lambda: job.predict(logic.frame(str(source), cfg), lambda: True))
            assert job.process is None
            # Stop a confirmed running native worker, not merely a preflight.
            job.predict(logic.frame(str(source), cfg)); pid = job.process.pid
            os.kill(pid, 19)
            started = time.monotonic()
            refused(lambda: job.predict(logic.frame(str(source), cfg), lambda: time.monotonic()-started > .1))
            assert time.monotonic()-started < 1 and job.process is None
            refused(lambda: os.kill(pid, 0))
        finally:
            job.close()
        linked = root/"linked.png"; linked.symlink_to(source)
        refused(lambda: engine.Session(cfg, root, storage).predict(logic.frame(str(linked), cfg)))
        refused(lambda: engine.Session(cfg, root, storage).predict(logic.frame("../escape.png", cfg)))
        oversized = root/"oversized.png"; Image.new("RGB", (400, 400)).save(oversized)
        job = engine.Session({**cfg, "max_pixels": 65536}, root, storage)
        refused(lambda: job.predict(logic.frame(str(oversized), cfg)))
        animated = root/"animated.png"
        Image.new("RGB", (10, 10)).save(animated, save_all=True, append_images=[Image.new("RGB", (10, 10), "red")])
        job = engine.Session(cfg, root, storage)
        refused(lambda: job.predict(logic.frame(str(animated), cfg)))
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            job = engine.Session({**cfg, "device": "cuda"}, root, storage)
            try:
                job.predict(logic.frame(str(source), cfg))
                raise AssertionError("CUDA silently fell back to CPU")
            except logic.OrientationError as exc:
                assert exc.code == "cuda_unavailable"
        empty = root/"empty"; empty.mkdir()
        refused(lambda: engine.Session(cfg, root, empty).predict(logic.frame(str(source), cfg)))
        spec = manifest()["assets"]["fast-s224-int8.onnx"]
        cached = storage/"rightwayup-cache"/spec["filename"]
        with cached.open("r+b") as stream:
            stream.seek(128); stream.write(b"INVALID")
        damaged = hashlib.sha256(cached.read_bytes()).hexdigest()
        refused(lambda: engine.Session(cfg, root, storage).predict(logic.frame(str(source), cfg)))
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == damaged
        assert not list(storage.glob("rightwayup-job-*"))
    print("[ok] Circular confidence-gated drift, source isolation, newest-pending queue, rate limits, Stop, source preservation")


if __name__ == "__main__":
    main()
