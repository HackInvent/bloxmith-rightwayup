"""Single model process per active listener, bounded IPC and newest-pending-frame scheduling."""

import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from . import logic
from .files import check, snapshot

MESSAGES = {"assets_missing": "Pinned weights are missing; enable downloads or prepare the offline cache.",
    "cache_corrupt": "A model failed its checksum and was neither loaded nor overwritten.",
    "cache_busy": "Another operation is preparing this block's model cache.",
    "download_failed": "Pinned model download or checksum verification failed.",
    "dependency": "Install Pillow, NumPy and ONNX Runtime in the runtime interpreter.",
    "cuda_unavailable": "CUDA was selected but its ONNX Runtime provider is unavailable.",
    "device_unavailable": "The selected execution device could not be activated.",
    "multiple_frames": "Supply one image, not an animation or a multi-frame container.",
    "dimensions": "Image exceeds the decoded pixel budget.",
    "inference_failed": "Image decoding or RightWayUp inference failed."}


class Session:
    def __init__(self, config, root, storage):
        self.config, self.root, self.storage = config, Path(root), Path(storage)
        self.process, self.temporary = None, None
        self.loaded = False

    def start(self):
        if self.process is not None:
            return
        if self.config["download_missing"]:
            self.temporary = tempfile.TemporaryDirectory(prefix="rightwayup-job-", dir=self.storage)
        environment = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"}}
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", DO_NOT_TRACK="1")
        # -E ignores Python environment overrides but keeps installed user-site distributions,
        # just like the runtime interpreter. This process is supervised, not an OS sandbox.
        self.process = subprocess.Popen([sys.executable, "-E", "-B", str(Path(__file__).with_name("worker.py")), str(os.getpid())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, close_fds=True)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def predict(self, item, cancel=lambda: False):
        started = time.monotonic()
        deadline = started + self.config["timeout_sec"] + (0 if self.loaded else self.config["download_timeout_sec"])
        try:
            binary, source = snapshot(item["path"], self.config, self.root, deadline, cancel)
            if item.get("sha256") and item["sha256"] != source["sha256"]:
                raise logic.OrientationError("Image no longer matches the supplied frame hash.", "source_changed")
            check(deadline, cancel)
            self.start()
            payload = json.dumps({"config": self.config, "storage": str(self.storage),
                "staging_dir": self.temporary.name if self.temporary is not None else None,
                "data": base64.b64encode(binary).decode("ascii")}, ensure_ascii=True).encode()+b"\n"
            position, response = 0, bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in response:
                    check(deadline, cancel)
                    for key, mask in selector.select(.03):
                        if mask & selectors.EVENT_WRITE:
                            try:
                                position += os.write(key.fd, payload[position:position+65536])
                            except BlockingIOError:
                                pass
                            if position == len(payload):
                                selector.unregister(self.process.stdin)
                        if mask & selectors.EVENT_READ:
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                raise logic.OrientationError("The owned inference process ended unexpectedly.", "worker_ended")
                            response.extend(chunk)
                    if len(response) > 65536:
                        raise logic.OrientationError("Model output exceeded its limit.", "invalid_prediction")
            check(deadline, cancel)
            result = json.loads(response)
            if result.get("error"):
                code = result["error"]
                raise logic.OrientationError(MESSAGES.get(code, "RightWayUp could not process this frame."), code)
            self.loaded = True
            # Staging belongs to initial weight loading, not to the live model process.
            # Managed hosts may end without Python finally at Stop; retain no idle job dir.
            if self.temporary is not None:
                self.temporary.cleanup(); self.temporary = None
            result["source"], result["elapsed_ms"] = source, round((time.monotonic()-started)*1000, 3)
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process.stdin.close(); self.process.stdout.close()
            self.process = None
        if self.temporary is not None:
            self.temporary.cleanup(); self.temporary = None
        self.loaded = False


class Listener:
    def __init__(self, config, root, storage, emit, factory=Session):
        self.config, self.emit = config, emit
        self.session = factory(config, root, storage)
        self.tracker = logic.Tracker(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rightwayup")
        self.cancel = threading.Event()
        self.pending, self.latest, self.next_at = OrderedDict(), {}, {}
        self.future, self.current = None, None
        self.dropped = 0

    def offer(self, item, now=None):
        now = time.time() if now is None else now
        source, stamp = item["source_id"], logic.epoch(item)
        if (now-stamp > self.config["max_age_ms"]/1000 or stamp <= self.latest.get(source, float('-inf'))
            or (source not in self.latest and len(self.latest) >= self.config["max_sources"])):
            self.dropped += 1; return False
        self.latest[source] = stamp
        if source in self.pending:
            self.dropped += 1
        self.pending[source] = item
        return True

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self.future is not None and self.future.done():
            try:
                prediction = self.future.result()
                values = self.tracker.apply(self.current, prediction)
            except Exception as exc:
                values = {"orientation": {"source_id": self.current["source_id"], "frame_id": self.current["frame_id"],
                    "timestamp": self.current["timestamp"], "accepted": False, "angle_cw": None,
                    "state": "cancelled" if isinstance(exc, logic.Cancelled) else "error",
                    "detail": str(exc) if isinstance(exc, logic.OrientationError) else "Local orientation is unavailable."}}
            values["orientation"]["dropped_frames"] = self.dropped
            if not self.cancel.is_set():
                self.emit(values)
            self.future, self.current = None, None
        if self.future is not None or self.cancel.is_set():
            return
        for source, item in list(self.pending.items()):
            if time.time()-logic.epoch(item) > self.config["max_age_ms"]/1000:
                self.pending.pop(source); self.dropped += 1; continue
            if now < self.next_at.get(source, 0):
                continue
            self.pending.pop(source)
            self.current = item
            self.next_at[source] = now + self.config["min_interval_ms"]/1000
            self.future = self.executor.submit(self.session.predict, item, self.cancel.is_set)
            break

    def close(self):
        self.cancel.set(); self.pending.clear()
        try:
            if self.future is not None:
                self.future.result(timeout=3)
        except Exception:
            pass
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.session.close()
