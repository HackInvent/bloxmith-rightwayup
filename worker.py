"""Persistent, parent-fenced ONNX process. No framework imports or image upload.

Preprocessing, angle refinement and confidence follow the ORTUS reference pipeline.
See NOTICE for attribution; resource, validation and process controls are adapter-owned.
"""

import base64
import ctypes
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import types

if __package__:
    from .assets import AssetError, load, manifest
else:
    directory = Path(__file__).resolve().parent
    package = types.ModuleType("rightwayup_owned")
    package.__path__ = [str(directory)]
    sys.modules["rightwayup_owned"] = package
    from rightwayup_owned.assets import AssetError, load, manifest


def letterbox(image, size):
    import numpy as np
    from PIL import Image
    scale = min(size/image.width, size/image.height)
    width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(image.resize((width, height), Image.Resampling.BICUBIC), ((size-width)//2, (size-height)//2))
    pixels = (np.asarray(canvas, np.float32)/255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return pixels.transpose(2, 0, 1).copy()[None]


def decode(prob):
    import numpy as np
    prob = np.asarray(prob)
    if prob.shape != (1, 360) or not np.isfinite(prob).all() or (prob < 0).any() or abs(float(prob.sum())-1) > .0001:
        raise AssetError("invalid_prediction")
    peak = int(prob[0].argmax())
    offsets = np.arange(-10, 11)
    weights = prob[0, (peak+offsets) % 360]
    angle = float((peak + (weights*offsets).sum()/max(float(weights.sum()), 1e-12)) % 360)
    distances = np.abs((np.arange(360)-angle+180) % 360-180)
    confidence = min(1.0, float((prob[0]*(distances <= 10)).sum()))
    return angle, confidence


class Predictor:
    def __init__(self):
        self.session, self.config = None, None

    def predict(self, item):
        from PIL import Image, ImageOps
        import onnxruntime as ort
        cfg = item["config"]
        if self.config is not None and self.config != cfg:
            raise AssetError("changed_config")
        if cfg["device"] == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise AssetError("cuda_unavailable")
        binary = base64.b64decode(item["data"], validate=True)
        with Image.open(io.BytesIO(binary), formats=["JPEG", "PNG", "WEBP", "TIFF", "BMP"]) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise AssetError("multiple_frames")
            if min(source.size) < 1 or max(source.size) > 16384 or source.width*source.height > cfg["max_pixels"]:
                raise AssetError("dimensions")
            image = ImageOps.exif_transpose(source).convert("RGB")
        info = manifest()
        tier = info["tiers"][cfg["tier"]]
        precision = "int8" if cfg["device"] == "cpu" else "fp16"
        if self.session is None:
            if cfg["device"] == "cpu":
                resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
            model, spec = load(item["storage"], cfg["tier"], cfg["device"], allow_download=cfg["download_missing"],
                timeout_sec=cfg["download_timeout_sec"], staging_dir=item["staging_dir"])
            options = ort.SessionOptions()
            options.intra_op_num_threads = cfg["threads"]
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            providers = ["CPUExecutionProvider"] if cfg["device"] == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model, options, providers=providers)
            self.session.disable_fallback()
            if self.session.get_providers()[0] != providers[0]:
                raise AssetError("device_unavailable")
            self.config = cfg
            self.model = {"name": "RightWayUp", "version": info["model_version"], "revision": info["revision"],
                "tier": cfg["tier"], "precision": precision, "device": cfg["device"], "sha256": spec["sha256"]}
        probabilities = self.session.run(["prob"], {"image": letterbox(image, tier["size"])})[0]
        angle, confidence = decode(probabilities)
        thresholds = tier["thresholds"][precision]
        threshold = thresholds["standard"] if cfg["abstention"] == "standard" else max(thresholds.values())
        return {"angle_cw": angle, "confidence": confidence, "abstain": confidence < threshold,
            "abstention_threshold": threshold, "model": self.model, "oriented_width": image.width, "oriented_height": image.height}


def main():
    if sys.platform != "linux" or len(sys.argv) != 2:
        raise SystemExit(2)
    parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if os.getppid() != parent or libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        raise SystemExit(2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    predictor = Predictor()
    while True:
        line = sys.stdin.buffer.readline(30000001)
        if not line:
            break
        if len(line) > 30000000 or not line.endswith(b"\n"):
            raise SystemExit(2)
        try:
            response = predictor.predict(json.loads(line))
        except ImportError:
            response = {"error": "dependency"}
        except AssetError as exc:
            response = {"error": str(exc)}
        except Exception:
            response = {"error": "inference_failed"}
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
