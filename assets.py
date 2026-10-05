"""Checksum-pinned, node-owned ONNX cache; no unreviewed model URL or implicit replacement."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class AssetError(ValueError):
    pass


def manifest():
    return json.loads(Path(__file__).with_name("model_artifacts.json").read_text(encoding="utf-8"))


class Redirects(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or not (
            host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")):
            raise AssetError("download_failed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verified_bytes(path, spec):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError:
        raise AssetError("cache_corrupt") from None
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != spec["bytes"]:
            raise AssetError("cache_corrupt")
        data = source.read(spec["bytes"]+1)
        after = os.fstat(source.fileno())
        if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]):
            raise AssetError("cache_corrupt")
        return data


def load(storage, tier, device, *, allow_download, timeout_sec, staging_dir=None):
    info = manifest()
    spec = info["assets"][info["tiers"][tier]["stem"] + ("-int8.onnx" if device == "cpu" else "-fp16.onnx")]
    storage = Path(storage)
    if storage.is_symlink() or not storage.is_dir():
        raise AssetError("cache_unavailable")
    cache = storage / "rightwayup-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or not cache.is_dir():
        raise AssetError("cache_unavailable")
    target = cache / spec["filename"]
    lock = os.open(cache / "assets.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    temporary = None
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise AssetError("cache_unavailable")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AssetError("cache_busy") from None
        data = verified_bytes(target, spec)
        if data is not None:
            return data, spec
        if not allow_download:
            raise AssetError("assets_missing")
        deadline = time.monotonic()+timeout_sec
        digest, total = hashlib.sha256(), 0
        opener = build_opener(ProxyHandler({}), Redirects())
        request = Request(spec["url"], headers={"User-Agent": "BloxSmith-RightWayUp/0.1.0", "Accept-Encoding": "identity"})
        try:
            with opener.open(request, timeout=min(10, timeout_sec)) as response, tempfile.NamedTemporaryFile(
                prefix=".download-", suffix=".part", dir=staging_dir or cache, delete=False) as output:
                temporary = Path(output.name)
                if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
                    raise AssetError("download_failed")
                while True:
                    if time.monotonic() > deadline:
                        raise AssetError("download_failed")
                    chunk = response.read(min(65536, spec["bytes"]+1-total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > spec["bytes"]:
                        raise AssetError("download_failed")
                    digest.update(chunk); output.write(chunk)
                output.flush(); os.fsync(output.fileno())
            if total != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                raise AssetError("download_failed")
            os.link(temporary, target, follow_symlinks=False)
            temporary.unlink(); temporary = None
            return verified_bytes(target, spec), spec
        except (OSError, ValueError):
            raise AssetError("download_failed") from None
    finally:
        os.close(lock)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
