"""Descriptor-relative stable snapshots of explicitly permitted image files."""

import hashlib
import os
from pathlib import Path
import stat
import time
from .logic import OrientationError, Cancelled

def check(deadline, cancel):
    if cancel():
        raise Cancelled("Orientation cancelled.")
    if time.monotonic() >= deadline:
        raise OrientationError("Orientation deadline exceeded.")


def snapshot(value,cfg,root_dir,deadline,cancel):
    """Read a stable, single-link regular file through held directory descriptors."""
    if not isinstance(value,str) or not value.strip() or len(value)>4096 or "\x00" in value:
        raise OrientationError("A local image path is required.")
    base = Path(root_dir).resolve(strict=True)
    root = Path(cfg["allowed_root"]).expanduser() if cfg["allowed_root"] else base
    root = (root if root.is_absolute() else base/root).resolve(strict=True)
    candidate = Path(value).expanduser()
    if ".." in candidate.parts:
        raise OrientationError("Parent traversal is not allowed.")
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            raise OrientationError("Image must remain below allowed_root.") from None
    else:
        relative = candidate
    parts = relative.parts
    if not parts or any(part in ("",".","..") for part in parts):
        raise OrientationError("A regular image file is required.")
    current = os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            check(deadline,cancel)
            following = os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=current)
            os.close(current)
            current = following
        descriptor = os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=current)
        with os.fdopen(descriptor,"rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=cfg["max_bytes"]:
                raise OrientationError("Source must be a nonempty single-link regular file within max_bytes.")
            chunks,total = [],0
            while chunk:=source.read(65536):
                check(deadline,cancel)
                total+=len(chunk)
                if total>cfg["max_bytes"]:
                    raise OrientationError("Source exceeds max_bytes.")
                chunks.append(chunk)
            after = os.fstat(source.fileno())
            current_info = os.stat(parts[-1],dir_fd=current,follow_symlinks=False)
            identity = lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink)
            if identity(before)!=identity(after) or identity(before)!=identity(current_info):
                raise OrientationError("Source changed while being read.")
        binary=b"".join(chunks)
        return binary,{"path":str(root/relative),"sha256":hashlib.sha256(binary).hexdigest(),"bytes":len(binary)}
    finally:
        os.close(current)
