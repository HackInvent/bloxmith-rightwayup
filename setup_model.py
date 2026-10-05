"""Optional explicit offline preparation; no blueprint or framework files are changed."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from assets import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--tier", choices=("nano", "fast", "max"), default="fast")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-sec", type=int, default=180)
    args = parser.parse_args()
    if not 10 <= args.timeout_sec <= 600 or args.storage.is_symlink():
        parser.error("Choose a real storage directory and a 10–600 second deadline")
    args.storage.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rightwayup-setup-", dir=args.storage) as staging:
        _, spec = load(args.storage, args.tier, args.device, allow_download=True,
                       timeout_sec=args.timeout_sec, staging_dir=staging)
    print("[ok] Verified " + spec["filename"])


if __name__ == "__main__":
    main()
