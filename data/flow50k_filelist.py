"""Write a rsync file list containing only the finalized 50k structures."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text())
    if payload.get("path_base") != "manifest":
        parser.error("expected a portable manifest with path_base=manifest")
    paths = {manifest.relative_to(ROOT)}
    audit = manifest.parent / "audit.json"
    if audit.is_file():
        paths.add(audit.relative_to(ROOT))
    for row in payload["proteins"]:
        path = (manifest.parent / row["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"structure outside repository: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if row.get("checksum"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["checksum"]:
                raise ValueError(f"checksum mismatch: {path}")
        paths.add(path.relative_to(ROOT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{path.as_posix()}\n" for path in sorted(paths)))
    print(f"{len(paths)} files listed in {args.out}")


if __name__ == "__main__":
    main()
