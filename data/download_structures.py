"""Download the public RCSB mmCIF files referenced by the portable manifest."""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def download(row):
    pdb_id = row["id"]
    destination = (ROOT / "outputs" / row["path"]).resolve()
    expected_dir = (ROOT / "data" / "cif").resolve()
    if destination.parent != expected_dir:
        raise ValueError(f"Unexpected structure path for {pdb_id}: {destination}")
    if destination.is_file() and destination.stat().st_size:
        return pdb_id, "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    with urllib.request.urlopen(url, timeout=30) as response:
        contents = response.read()
    temporary = destination.with_suffix(".cif.tmp")
    temporary.write_bytes(contents)
    temporary.replace(destination)
    return pdb_id, "downloaded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_1000train_200holdout.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["proteins"]
    unique = list({row["id"]: row for row in rows}.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (pdb_id, status) in enumerate(pool.map(download, unique), 1):
            if index % 100 == 0 or index == len(unique):
                print(f"{index}/{len(unique)} structures ({pdb_id}: {status})", flush=True)


if __name__ == "__main__":
    main()
