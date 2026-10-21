"""Collect high-confidence 40–128 residue AlphaFold DB candidates.

Produces TRAIN CANDIDATES, not a leakage-checked training split. Run sequence
clustering against experimental holdouts before embedding or model training.
The script is resumable: accepted PDB files are retained and the manifest is
written atomically after every page. It never fabricates a requested count.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/"
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "flow-fold-research/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read(), response.headers
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in (400, 404):
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(min(30, 2 ** attempt))


def validate_structure(path, expected_sequence, mean_floor, minimum_floor, std_ceiling):
    structure = PDBParser(QUIET=True).get_structure("afdb", str(path))
    chains = list(structure[0].get_chains())
    if len(chains) != 1:
        raise ValueError("expected one chain")
    residues = [residue for residue in chains[0] if residue.id[0] == " "]
    if not residues or any(not all(atom in residue for atom in ("N", "CA", "C")) for residue in residues):
        raise ValueError("missing backbone atom")
    sequence = "".join(seq1(residue.resname, custom_map={"UNK": "X"}) for residue in residues)
    if sequence != expected_sequence:
        raise ValueError("structure/UniProt sequence mismatch")
    ids = [residue.id[1] for residue in residues]
    if any(right != left + 1 for left, right in zip(ids, ids[1:])):
        raise ValueError("residue numbering gap")
    scores = [float(residue["CA"].bfactor) for residue in residues]
    if min(scores) < minimum_floor or statistics.mean(scores) < mean_floor:
        raise ValueError("low pLDDT")
    if statistics.pstdev(scores) > std_ceiling:
        raise ValueError("high pLDDT variation")
    for left, right in zip(residues, residues[1:]):
        delta = left["C"].coord - right["N"].coord
        if math.sqrt(float((delta * delta).sum())) > 2.2:
            raise ValueError("peptide chain break")
    return dict(mean_plddt=statistics.mean(scores), std_plddt=statistics.pstdev(scores))


def one_candidate(candidate, out_dir, args):
    accession, sequence = candidate
    if not 40 <= len(sequence) <= 128 or set(sequence) - ALPHABET:
        return None, "length/nonstandard sequence"
    destination = out_dir / "pdb" / f"{accession}.pdb"
    model_version = None
    try:
        if not destination.is_file():
            metadata, _ = fetch(AFDB + urllib.parse.quote(accession))
            entries = json.loads(metadata)
            if not entries:
                return None, "no AFDB entry"
            url = entries[0].get("pdbUrl")
            model_version = entries[0].get("latestVersion") or entries[0].get("modelVersion")
            if not url or urllib.parse.urlparse(url).hostname not in {
                "alphafold.ebi.ac.uk", "www.alphafold.ebi.ac.uk", "alphafold.com"
            }:
                return None, "untrusted or missing PDB URL"
            contents, _ = fetch(url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pdb.tmp")
            temporary.write_bytes(contents)
            temporary.replace(destination)
        quality = validate_structure(destination, sequence, args.mean_plddt,
                                     args.min_plddt, args.max_plddt_std)
        row = dict(id=accession, chain="A", sequence=sequence, residues=len(sequence),
                   source="AlphaFoldDB", split="candidate", path=str(destination.relative_to(out_dir)),
                   quality=quality, model_version=model_version,
                   checksum=hashlib.sha256(destination.read_bytes()).hexdigest())
        return row, None
    except (ValueError, KeyError, IndexError, urllib.error.URLError, OSError) as error:
        destination.unlink(missing_ok=True)
        return None, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--max-candidates", type=int, default=500000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mean-plddt", type=float, default=85)
    parser.add_argument("--min-plddt", type=float, default=70)
    parser.add_argument("--max-plddt-std", type=float, default=15)
    args = parser.parse_args()
    if args.target < 1 or args.workers < 1 or args.max_candidates < args.target:
        parser.error("target/workers must be positive and max-candidates >= target")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "candidates.json"
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "source": "UniProtKB sequences + AlphaFold DB PDB coordinates",
        "path_base": "manifest",
        "query": "length:[40 TO 128]", "proteins": [], "processed_count": 0,
        "next_url": None, "exhausted": False, "rejections": {}
    }
    accepted = {row["id"] for row in state["proteins"]}
    params = urllib.parse.urlencode({"query": "length:[40 TO 128]", "format": "tsv",
                                     "fields": "accession,sequence", "size": 500})
    url = None if state.get("exhausted", False) else state["next_url"] or (UNIPROT + "?" + params)
    seen = state["processed_count"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while url and len(accepted) < args.target and seen < args.max_candidates:
            body, headers = fetch(url)
            lines = body.decode().splitlines()
            candidates = []
            for line in lines[1:]:
                fields = line.split("\t")
                if len(fields) >= 2 and fields[0] not in accepted:
                    candidates.append((fields[0], fields[1]))
            for candidate, (row, reason) in zip(candidates, pool.map(
                    lambda item: one_candidate(item, args.out, args), candidates)):
                accession = candidate[0]
                seen += 1
                if row and len(accepted) < args.target:
                    accepted.add(accession)
                    state["proteins"].append(row)
                elif reason:
                    state["rejections"][reason] = state["rejections"].get(reason, 0) + 1
                if seen >= args.max_candidates:
                    break
            state["processed_count"] = seen
            links = headers.get("Link", "")
            url = next((item[item.index("<") + 1:item.index(">")] for item in links.split(",")
                        if 'rel="next"' in item and "<" in item and ">" in item), None)
            state["next_url"] = url
            state["exhausted"] = url is None
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(manifest_path)
            print(json.dumps({"processed": seen, "accepted": len(accepted),
                              "target": args.target}), flush=True)
    print(json.dumps({"accepted": len(accepted), "target": args.target,
                      "manifest": str(manifest_path)}))
    if len(accepted) < args.target:
        raise SystemExit("Target not reached; inspect rejections and expand candidate source")


if __name__ == "__main__":
    main()
