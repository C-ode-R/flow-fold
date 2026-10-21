import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from collect_afdb_short import next_page_url, one_candidate  # noqa: E402
from finalize_flow_manifest import main as finalize_main  # noqa: E402
from flow50k_filelist import ROOT, main as filelist_main  # noqa: E402


class FlowCollectionTests(unittest.TestCase):
    def test_uniprot_cursor_with_comma_in_fields(self):
        link = ('<https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence'
                '&cursor=abc>; rel="next"')
        self.assertEqual(next_page_url(link),
                         "https://rest.uniprot.org/uniprotkb/search?fields=accession,sequence&cursor=abc")
        self.assertIsNone(next_page_url(""))

    def test_metadata_filter_avoids_structure_download(self):
        sequence = "A" * 50
        metadata = json.dumps([dict(sequence=sequence, globalMetricValue=90,
                                    fractionPlddtLow=0.02, fractionPlddtVeryLow=0,
                                    pdbUrl="https://alphafold.ebi.ac.uk/files/example.pdb")]).encode()
        with tempfile.TemporaryDirectory() as directory, patch(
                "collect_afdb_short.fetch", return_value=(metadata, {})) as fetch:
            row, reason = one_candidate(("P12345", sequence), Path(directory),
                                        SimpleNamespace(mean_plddt=85, min_plddt=70,
                                                        max_plddt_std=15))
            self.assertIsNone(row)
            self.assertEqual(reason, "metadata low minimum pLDDT")
            fetch.assert_called_once()

    def test_transfer_list_uses_portable_manifest_and_checks_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            pdb = base / "pdb" / "test.pdb"
            pdb.parent.mkdir()
            pdb.write_bytes(b"example structure")
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps(dict(path_base="manifest", proteins=[
                dict(path="pdb/test.pdb", checksum=hashlib.sha256(pdb.read_bytes()).hexdigest())
            ])))
            output = base / "files.txt"
            with patch.object(sys, "argv", ["filelist", "--manifest", str(manifest),
                                             "--out", str(output)]):
                filelist_main()
            listed = output.read_text().splitlines()
            self.assertIn(str(pdb.relative_to(ROOT)), listed)
            self.assertIn(str(manifest.relative_to(ROOT)), listed)

    def test_direct_holdout_search_excludes_candidate(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            candidates = base / "candidates.json"
            holdouts = base / "holdouts.json"
            output = base / "final" / "manifest.json"
            candidates.write_text(json.dumps(dict(proteins=[
                dict(id="A", chain="A", sequence="A" * 40, path="pdb/A.pdb"),
                dict(id="B", chain="A", sequence="C" * 40, path="pdb/B.pdb"),
            ])))
            holdouts.write_text(json.dumps(dict(proteins=[
                dict(id="V", chain="A", sequence="D" * 40, split="val", path="../data/cif/V.cif"),
                dict(id="T", chain="A", sequence="E" * 40, split="test", path="../data/cif/T.cif"),
            ])))

            def fake_mmseqs(command, check):
                if command[1] == "easy-cluster":
                    Path(str(command[3]) + "_cluster.tsv").write_text(
                        "C0\tC0\nC1\tC1\nH0\tH0\nH1\tH1\n")
                else:
                    Path(command[4]).write_text("C1\tH0\n")

            with patch.object(sys, "argv", ["finalize", "--candidates", str(candidates),
                                            "--holdout-manifest", str(holdouts),
                                            "--out", str(output), "--target", "1"]), patch(
                    "finalize_flow_manifest.shutil.which", return_value="/usr/bin/mmseqs"), patch(
                    "finalize_flow_manifest.subprocess.run", side_effect=fake_mmseqs):
                finalize_main()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["proteins"][0]["id"], "A")
            self.assertEqual(payload["direct_holdout_hits"], 1)


if __name__ == "__main__":
    unittest.main()
