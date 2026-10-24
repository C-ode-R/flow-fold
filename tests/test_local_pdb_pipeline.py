import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_pdb_pipeline import extract


class ExtractionTests(unittest.TestCase):
    def test_full_sequence_keeps_modified_and_unobserved_residues(self):
        # No atom coordinates: sequence extraction must still retain all deposited residues.
        source='''data_TEST
_entry.id TEST
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
2 polydeoxyribonucleotide
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 MSE
1 3 GLY
2 1 DA
'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.cif.gz'
            with gzip.open(path,'wt') as f:f.write(source)
            payload=extract((str(path),path.stat().st_size,path.stat().st_mtime_ns))[-1]
            self.assertEqual(payload['status'],'ok')
            self.assertEqual(len(payload['entities']),1)
            e=payload['entities'][0]
            self.assertEqual(e['sequence'],'AXG')
            self.assertEqual(e['length'],3)
            self.assertFalse(e['standard'])

    def test_corrupt_input_is_unknown_not_empty_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.cif.gz'
            path.write_bytes(b'corrupt')
            payload=extract((str(path),7,0))[-1]
            self.assertEqual(payload['status'],'error')


if __name__=='__main__':unittest.main()
