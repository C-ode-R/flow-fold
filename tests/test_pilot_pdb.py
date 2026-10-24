"""Offline checks: rates must use whole-chain conjunction and failures stay unknown."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pilot_pdb as pilot


def chain(entry, entity, flags):
    return dict(entry_id=entry, entity_id=entity, flags=flags,
                method=['X-RAY DIFFRACTION'], assembly_class='monomer_annotated')


class PilotTests(unittest.TestCase):
    def test_conjunction_duplicates_and_unknown_failures(self):
        rows = [dict(status='audited', chains=[chain('A','1',['quality']), chain('A','1',['assembly'])]),
                dict(status='audited', chains=[chain('B','1',[]), chain('B','1',[])]),
                dict(status='failed', error='timeout')]
        result = pilot.summarize(rows, 4)
        self.assertEqual(result['entity_at_least_one_chain_passes'], pilot.rate(1, 2))
        self.assertEqual(result['entry_at_least_one_candidate_passes_among_successes'], pilot.rate(1, 2))
        self.assertEqual(result['entry_confirmed_passes_over_selected_lower_bound'], pilot.rate(1, 4))
        self.assertEqual(result['pending_entries'], 1)
        self.assertEqual(result['failed_entries'], 1)

    def test_local_cache_end_to_end_and_repeat_without_network(self):
        cache = pilot.ROOT / 'data/raw/pdb/10CY.cif.gz'
        if not cache.exists() or not (pilot.ROOT / 'data/catalog/20261023T032824Z/pilot.json').exists():
            self.skipTest('local 20-entry raw pilot not available')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            pilot.save(out / 'sample.json', dict(seed=7, query=pilot.build_query(),
                snapshot_utc='fixture', entries=[dict(entry_id='10CY', candidate_entities=['10CY_2'])]))
            # Read actual entity IDs from the already-frozen local search snapshot.
            source = json.loads((pilot.ROOT / 'data/catalog/20261023T032824Z/pilot.json').read_text())
            record = next(r for r in source['records'] if r['entry_id']=='10CY')
            payload=json.loads((out/'sample.json').read_text())
            payload['entries'][0]['candidate_entities']=record['candidate_entities']
            pilot.save(out/'sample.json',payload)
            for _ in range(2):
                with patch.object(sys, 'argv', ['pilot_pdb.py','--sample-size','1','--seed','7','--out',folder]), \
                     patch.object(pilot, 'fetch', side_effect=AssertionError('network prohibited')), \
                     patch.object(pilot.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                    pilot.main()
                result=json.loads((out/'summary.json').read_text())
                self.assertEqual(result['successful_entries'],1)
                self.assertEqual(result['failed_entries'],0)
                self.assertEqual(result['entry_at_least_one_candidate_passes_among_successes']['passed'],0)


if __name__ == '__main__':
    unittest.main()
