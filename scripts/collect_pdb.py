"""Freeze experimental protein entity candidates and download a small raw pilot.
No training split or biological quality claim is made here. Standard library only.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers={
                'Content-Type': 'application/json', 'User-Agent': 'FoldFlow-data-pilot/0.1'})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def build_query(min_length=40, max_length=128):
    return {
        'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'entity_poly.rcsb_entity_polymer_type',
                'operator': 'exact_match', 'value': 'Protein'}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'entity_poly.rcsb_sample_sequence_length',
                'operator': 'range', 'value': {'from': min_length,
                    'to': max_length, 'include_lower': True, 'include_upper': True}}}]},
        'return_type': 'polymer_entity',
        'request_options': {'results_content_type': ['experimental'], 'return_all_hits': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-length', type=int, default=40)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--pilot-entries', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.min_length <= args.max_length or args.pilot_entries < 0:
        parser.error('invalid length range or pilot size')
    query = build_query(args.min_length, args.max_length)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    catalog = ROOT / 'data/catalog' / stamp
    catalog.mkdir(parents=True)
    (catalog / 'query.json').write_text(json.dumps(query, indent=2) + '\n')
    raw = fetch('https://search.rcsb.org/rcsbsearch/v2/query', query)
    (catalog / 'search_response.json').write_bytes(raw)
    result = json.loads(raw)
    ids = sorted({row['identifier'] for row in result['result_set']})
    (catalog / 'candidate_entities.txt').write_text('\n'.join(ids) + '\n')
    entries = sorted({identifier.split('_')[0] for identifier in ids})
    records = []
    for entry in entries[:args.pilot_entries]:
        path = ROOT / 'data/raw/pdb' / f'{entry}.cif.gz'
        url = f'https://files.rcsb.org/download/{entry}.cif.gz'
        try:
            blob = path.read_bytes() if path.exists() else fetch(url)
            unpacked = gzip.decompress(blob)
            if not unpacked.lstrip().startswith(b'data_'):
                raise ValueError('not an mmCIF data block')
            path.write_bytes(blob)
            records.append({'entry_id': entry, 'candidate_entities': [
                x for x in ids if x.startswith(entry + '_')],
                'url': url, 'path': path.relative_to(ROOT).as_posix(),
                'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob),
                'status': 'downloaded'})
        except Exception as error:
            records.append({'entry_id': entry, 'url': url, 'status': 'failed',
                            'error': str(error)})
        time.sleep(0.1)
    summary = {'snapshot_utc': stamp, 'length_range': [args.min_length, args.max_length],
        'candidate_entities': len(ids), 'unique_entries': len(entries),
        'pilot_selection': 'lexicographic entry IDs; connectivity pilot, not representative',
        'downloaded': sum(r['status'] == 'downloaded' for r in records),
        'failed': sum(r['status'] == 'failed' for r in records),
        'records': records}
    (catalog / 'pilot.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))
    print(f'Catalog: {catalog}')
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
