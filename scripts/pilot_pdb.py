"""Random entry pilot with frozen sampling, resumable downloads and audit rates."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

from collect_pdb import ROOT, build_query, fetch
from audit_pdb import audit


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def rate(passed, total):
    return {'passed': passed, 'total': total,
            'percent': round(100 * passed / total, 2) if total else None}


def summarize(records, sample_size):
    successful = [r for r in records if r['status'] == 'audited']
    chains = [c for r in successful for c in r['chains']]
    entities = defaultdict(list)
    for c in chains:
        entities[(c['entry_id'], c['entity_id'])].append(c)
    eligible = lambda c: not c['flags']
    entry_pass = sum(any(eligible(c) for c in r['chains']) for r in successful)
    entity_pass = sum(any(eligible(c) for c in cc) for cc in entities.values())
    flags = sorted({f for c in chains for f in c['flags']})
    return {
        'meaning': 'Preliminary automated criteria only; validation/context review remains. No homology deduplication.',
        'sampling': 'Uniform random PDB entry sampling; entity/chain rates are descriptive, not uniform entity population estimates.',
        'selected_entries': sample_size,
        'processed_entries': len(records),
        'successful_entries': len(successful),
        'failed_entries': sum(r['status'] == 'failed' for r in records),
        'pending_entries': sample_size - len(records),
        'entry_at_least_one_candidate_passes_among_successes': rate(entry_pass, len(successful)),
        'entry_confirmed_passes_over_selected_lower_bound': rate(entry_pass, sample_size),
        'entity_at_least_one_chain_passes': rate(entity_pass, len(entities)),
        'chain_passes': rate(sum(eligible(c) for c in chains), len(chains)),
        'entry_counts_by_method': dict(Counter(
            '|'.join(r['chains'][0]['method']) for r in successful if r['chains'])),
        'entity_failures_all_copies_fail_each_criterion': {
            f: rate(sum(all(f in c['flags'] for c in cc) for cc in entities.values()), len(entities))
            for f in flags},
        'entity_assembly_classes': dict(Counter(
            '|'.join(sorted({c['assembly_class'] for c in cc})) for cc in entities.values())),
        'not_yet_automated': ['wwPDB validation report quality', 'ligand/nucleic acid dependence',
            'membrane/fusion/tag/fragment context', 'sequence clustering and split leakage'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--out', type=Path, default=ROOT / 'data/catalog/pilot1000')
    args = p.parse_args()
    if args.sample_size < 1:
        p.error('--sample-size must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    query = build_query()
    sample_path = out / 'sample.json'
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())
        if sample['seed'] != args.seed or len(sample['entries']) != args.sample_size or sample['query'] != query:
            p.error('existing snapshot settings differ; reuse original options or choose a new --out')
    else:
        print('Fetching candidate snapshot (40–128 residues, experimental proteins)...', flush=True)
        raw = fetch('https://search.rcsb.org/rcsbsearch/v2/query', query)
        response = json.loads(raw)
        grouped = defaultdict(list)
        for row in response.get('result_set', []):
            entity = row['identifier']
            grouped[entity.split('_', 1)[0]].append(entity)
        if len(grouped) < args.sample_size:
            p.error(f'only {len(grouped)} candidate entries available')
        if response.get('total_count') != len(response.get('result_set', [])):
            p.error('search response incomplete; refusing biased sampling')
        entries = random.Random(args.seed).sample(sorted(grouped), args.sample_size)
        (out / 'search_response.json').write_bytes(raw)
        sample = {'snapshot_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'seed': args.seed, 'query': query, 'population_entries': len(grouped),
            'population_entities': sum(len(set(v)) for v in grouped.values()),
            'entries': [{'entry_id': e, 'candidate_entities': sorted(set(grouped[e]))} for e in entries]}
        save(sample_path, sample)
    raw_dir = ROOT / 'data/raw/pdb'
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, selected in enumerate(sample['entries'], 1):
        entry = selected['entry_id']
        path = raw_dir / f'{entry}.cif.gz'
        url = f'https://files.rcsb.org/download/{entry}.cif.gz'
        record = dict(selected, url=url, path=path.relative_to(ROOT).as_posix())
        try:
            blob = path.read_bytes() if path.exists() else fetch(url)
            try:
                valid = gzip.decompress(blob).lstrip().startswith(b'data_')
            except (OSError, EOFError):
                valid = False
            if not valid:
                # Recover an interrupted/corrupt local cache once.
                blob = fetch(url)
                if not gzip.decompress(blob).lstrip().startswith(b'data_'):
                    raise ValueError('invalid mmCIF gzip response')
            temporary = path.with_suffix('.gz.tmp')
            temporary.write_bytes(blob)
            temporary.replace(path)
            record.update(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
            chain_rows = audit(record)
            expected = {x.split('_', 1)[1] for x in selected['candidate_entities']}
            if {c['entity_id'] for c in chain_rows} != expected:
                raise ValueError('candidate entity missing from parsed chains')
            if any(not 40 <= c['sequence_length'] <= 128 for c in chain_rows):
                raise ValueError('search/file sequence-length mismatch')
            record.update(status='audited', chains=chain_rows)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        records.append(record)
        save(out / 'results.json', {'records': records})
        summary = summarize(records, len(sample['entries']))
        summary['snapshot_utc'] = sample['snapshot_utc']
        summary['seed'] = sample['seed']
        save(out / 'summary.json', summary)
        print(f"[{i}/{len(sample['entries'])}] {entry}: {record['status']} | "
              f"entry provisional pass {summary['entry_at_least_one_candidate_passes_among_successes']['percent']}%",
              flush=True)
        time.sleep(0.1)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Results: {out}', flush=True)
    if summary['failed_entries']:
        print('Failures remain unknown, not biological rejections. Rerun the same command to retry.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
