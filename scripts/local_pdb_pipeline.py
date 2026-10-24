"""Offline archive -> sequence index -> random entity pilot. No network access."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import random
import sqlite3

from Bio.Data.PDBData import protein_letters_3to1
from sequence_cif import read_sequence_categories
from audit_pdb import audit, rows, STANDARD
from pilot_pdb import save, rate


def extract(task):
    path, size, mtime = task
    try:
        with gzip.open(path, 'rt') as stream:
            d = read_sequence_categories(stream)
        entry = d['_entry.id'][0]
        protein_ids = {r['entity_id'] for r in rows(d, '_entity_poly', ['entity_id', 'type'])
                       if r['type'] == 'polypeptide(L)'}
        seq = defaultdict(dict)
        for r in rows(d, '_entity_poly_seq', ['entity_id', 'num', 'mon_id']):
            if r['entity_id'] in protein_ids:
                seq[r['entity_id']].setdefault(int(r['num']), set()).add(r['mon_id'])
        entities = []
        for entity in sorted(protein_ids):
            residues = seq[entity]
            if not residues:
                raise ValueError(f'protein entity {entity} has no deposited sequence')
            if sorted(residues) != list(range(1, len(residues) + 1)):
                raise ValueError(f'non-contiguous entity_poly_seq numbering: {entity}')
            sequence = ''.join(protein_letters_3to1.get(next(iter(v)), 'X')
                               if len(v) == 1 else 'X' for _, v in sorted(residues.items()))
            entities.append(dict(entry_id=entry, entity_id=entity, sequence=sequence,
                length=len(residues), standard=all(len(v)==1 and v <= STANDARD for v in residues.values())))
        return path, size, mtime, dict(status='ok', entities=entities)
    except Exception as exc:
        return path, size, mtime, dict(status='error', error=f'{type(exc).__name__}: {exc}')


def bounded_results(pool, tasks, workers):
    source = iter(tasks)
    pending = set()
    for _ in range(workers * 2):
        task = next(source, None)
        if task is not None:
            pending.add(pool.submit(extract, task))
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            task = next(source, None)
            if task is not None:
                pending.add(pool.submit(extract, task))


def run(args):
    archive, out = args.archive.resolve(), args.out.resolve()
    if not archive.is_dir():
        raise ValueError(f'archive directory not found: {archive}')
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / 'index.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, payload TEXT)')
    cached = {p: (s,m,json.loads(v)['status']) for p,s,m,v in db.execute('SELECT * FROM files')}
    tasks, active = [], []
    for path in sorted(archive.rglob('*.cif.gz')):
        stat = path.stat()
        name = str(path)
        active.append((name,stat.st_size,stat.st_mtime_ns))
        if cached.get(name) != (stat.st_size,stat.st_mtime_ns,'ok'):
            tasks.append((name,stat.st_size,stat.st_mtime_ns))
    if not active:
        raise ValueError('no *.cif.gz files found')
    print(f'Archive files: {len(active)}; to index: {len(tasks)}; workers: {args.workers}', flush=True)
    # Single SQLite writer, bounded worker queues; each worker parses one entry at a time.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, (path,size,mtime,payload) in enumerate(bounded_results(pool,tasks,args.workers),1):
            db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?)',
                       (path,size,mtime,json.dumps(payload)))
            db.commit()
            if count % 100 == 0 or count == len(tasks):
                print(f'Indexed {count}/{len(tasks)} new/changed files', flush=True)
    active_names = {x[0] for x in active}
    candidates, errors, seen = [], [], set()
    entity_count = 0
    # Derive all outputs from current inventory only, excluding removed files in the DB.
    with (out/'protein_sequences.fasta.tmp').open('w') as fasta, (out/'candidates.jsonl.tmp').open('w') as candidate_file:
        for path,payload in db.execute('SELECT path,payload FROM files ORDER BY path'):
            if path not in active_names:
                continue
            payload=json.loads(payload)
            if payload['status'] != 'ok':
                errors.append(dict(path=path,error=payload['error']))
                continue
            for e in payload['entities']:
                key=(e['entry_id'],e['entity_id'])
                if key in seen:
                    raise ValueError(f'duplicate entry/entity in archive: {key}; use divided directory only')
                seen.add(key)
                entity_count+=1
                fasta.write(f">{e['entry_id']}_{e['entity_id']} length={e['length']} standard={e['standard']}\n{e['sequence']}\n")
                if 40 <= e['length'] <= 128:
                    record=dict(e,path=path)
                    candidates.append(record)
                    candidate_file.write(json.dumps(record)+'\n')
    for name in ('protein_sequences.fasta','candidates.jsonl'):
        (out/(name+'.tmp')).replace(out/name)
    fingerprint=hashlib.sha256(json.dumps(active).encode()).hexdigest()
    save(out/'index_summary.json',dict(archive=str(archive),files=len(active),
        protein_entities=entity_count,candidate_entities=len(candidates),errors=errors,
        inventory_sha256=fingerprint,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    if errors:
        raise ValueError(f'{len(errors)} files could not be indexed; inspect index_summary.json and rerun. No biased pilot created.')
    if len(candidates) < args.sample_size:
        raise ValueError(f'only {len(candidates)} eligible-length entities; need {args.sample_size}')
    candidates.sort(key=lambda e:(e['entry_id'],e['entity_id']))
    selected=random.Random(args.seed).sample(candidates,args.sample_size)
    sample=dict(seed=args.seed,unit='entry+entity, uniform without replacement; not sequence-deduplicated',
        inventory_sha256=fingerprint,entities=selected)
    sample_path=out/'sample.json'
    if sample_path.exists() and json.loads(sample_path.read_text()) != sample:
        raise ValueError('archive/seed/sample changed; choose new --out to preserve previous sample')
    save(sample_path,sample)
    grouped=defaultdict(list)
    for e in selected:
        grouped[e['path']].append(e)
    results, audit_errors=[],[]
    for i,(path,entities) in enumerate(grouped.items(),1):
        try:
            entry=entities[0]['entry_id']
            chain_rows=audit(dict(path=path,entry_id=entry,candidate_entities=[f"{entry}_{e['entity_id']}" for e in entities]))
            for e in entities:
                cc=[c for c in chain_rows if c['entity_id']==e['entity_id']]
                if not cc or any(c['sequence_length'] != e['length'] for c in cc):
                    raise ValueError(f"sequence/chain mismatch: {e['entity_id']}")
            results.extend(chain_rows)
        except Exception as exc:
            audit_errors.append(dict(path=path,error=f'{type(exc).__name__}: {exc}'))
        save(out/'audit.json',dict(records=results,errors=audit_errors))
        if i%10==0 or i==len(grouped):
            print(f'Audited {i}/{len(grouped)} sampled entries',flush=True)
    by_entity=defaultdict(list)
    for c in results:
        by_entity[(c['entry_id'],c['entity_id'])].append(c)
    passed=sum(any(not c['flags'] for c in cc) for cc in by_entity.values())
    flags=sorted({f for c in results for f in c['flags']})
    summary=dict(sampled_entities=len(selected),audited_entities=len(by_entity),
        unknown_entities=len(selected)-len(by_entity),
        preliminary_pass_among_audited=rate(passed,len(by_entity)),
        confirmed_pass_over_sample_lower_bound=rate(passed,len(selected)),
        criterion_failure_rates={f:dict(failed=sum(all(f in c['flags'] for c in cc) for cc in by_entity.values()), total=len(by_entity), percent=rate(sum(all(f in c['flags'] for c in cc) for cc in by_entity.values()),len(by_entity))['percent']) for f in flags},
        methods=dict(Counter('|'.join(cc[0]['method']) for cc in by_entity.values())),
        limitations=['Preliminary automated criteria, not final training acceptance.',
            'Entity-level uniform sample; identical/homologous sequences not deduplicated.',
            'wwPDB validation and biological context/ligand dependence remain unassessed.',
            'NMR first-model observation only; held out by method criterion.'],errors=audit_errors)
    save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    db.close()
    if audit_errors:
        raise ValueError('audit errors remain; see audit.json; unknowns are not biological rejections')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('data/catalog/local1000'))
    p.add_argument('--sample-size',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.workers<1 or args.sample_size<1:
        p.error('workers and sample size must be positive')
    run(args)


if __name__=='__main__':
    main()
