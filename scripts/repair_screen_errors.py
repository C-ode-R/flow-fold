"""Back up and repair only the two known assembly inconsistency errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from screen_archive import inspect_entry, export_results
from pilot_pdb import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('data/catalog/full_screen_v1'))
    args=p.parse_args()
    out=args.out.resolve()
    config=json.loads((out/'config.json').read_text())
    old_summary=json.loads((out/'summary.json').read_text())
    db=sqlite3.connect(out/'screen.sqlite')
    jobs=[]
    for path,payload in db.execute('SELECT path,payload FROM results'):
        r=json.loads(payload)
        if r['status']=='error':raise ValueError('unexpected entry error; manual review required')
        errors=[c for c in r['chains'] if c['decision']=='error']
        if errors:
            if not all(c.get('ligand_error')=='ValueError: expected one target chain instance in monomer assembly 1' for c in errors):
                raise ValueError('unexpected error type; refusing migration')
            jobs.append((path,r['entities'],config['contact_cutoff_A']))
    if not jobs:
        print('No known errors left; no changes made.');return
    backup=out/'before_assembly_repair'
    backup.mkdir(exist_ok=False)
    target=sqlite3.connect(backup/'screen.sqlite')
    db.backup(target);target.close()
    for name in ('config.json','summary.json','errors.jsonl'):
        (backup/name).write_bytes((out/name).read_bytes())
    for job in jobs:
        r=inspect_entry(job)
        if r['status']!='ok' or any(c['decision']=='error' for c in r['chains']):
            raise ValueError(f'repair failed: {job[0]}; original database not changed')
        if not all(c['decision']=='review' and 'assembly_generated_copy_mismatch' in c['flags'] for c in r['chains']):
            raise ValueError('repair must only turn known errors into assembly review')
        db.execute('UPDATE results SET payload=? WHERE path=?',(json.dumps(r),r['path']))
    db.commit()
    summary=export_results(db,out,old_summary['indexed_files'],old_summary['protein_entities'],old_summary['candidate_entries'],old_summary['screened_entities'])
    code_files=['screen_archive.py','sequence_cif.py','audit_pdb.py']
    config['code_sha256']=hashlib.sha256(b''.join(Path(__file__).with_name(x).read_bytes() for x in code_files)+str(config['contact_cutoff_A']).encode()).hexdigest()
    save(out/'config.json',config)
    save(out/'assembly_repair.json',dict(repaired_entries=[j[0] for j in jobs],backup=str(backup),
        change='Only previously failed assembly-generated-copy mismatches moved to review; other saved entry results retained.'))
    print(f'Repaired {len(jobs)} entries; errors remaining: {summary["errors"]}')


if __name__=='__main__':main()
