"""Offline parallel screening of indexed 40–128 aa entities; preserve ligand contacts.

No network, no archive mutation. Outputs are preliminary candidates, not a split.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from audit_pdb import audit, rows, number
from sequence_cif import read_categories
from pilot_pdb import save

META = ('_entry.', '_entity.', '_entity_poly.', '_entity_poly_seq.', '_struct_asym.',
        '_exptl.', '_refine.', '_em_3d_reconstruction.', '_pdbx_struct_assembly.',
        '_pdbx_struct_assembly_gen.', '_pdbx_struct_oper_list.', '_chem_comp.')


class AssemblyAnnotationMismatch(ValueError):
    pass


def operations(expression):
    """Expand mmCIF operator products, preserving right-to-left application."""
    expression = expression.replace(' ', '')
    groups = re.findall(r'\(([^()]*)\)', expression)
    if groups:
        if ''.join('('+g+')' for g in groups) != expression:
            raise ValueError(f'unsupported operator expression: {expression}')
    else:
        groups = [expression]
    expanded = []
    for group in groups:
        values = []
        for term in group.split(','):
            if re.fullmatch(r'\d+-\d+', term):
                left, right = map(int, term.split('-'))
                if left > right:
                    raise ValueError('reversed operator range')
                values.extend(str(x) for x in range(left, right+1))
            elif re.fullmatch(r'[A-Za-z0-9_]+', term):
                values.append(term)
            else:
                raise ValueError(f'unsupported operator: {term}')
        expanded.append(values)
    return itertools.product(*expanded)


def transforms(d):
    fields=['id']+[f'matrix[{i}][{j}]' for i in range(1,4) for j in range(1,4)]+[f'vector[{i}]' for i in range(1,4)]
    result={}
    for r in rows(d,'_pdbx_struct_oper_list',fields):
        m=np.eye(4)
        m[:3,:3]=[[float(r[f'matrix[{i}][{j}]']) for j in range(1,4)] for i in range(1,4)]
        m[:3,3]=[float(r[f'vector[{i}]']) for i in range(1,4)]
        result[r['id']]=m
    return result


def assembly_instances(d, assembly_id):
    ops=transforms(d)
    output=[]
    seen=set()
    for r in rows(d,'_pdbx_struct_assembly_gen',['assembly_id','oper_expression','asym_id_list']):
        if r['assembly_id'] != assembly_id:
            continue
        for product in operations(r['oper_expression']):
            m=np.eye(4)
            for op in product:
                m=m@ops[op]
            for asym in r['asym_id_list'].split(','):
                key=(asym,tuple(np.round(m.flatten(),8)))
                if key not in seen:
                    output.append((asym,list(product),m))
                    seen.add(key)
    return output


def selected_atoms(d, target_entities):
    fields=['label_asym_id','label_entity_id','label_seq_id','auth_seq_id','pdbx_PDB_ins_code',
            'label_comp_id','label_atom_id','label_alt_id','occupancy','type_symbol',
            'pdbx_PDB_model_num','Cartn_x','Cartn_y','Cartn_z']
    aa=rows(d,'_atom_site',fields)
    models=sorted({r['pdbx_PDB_model_num'] for r in aa},key=lambda x:int(x) if x.isdigit() else 0)
    if not models:
        raise ValueError('no retained atoms')
    model=models[0]
    protein=defaultdict(list)
    ligand=defaultdict(list)
    for a in aa:
        if a['pdbx_PDB_model_num'] != model or not (number(a['occupancy']) or 0)>0:
            continue
        xyz=np.array([float(a[k]) for k in ('Cartn_x','Cartn_y','Cartn_z')])
        if not np.isfinite(xyz).all():
            raise ValueError('nonfinite atom coordinate')
        a=dict(a,xyz=xyz)
        if a['label_entity_id'] in target_entities and a['label_seq_id'] not in ('.','?'):
            protein[(a['label_asym_id'],a['label_seq_id'])].append(a)
        elif a['type_symbol'].upper() not in ('H','D'):
            # entity type was already filtered to non-polymer/branched.
            ligand[(a['label_asym_id'],a['label_comp_id'],a['auth_seq_id'],a['pdbx_PDB_ins_code'])].append(a)
    by_chain=defaultdict(list)
    for (chain,seq),atoms in protein.items():
        scores=defaultdict(float)
        for a in atoms:
            if a['label_alt_id'] not in ('.','?') and a['label_atom_id'] in ('N','CA','C','O'):
                scores[a['label_alt_id']]+=float(a['occupancy'])
        chosen=sorted(scores,key=lambda x:(-scores[x],x))[0] if scores else None
        # Side-chain-only altloc uses occupancy if backbone had no alternate.
        if chosen is None:
            for a in atoms:
                if a['label_alt_id'] not in ('.','?'):
                    scores[a['label_alt_id']]+=float(a['occupancy'])
            chosen=sorted(scores,key=lambda x:(-scores[x],x))[0] if scores else None
        by_chain[chain].extend(a for a in atoms if a['label_alt_id'] in ('.','?',chosen) and a['type_symbol'].upper() not in ('H','D'))
    return by_chain,ligand,model


def ligand_metadata(d, chain, target_entities, cutoff):
    protein,ligands,model=selected_atoms(d,target_entities)
    atoms=protein[chain]
    if not atoms:
        raise ValueError(f'no observed heavy atoms for {chain}')
    xyz=np.stack([a['xyz'] for a in atoms])
    descriptors={r['id']:r for r in rows(d,'_chem_comp',['id','name','type','formula'])}
    inventory=[]
    for key,aa in sorted(ligands.items()):
        asym,comp,auth,ins=key
        inventory.append(dict(label_asym_id=asym,component_id=comp,auth_seq_id=auth,
            insertion_code=ins,heavy_atoms=len(aa),**{k:v for k,v in descriptors.get(comp,{}).items() if k!='id'}))
    assembly_ids=sorted({r['assembly_id'] for r in rows(d,'_pdbx_struct_assembly_gen',['assembly_id','asym_id_list'])
                         if chain in r['asym_id_list'].split(',')})
    contacts=[]
    assembly_presence=[]
    for assembly_id in assembly_ids:
        instances=assembly_instances(d,assembly_id)
        target_instances=[(op,m) for asym,op,m in instances if asym==chain]
        if len(target_instances)!=1:
            raise AssemblyAnnotationMismatch(f'assembly {assembly_id}: annotated monomer but generated target copies={len(target_instances)}')
        target_op,target_m=target_instances[0]
        pxyz=xyz@target_m[:3,:3].T+target_m[:3,3]
        present=[]
        for key,aa in sorted(ligands.items()):
            asym,comp,auth,ins=key
            for lig_asym,op,m in instances:
                if lig_asym!=asym:
                    continue
                present.append(dict(label_asym_id=asym,component_id=comp,auth_seq_id=auth,
                                    insertion_code=ins,operators=op))
                lxyz=np.stack([a['xyz'] for a in aa])@m[:3,:3].T+m[:3,3]
                # Chunk to avoid materializing chain_atoms x huge_ligand_atoms.
                minima=np.full(len(atoms),np.inf)
                for start in range(0,len(lxyz),128):
                    distance=np.linalg.norm(pxyz[:,None,:]-lxyz[None,start:start+128,:],axis=-1)
                    minima=np.minimum(minima,distance.min(axis=1))
                residue_dist={}
                for a,dist in zip(atoms,minima):
                    if dist<=cutoff:
                        idx=int(a['label_seq_id'])
                        residue_dist[idx]=min(residue_dist.get(idx,float('inf')),float(dist))
                if residue_dist:
                    contacts.append(dict(assembly_id=assembly_id,target_operators=target_op,
                        ligand_operators=op,label_asym_id=asym,component_id=comp,auth_seq_id=auth,
                        insertion_code=ins,minimum_distance_A=round(float(minima.min()),4),
                        contact_residues=[dict(label_seq_id=i,min_heavy_atom_distance_A=round(v,4)) for i,v in sorted(residue_dist.items())]))
        assembly_presence.append(dict(assembly_id=assembly_id,ligand_instances=present))
    # Coordinates remain in original deposited reference frame, not transformed assembly frame.
    backbone=[dict(label_seq_id=int(a['label_seq_id']),auth_seq_id=a['auth_seq_id'],
        insertion_code=a['pdbx_PDB_ins_code'],residue_name=a['label_comp_id'],
        atom=a['label_atom_id'],altloc=a['label_alt_id'],occupancy=float(a['occupancy']),
        xyz_A=a['xyz'].tolist()) for a in atoms if a['label_atom_id'] in ('N','CA','C','O')]
    return dict(model_id=model,contact_cutoff_A=cutoff,contact_definition='any selected protein heavy atom to any positive-occupancy ligand heavy atom, within biological assembly',
        ligand_altloc_policy='union of positive-occupancy conformers; contacts do not prove binding/dependence',
        entry_has_nonwater_ligand=bool(inventory),entry_ligands=inventory,
        biological_assembly_ligands=assembly_presence,has_contacting_ligand=bool(contacts),
        ligand_contacts=contacts,backbone_atoms=backbone)


def inspect_entry(task):
    path,entities,cutoff=task
    entry=entities[0]['entry_id']
    result=dict(entry_id=entry,path=path,entities=entities,chains=[],status='ok')
    try:
        with gzip.open(path,'rt') as stream:
            meta=read_categories(stream,META)
        if meta.get('_entry.id') != [entry]:
            raise ValueError('index/structure entry ID mismatch')
        # Cheap eligibility before retaining any coordinates.
        skeleton=dict(meta)
        skeleton['_atom_site.pdbx_PDB_model_num']=['1']
        skeleton_rows=audit(dict(path=path,entry_id=entry,_parsed=skeleton,
            candidate_entities=[f"{entry}_{e['entity_id']}" for e in entities]))
        expected={e['entity_id']:e['length'] for e in entities}
        if {r['entity_id'] for r in skeleton_rows} != set(expected) or any(r['sequence_length'] != expected[r['entity_id']] for r in skeleton_rows):
            raise ValueError('index/structure candidate sequence mismatch')
        eligible={r['label_asym_id'] for r in skeleton_rows if not set(r['flags'])-{'backbone_coverage_lt_95pct'}}
        if not eligible:
            unmeasured={'complete_backbone_residues','backbone_coverage','oxygen_on_complete',
                'missing_backbone_label_seq_ids','model_count','altloc_atom_count',
                'peptide_geometry_flags','entry_nonwater_nonpolymer_components'}
            result['chains']=[dict({k:v for k,v in r.items() if k not in unmeasured},
                screen_stage='metadata',flags=[f for f in r['flags'] if f!='backbone_coverage_lt_95pct'],
                decision='review',geometry_checked=False) for r in skeleton_rows]
            if not skeleton_rows:
                raise ValueError('no candidate chains in structure')
            return result
        target_entities={e['entity_id'] for e in entities}
        ligand_entities={r['id'] for r in rows(meta,'_entity',['id','type']) if r['type'] in ('non-polymer','branched')}
        def keep_atom(a):
            return (a.get('_atom_site.label_entity_id') in target_entities or
                    (a.get('_atom_site.label_entity_id') in ligand_entities and a.get('_atom_site.label_comp_id') not in ('HOH','DOD')))
        with gzip.open(path,'rt') as stream:
            parsed=read_categories(stream,META+('_atom_site.',),keep_atom)
        chain_rows=audit(dict(path=path,entry_id=entry,_parsed=parsed,
            candidate_entities=[f"{entry}_{e['entity_id']}" for e in entities]))
        for row in chain_rows:
            row.update(screen_stage='geometry',geometry_checked=True,decision='review' if row['flags'] else 'candidate')
            if not row['flags']:
                try:
                    row['ligand_metadata']=ligand_metadata(parsed,row['label_asym_id'],target_entities,cutoff)
                except AssemblyAnnotationMismatch as exc:
                    row['flags'].append('assembly_generated_copy_mismatch')
                    row.update(decision='review',assembly_consistency_note=str(exc))
                except Exception as exc:
                    row.update(decision='error',ligand_error=f'{type(exc).__name__}: {exc}')
            result['chains'].append(row)
        if {r['entity_id'] for r in chain_rows} != target_entities:
            raise ValueError('index/structure entity mismatch')
    except Exception as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}',chains=[])
    return result


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    code_files=['screen_archive.py','sequence_cif.py','audit_pdb.py']
    version=hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in code_files)+str(args.contact_cutoff).encode()).hexdigest()
    source=sqlite3.connect(f'file:{args.index.resolve()}?mode=ro',uri=True)
    jobs=[]
    inventory=[]
    source_errors=[]
    all_entities=0
    for path,size,mtime,payload in source.execute('SELECT path,size,mtime,payload FROM files ORDER BY path'):
        record=json.loads(payload)
        p=Path(path)
        if not p.is_file() or (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):
            source_errors.append(dict(path=path,error='missing or changed source file; rebuild index'))
            continue
        inventory.append((path,size,mtime))
        if record['status']!='ok':
            source_errors.append(dict(path=path,error=record.get('error','index failed')))
            continue
        all_entities+=len(record['entities'])
        entities=[e for e in record['entities'] if 40<=e['length']<=128]
        if entities:
            jobs.append((path,entities,args.contact_cutoff))
    source.close()
    save(out/'source_summary.json',dict(index_files=len(inventory)+len(source_errors),protein_entities=all_entities,
        candidate_entries=len(jobs),candidate_entities=sum(len(j[1]) for j in jobs),errors=source_errors))
    if source_errors:
        raise ValueError('source index contains missing/changed/failed files; see source_summary.json')
    fingerprint=hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    config=dict(code_sha256=version,inventory_sha256=fingerprint,contact_cutoff_A=args.contact_cutoff)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('code/settings/archive changed; use a new --out to avoid mixing versions')
    save(out/'config.json',config)
    db=sqlite3.connect(out/'screen.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results(path TEXT PRIMARY KEY,payload TEXT)')
    done={p for p,v in db.execute('SELECT path,payload FROM results')
          if json.loads(v)['status']=='ok' and not any(c['decision']=='error' for c in json.loads(v)['chains'])}
    pending=[j for j in jobs if j[0] not in done]
    print(f'Candidate entries: {len(jobs)}; entities: {sum(len(j[1]) for j in jobs)}; pending: {len(pending)}; workers: {args.workers}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=iter(pending)
        futures=set()
        for _ in range(args.workers*2):
            task=next(tasks,None)
            if task is not None:futures.add(pool.submit(inspect_entry,task))
        count=0
        while futures:
            finished,futures=wait(futures,return_when=FIRST_COMPLETED)
            for future in finished:
                result=future.result()  # Broken workers fail loudly; no silent permanent wait.
                db.execute('INSERT OR REPLACE INTO results VALUES(?,?)',(result['path'],json.dumps(result)))
                db.commit()
                count+=1
                if count%25==0 or count==len(pending):print(f'Screened {count}/{len(pending)} remaining entries',flush=True)
                task=next(tasks,None)
                if task is not None:futures.add(pool.submit(inspect_entry,task))
    summary=export_results(db,out,len(inventory),all_entities,len(jobs),sum(len(j[1]) for j in jobs))
    if summary['errors']:raise ValueError('screen errors remain; inspect errors.jsonl and retry same command')


def export_results(db,out,indexed_files,protein_entities,candidate_entries,screened_entities):
    decision_counts=Counter()
    flags=Counter()
    accepted_entities=set()
    errors=[]
    with (out/'candidates.jsonl').open('w') as accepted,(out/'review.jsonl').open('w') as review,(out/'errors.jsonl').open('w') as bad:
        for path,value in db.execute('SELECT path,payload FROM results ORDER BY path'):
            result=json.loads(value)
            if result['status']=='error':
                errors.append(path);bad.write(value+'\n');continue
            for chain in result['chains']:
                decision_counts[chain['decision']]+=1
                flags.update(chain['flags'])
                record=dict(chain,source_path=path,entity=next(e for e in result['entities'] if e['entity_id']==chain['entity_id']))
                handle=accepted if chain['decision']=='candidate' else bad if chain['decision']=='error' else review
                handle.write(json.dumps(record)+'\n')
                if chain['decision']=='candidate':accepted_entities.add((chain['entry_id'],chain['entity_id']))
                elif chain['decision']=='error':errors.append(path)
    summary=dict(indexed_files=indexed_files,protein_entities=protein_entities,candidate_entries=candidate_entries,
        screened_entities=screened_entities,chain_decisions=dict(decision_counts),
        preliminary_accepted_entities=len(accepted_entities),
        preliminary_entity_percent=round(100*len(accepted_entities)/screened_entities,3) if screened_entities else None,
        chain_flag_counts=dict(flags),errors=len(errors),
        limitations=['Not sequence/homology deduplicated; no train/validation/test split.',
            'No wwPDB validation report gate; no biological independence guarantee.',
            'Ligand presence/contact is geometric metadata, not binding or functional dependence.',
            'Metadata-rejected chains have not had geometry inspected.',
            'Only indexed files checked; index must cover the completed archive.'])
    save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=Path('data/catalog/local1000/index.sqlite'))
    p.add_argument('--out',type=Path,default=Path('data/catalog/full_screen_v1'))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--contact-cutoff',type=float,default=4.5)
    args=p.parse_args()
    if args.workers<1 or not 0<args.contact_cutoff<20:p.error('invalid workers/contact cutoff')
    run(args)
