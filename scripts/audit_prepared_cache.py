"""Audit CPU cache geometry/split alignment and write a GPU transfer manifest.

Train/validation only. No test index, native file or cache is read.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from foldflow.dataset import ShardDataset,audit_split_isolation,sha256,atomic_json
from foldflow.geometry import backbone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/cache150'))
    p.add_argument('--dataset',type=Path,default=Path('data/dataset_v1_final'))
    p.add_argument('--out',type=Path,default=Path('data/gpu_ready'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'READY.json').unlink(missing_ok=True)
    torch.set_num_threads(4)
    datasets={s:ShardDataset(args.cache,s) for s in ('train','validation')}
    audit_split_isolation(datasets['train'],datasets['validation'])
    report=dict(scope='train/validation only; no test access',splits={},errors=[],warnings=[],
        embedding=datasets['train'].index['embedding'],dataset_id=datasets['train'].index['dataset_id'])
    transfer={'config.json':sha256(args.cache/'config.json')}
    with (args.out/'preprocessing_manifest.jsonl').open('w') as manifest:
        for split,dataset in datasets.items():
            source=args.dataset/(split+'.jsonl')
            if sha256(source)!=dataset.index['source_sha256']:raise ValueError('cache/source fingerprint mismatch')
            rows={r['sequence_id']:r for r in map(json.loads,source.open())}
            if set(rows)!={r['id'] for r in dataset.records}:raise ValueError('source/cache record set mismatch')
            lengths=[];coverage=[];observed_atoms=Counter();ligands=Counter();contact_count=0
            max_orth=0.;min_det=1.;max_det=1.;ideal_sum=0.;ideal_count=0
            for record_index,ref in enumerate(dataset.records):
                record=dataset[record_index]
                length=len(record['sequence']);frame=record['frame_mask'];obs=record['observed']
                label=record['id'];valid_count=int(frame.sum())
                errors=[]
                for key in ('embedding','x','r','atoms'):
                    if not torch.isfinite(record[key]).all():errors.append(f'nonfinite {key}')
                source_row=rows[label]
                if record['sequence']!=source_row['entity']['sequence'] or record['cluster_id']!=source_row['cluster_id']:
                    errors.append('source sequence/cluster mismatch')
                if record['embedding'].shape!=(length,dataset.index['embedding']['dim']):errors.append('embedding shape')
                if not 40<=length<=128 or valid_count<3 or valid_count/length<.95:errors.append('invalid sequence/frame count or coverage below 95%')
                if not torch.equal(frame,obs[:,:3].all(-1)):errors.append('frame/atom mask mismatch or degenerate native frame')
                if errors:report['errors'].append(dict(id=label,reasons=errors));continue
                rotation=record['r'][frame]
                orth=float((rotation.transpose(-1,-2)@rotation-torch.eye(3)).abs().max())
                determinant=torch.linalg.det(rotation)
                local=torch.einsum('lij,laj->lai',rotation.transpose(-1,-2),record['atoms'][frame,:3]-record['x'][frame,None,:])
                if (local[:,2,1:].abs().max()>1e-4 or local[:,0,2].abs().max()>1e-4 or
                    not (local[:,2,0]>0).all() or not (local[:,0,1]>0).all()):
                    report['errors'].append(dict(id=label,reasons=['native frame axis/plane convention mismatch']))
                max_orth=max(max_orth,orth);min_det=min(min_det,float(determinant.min()));max_det=max(max_det,float(determinant.max()))
                if orth>1e-4 or ((determinant-1).abs()>1e-4).any():report['errors'].append(dict(id=label,reasons=['native frame not proper rotation']))
                if length>1:
                    links=record['links'][:-1]
                    distance=(record['atoms'][:-1,2]-record['atoms'][1:,0]).norm(dim=-1)*10
                    if (links&(~frame[:-1]|~frame[1:]| (distance<1.1)|(distance>1.8))).any():
                        report['errors'].append(dict(id=label,reasons=['peptide link/mask inconsistency']))
                reconstructed=backbone(record['x'][None],record['r'][None])[0]
                delta=(reconstructed-record['atoms'][:,:3]).norm(dim=-1)[frame]*10
                ideal_sum+=float(delta.square().sum());ideal_count+=delta.numel()
                ratio=valid_count/length;lengths.append(length);coverage.append(ratio)
                observed_atoms.update({a:int(obs[:,i].sum()) for i,a in enumerate(('N','CA','C','O'))})
                metadata=rows[label]['ligand_metadata']
                ids=sorted({r['component_id'] for r in metadata['entry_ligands']})
                ligands.update(ids);contact_count+=bool(metadata['has_contacting_ligand'])
                manifest.write(json.dumps(dict(id=label,split=split,cluster_id=record['cluster_id'],length=length,
                    frame_coverage=ratio,missing_backbone_label_seq_ids=(~frame).nonzero().flatten().add(1).tolist(),
                    ligand_component_ids=ids,has_contacting_ligand=metadata['has_contacting_ligand'],
                    ligand_contacts=metadata['ligand_contacts']))+'\n')
            report['splits'][split]=dict(count=len(dataset),clusters=len({r['cluster_id'] for r in dataset.records}),
                min_length=min(lengths) if lengths else None,max_length=max(lengths) if lengths else None,mean_length=sum(lengths)/len(lengths) if lengths else None,
                min_frame_coverage=min(coverage) if coverage else None,mean_frame_coverage=sum(coverage)/len(coverage) if coverage else None,
                observed_atom_counts=dict(observed_atoms),ligand_component_counts=dict(ligands),
                contacting_ligand_records=contact_count,max_frame_orthogonality_error=max_orth,
                min_frame_determinant=min_det,max_frame_determinant=max_det,
                ideal_N_CA_C_rms_error_A=(ideal_sum/max(1,ideal_count))**.5,
                index_sha256=dataset.fingerprint)
            transfer[f'{split}/index.json']=dataset.fingerprint
            for shard in dataset.index['shards']:transfer[f'{split}/{shard["file"]}']=shard['sha256']
    report['passed']=not report['errors']
    report['transfer_bytes']=sum((args.cache/name).stat().st_size for name in transfer)
    report['limitations']=['No full wwPDB validation, remote-homology exclusion or all-atom physical validation.',
        'Ligand contacts are metadata, not a model input or proof of binding dependence.']
    atomic_json(args.out/'audit.json',report)
    if not report['passed']:raise ValueError('cache QA errors; inspect audit.json; no ready marker created')
    atomic_json(args.out/'cache_checksums.json',transfer)
    atomic_json(args.out/'READY.json',dict(cache_path=str(args.cache.resolve()),splits=['train','validation'],
        records={s:len(d) for s,d in datasets.items()},audit_sha256=sha256(args.out/'audit.json'),
        cache_checksum_manifest_sha256=sha256(args.out/'cache_checksums.json')))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
