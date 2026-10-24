"""Offline train/validation diagnosis of a saved run; never reads test."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foldflow.dataset import ShardDataset,audit_split_isolation,atomic_json
from foldflow.folding import FoldingModel
from foldflow.runtime import evaluate,code_fingerprint,validate_device


def summarize(rows):
    d=np.array([r['ca_lddt'] for r in rows]);r=np.array([r['ca_rmsd_A'] for r in rows])
    return dict(count=len(rows),mean_lddt=float(d.mean()),median_lddt=float(np.median(d)),
        mean_rmsd_A=float(r.mean()),median_rmsd_A=float(np.median(r)),
        rmsd_quantiles_A=np.quantile(r,[.1,.25,.5,.75,.9]).tolist(),
        lddt_ge_05=int((d>=.5).sum()),lddt_ge_06=int((d>=.6).sum()),
        lddt_ge_07=int((d>=.7).sum()),rmsd_lt2=int((r<2).sum()),
        rmsd_lt4=int((r<4).sum()),rmsd_gt10=int((r>10).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    validate_device(a.device,'fp32');torch.set_num_threads(4)
    saved=torch.load(a.run/'best.pt',map_location='cpu',weights_only=True)
    assert saved['contract']['code_sha256']==code_fingerprint(),'Model/training code changed'
    train=ShardDataset(a.cache,'train');val=ShardDataset(a.cache,'validation');audit_split_isolation(train,val)
    assert train.fingerprint==saved['contract']['train_fingerprint']
    assert val.fingerprint==saved['contract']['validation_fingerprint']
    model=FoldingModel(**saved['model_config']).to(a.device);model.load_state_dict(saved['model'])
    for dataset in [train,val]:
        print('Evaluating',dataset.split,len(dataset),flush=True)
        result,rows=evaluate(model,dataset,a.batch_size,a.device,'fp32')
        atomic_json(a.out/f'{dataset.split}_reevaluated.json',dict(summary=result,records=rows))
        print(json.dumps(result),flush=True)
    original=json.loads((a.run/f"validation_{saved['best_epoch']:04d}.json").read_text())
    records={r['id']:r for r in val.records}
    metadata={r['id']:r for line in a.metadata.read_text().splitlines() if (r:=json.loads(line))['split']=='validation'}
    rows=[dict(r,length=records[r['id']]['length'],contacting_ligand=metadata[r['id']]['has_contacting_ligand']) for r in original['records']]
    first=json.loads((a.run/'validation_0001.json').read_text());first={r['id']:r for r in first['records']}
    history=[json.loads(s) for s in (a.run/'history.jsonl').read_text().splitlines()]
    epochs=[r for r in history if r['event']=='validation']
    clusters={}
    for r in rows:clusters.setdefault(r['cluster_id'],[]).append(r['ca_lddt'])
    cluster_means=np.array([np.mean(v) for v in clusters.values()])
    rng=np.random.default_rng(20261024)
    bootstrap=cluster_means[rng.integers(len(clusters),size=(10000,len(clusters)))].mean(1)
    tr=json.loads((a.out/'train_reevaluated.json').read_text());vr=json.loads((a.out/'validation_reevaluated.json').read_text())
    score_deltas=[r['ca_lddt']-first[r['id']]['ca_lddt'] for r in rows]
    train_meta={r['id']:r for r in train.records}
    train_lengths={f'{lo}-{hi}':summarize([r for r in tr['records'] if lo<=train_meta[r['id']]['length']<=hi]) for lo,hi in [(40,64),(65,96),(97,128)]}
    report=dict(train_length_groups=train_lengths,best_epoch=saved['best_epoch'],best_step=saved['step'],validation=summarize(rows),
        train=summarize(tr['records']),mps_validation=vr['summary'],cuda_validation=original['summary'],
        cluster_count=len(clusters),cluster_macro_lddt=float(cluster_means.mean()),
        cluster_macro_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
        improved_over_epoch1=sum(d>0 for d in score_deltas),mean_lddt_gain=float(np.mean(score_deltas)),
        lengths={f'{lo}-{hi}':summarize([r for r in rows if lo<=r['length']<=hi]) for lo,hi in [(40,64),(65,96),(97,128)] if any(lo<=r['length']<=hi for r in rows)},
        ligand_groups={str(flag):summarize([r for r in rows if r['contacting_ligand']==flag]) for flag in [False,True] if any(r['contacting_ligand']==flag for r in rows)},
        length_lddt_pearson=float(np.corrcoef([r['length'] for r in rows],[r['ca_lddt'] for r in rows])[0,1]),
        best_examples=sorted(rows,key=lambda r:r['ca_lddt'],reverse=True)[:10],
        worst_examples=sorted(rows,key=lambda r:r['ca_lddt'])[:10],
        epochs=epochs,validation_records=rows,test_read=False,
        limitations=['One seed; validation selected checkpoint, not held-out test performance.',
            'Cluster bootstrap is descriptive and does not correct checkpoint selection bias.',
            'Train and validation differ in length distribution; train/validation gaps are descriptive.',
            'Ligand/length associations are not causal. C-alpha metric is not all-atom lDDT.'])
    atomic_json(a.out/'analysis.json',report)
    print(json.dumps({k:report[k] for k in ['validation','train','lengths','ligand_groups','cluster_macro_lddt','cluster_macro_bootstrap95','improved_over_epoch1']},indent=2),flush=True)

if __name__=='__main__':main()
