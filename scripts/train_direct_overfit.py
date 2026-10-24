"""Small-set deterministic control. Sequence-only forward, fixed-gauge or invariant losses."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from foldflow.data import collate
from foldflow.direct import DirectFrames,direct_loss
from foldflow.folding import FoldingModel,invariant_loss,augment_target
from foldflow.geometry import backbone_with_oxygen
from foldflow.training import structure_metrics
from scripts.train_mps_overfit import export_pdb
from scripts.diagnose_flow import angular_error_deg


@torch.no_grad()
def evaluate(model,batch,record,out,step):
    model.eval()
    x,r=model(batch['embedding'],batch['aa'],batch['mask'])
    metric=structure_metrics(x[0],batch['x'][0],batch['frame_mask'][0])
    result=dict(step=step,id=record['id'],scope='per-train-sequence deterministic direct regression',
        **metric,max_frame_rotation_error_deg=angular_error_deg(r,batch['r'],batch['frame_mask']),
        unaligned_ca_rmsd_A=float(torch.sqrt((x-batch['x']).square().sum(-1)[batch['frame_mask']].mean()).cpu())*10)
    # Operational debug target, not a published benchmark criterion.
    import numpy as np
    valid=batch['frame_mask'][0].cpu().numpy()
    a=x[0].cpu().numpy()[valid].copy();b=batch['x'][0].cpu().numpy()[valid].copy()
    a-=a.mean(0);b-=b.mean(0)
    u,_,vh=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vh)
    q=torch.tensor((u@fix@vh).T,dtype=r.dtype,device=r.device)
    result['max_aligned_frame_rotation_error_deg']=angular_error_deg(q@r,batch['r'],batch['frame_mask'])
    result['operational_overfit_target_met']=metric['ca_rmsd_A']<2 and metric['ca_lddt']>.8
    (out/f'eval_{step:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
    export_pdb(out/'prediction.pdb',backbone_with_oxygen(x,r)[0],record['sequence'])
    print(json.dumps(result),flush=True)
    model.train()
    return result


def schedule_index(step,count,seed):
    # One visit per record per shuffled epoch, reproducible after resume.
    epoch,offset=divmod(step-1,count)
    return random.Random(seed+epoch).sample(range(count),count)[offset]


def evaluate_set(model,batches,records,out,step):
    if len(records)==1:
        return evaluate(model,batches[0],records[0],out,step)
    results=[]
    for batch,record in zip(batches,records):
        folder=out/record['id'];folder.mkdir(exist_ok=True)
        results.append(evaluate(model,batch,record,folder,step))
    report=dict(step=step,scope='train-only multi-sequence direct overfit; one shared model',
        examples=len(results),
        mean_ca_rmsd_A=sum(r['ca_rmsd_A'] for r in results)/len(results),
        worst_ca_rmsd_A=max(r['ca_rmsd_A'] for r in results),
        mean_ca_lddt=sum(r['ca_lddt'] for r in results)/len(results),
        worst_ca_lddt=min(r['ca_lddt'] for r in results),
        max_frame_rotation_error_deg=max(r['max_frame_rotation_error_deg'] for r in results),
        max_aligned_frame_rotation_error_deg=max(r['max_aligned_frame_rotation_error_deg'] for r in results),
        ca_overfit_pass_count=sum(r['operational_overfit_target_met'] for r in results),
        ca_overfit_criteria='per sequence RMSD < 2 A and CA lDDT > 0.8; rotation errors reported separately',
        records=results)
    (out/f'eval_{step:06d}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/mps_pilot/cache1.pt'))
    p.add_argument('--out',type=Path,default=Path('outputs/mps_direct1'))
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--cpu-threads',type=int,help='CPU intra-op threads; CPU default is PyTorch default, MPS default is 4')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--architecture',choices=['direct','fold'],default='direct')
    p.add_argument('--loss-mode',choices=['fixed','invariant'],default='fixed')
    p.add_argument('--augment-rigid',action='store_true')
    p.add_argument('--pair-dim',type=int,default=32)
    p.add_argument('--pair-layers',type=int,default=2)
    p.add_argument('--geometry-layers',type=int,default=2)
    p.add_argument('--recycles',type=int,default=3)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.hidden%4 or min(args.steps,args.eval_every)<1:p.error('invalid hidden or step count')
    if args.augment_rigid and args.loss_mode!='invariant':p.error('rigid augmentation requires invariant loss')
    if min(args.pair_dim,args.pair_layers,args.geometry_layers,args.recycles)<1:p.error('positive model dimensions/layers required')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if args.cpu_threads is not None and args.cpu_threads<1:p.error('--cpu-threads must be positive')
    threads=args.cpu_threads if args.cpu_threads is not None else (torch.get_num_threads() if args.device=='cpu' else 4)
    torch.set_num_threads(threads);torch.manual_seed(args.seed)
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    records=cache['records']
    if not 1<=len(records)<=50 or any(r['split']!='train' for r in records):raise ValueError('1..50 train records required')
    if len({r['id'] for r in records})!=len(records) or len({r['sequence'] for r in records})!=len(records):raise ValueError('duplicate training sequences')
    batches=[collate([r],args.device) for r in records]
    dimensions={r['embedding'].shape[-1] for r in records}
    if len(dimensions)!=1:raise ValueError('mixed embedding dimensions')
    kwargs=dict(esm_dim=dimensions.pop(),hidden=args.hidden,pair_dim=args.pair_dim,pair_layers=args.pair_layers,geometry_layers=args.geometry_layers)
    model=(FoldingModel(recycles=args.recycles,**kwargs) if args.architecture=='fold' else DirectFrames(**kwargs)).to(args.device)
    objective=invariant_loss if args.loss_mode=='invariant' else direct_loss
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=args.out/'last.pt'
    fingerprint=hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if checkpoint.exists() and not args.resume:raise ValueError('existing output: choose new --out or --resume')
    start=0
    if args.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['cache_sha256']!=fingerprint or saved['config']!=model.config:raise ValueError('resume mismatch')
        if saved.get('architecture','direct')!=args.architecture or saved.get('loss_mode','fixed')!=args.loss_mode or saved.get('augment_rigid',False)!=args.augment_rigid:raise ValueError('resume architecture/loss mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(model=model.config,sequence_ids=[r['id'] for r in records],cache_sha256=fingerprint,
        parameters=sum(p.numel() for p in model.parameters()),torch_intraop_threads=torch.get_num_threads(),
        control='Sequence-only direct prediction; architecture, recycling count and loss mode recorded explicitly above.',
        limitation='Fixed mode supervises native gauge; invariant mode uses only local-frame/pair geometry. Train overfit success does not prove generalization or a flow implementation bug.')
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(device=args.device,examples=len(records),parameters=config['parameters'],cpu_threads=torch.get_num_threads())),flush=True)
    evaluate_set(model,batches,records,args.out,start)
    begun=time.monotonic()
    with (args.out/'history.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start+1,args.steps+1):
            selected=schedule_index(step,len(records),args.seed)
            batch=batches[selected]
            supervised=augment_target(batch,args.seed+step) if args.augment_rigid else batch
            optimizer.zero_grad(set_to_none=True)
            x,r=model(batch['embedding'],batch['aa'],batch['mask'])
            total,terms=objective(x,r,supervised)
            if not torch.isfinite(total):raise FloatingPointError('nonfinite loss')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            if step%10==0 or step==1:
                row=dict(step=step,sequence_id=records[selected]['id'],total=float(total.detach().cpu()),elapsed_s=time.monotonic()-begun,
                    **{k:float(v.detach().cpu()) for k,v in terms.items()})
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                evaluate_set(model,batches,records,args.out,step)
                saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    config=model.config,step=step,cache_sha256=fingerprint,architecture=args.architecture,loss_mode=args.loss_mode,augment_rigid=args.augment_rigid)
                temporary=args.out/'last.tmp.pt';torch.save(saved,temporary);temporary.replace(checkpoint)


if __name__=='__main__':main()
