"""Train-only small-set overfit check. No validation/test loading or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from foldflow.data import collate
from foldflow.model import FrameFlow
from foldflow.training import flow_sample,loss,generate,structure_metrics
from foldflow.geometry import backbone_with_oxygen

AA3=dict(zip('ACDEFGHIKLMNPQRSTVWY','ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split()))


def export_pdb(path,atoms,sequence):
    atoms=atoms.detach().cpu().numpy()*10
    with path.open('w') as f:
        serial=1
        for i,aa in enumerate(sequence):
            for j,name in enumerate(('N','CA','C','O')):
                if name=='O' and i==len(sequence)-1:continue
                x,y,z=atoms[i,j]
                f.write(f'ATOM  {serial:5d} {name:^4s} {AA3[aa]:>3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}\n')
                serial+=1
        f.write('TER\nEND\n')


@torch.no_grad()
def evaluate(model,records,args,step):
    model.eval();rows=[]
    for i,record in enumerate(records):
        b=collate([record],args.device)
        x,r=generate(model,b,args.ode_steps,seed=args.seed+10000+i)
        metrics=structure_metrics(x[0],b['x'][0],b['frame_mask'][0])
        rows.append(dict(id=record['id'],**metrics))
        if step==args.steps:
            export_pdb(args.out/(record['id']+'.pdb'),backbone_with_oxygen(x,r)[0],record['sequence'])
    result=dict(step=step,scope='train-only overfit, one fixed generation seed per sequence',
        mean_ca_rmsd_A=float(np.mean([r['ca_rmsd_A'] for r in rows])),
        mean_ca_lddt=float(np.mean([r['ca_lddt'] for r in rows])),records=rows)
    (args.out/f'eval_{step:06d}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    model.train()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/mps_pilot/cache.pt'))
    p.add_argument('--out',type=Path,default=Path('outputs/mps_overfit16'))
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--ode-steps',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.batch_size,args.eval_every,args.ode_steps)<1 or args.hidden%4:
        p.error('positive step/batch sizes and hidden divisible by 4 required')
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this process; run on the Apple Silicon Mac with an MPS-enabled PyTorch. No silent CPU fallback.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);random.seed(args.seed)
    rng=np.random.default_rng(args.seed)
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    records=cache['records']
    if not 1<=len(records)<=50 or any(r['split']!='train' for r in records):raise ValueError('cache must contain 1..50 train records only')
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=args.out/'last.pt'
    if checkpoint.exists() and not args.resume:raise ValueError('output exists; use --resume or a fresh --out')
    model=FrameFlow(esm_dim=records[0]['embedding'].shape[-1],hidden=args.hidden).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    fingerprint=hashlib.sha256(args.cache.read_bytes()).hexdigest()
    start=0
    if args.resume:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['cache_sha256']!=fingerprint or state['config']!=model.config:raise ValueError('resume cache/model mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng'];start=state['step']
        torch.set_rng_state(state['torch_rng'])
    run_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    run_config.update(model=model.config,cache_sha256=fingerprint,cache_provenance=cache['provenance'],
        parameters=sum(p.numel() for p in model.parameters()),scope='train-only overfit; no generalization claim')
    (args.out/'config.json').write_text(json.dumps(run_config,indent=2)+'\n')
    print(json.dumps(dict(device=args.device,torch=torch.__version__,examples=len(records),parameters=run_config['parameters'])),flush=True)
    evaluate(model,records,args,start)
    begun=time.monotonic()
    with (args.out/'history.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(start+1,args.steps+1):
            ids=rng.choice(len(records),size=args.batch_size,replace=len(records)<args.batch_size)
            batch=collate([records[int(i)] for i in ids],args.device)
            state=flow_sample(batch,rng)
            optimizer.zero_grad(set_to_none=True)
            total,terms=loss(model,batch,state)
            if not torch.isfinite(total):raise FloatingPointError('nonfinite training loss')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            optimizer.step()
            if step%10==0 or step==1:
                row=dict(step=step,total=float(total.detach().cpu()),elapsed_s=time.monotonic()-begun,
                    **{k:float(v.detach().cpu()) for k,v in terms.items()})
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                evaluate(model,records,args,step)
                snapshot=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    config=model.config,step=step,cache_sha256=fingerprint,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
                temporary=args.out/'last.tmp.pt';torch.save(snapshot,temporary);temporary.replace(checkpoint)


if __name__=='__main__':main()
