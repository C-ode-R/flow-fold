"""Short full-model train-path benchmark; no checkpoint selection or test access."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from foldflow.dataset import ShardDataset,audit_split_isolation,atomic_json
from foldflow.folding import FoldingModel,invariant_loss,augment_target
from foldflow.runtime import validate_device,loader,move,predict,synchronize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--updates',type=int,default=10)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('outputs/preflight.json'))
    args=p.parse_args()
    if min(args.batch_size,args.updates)<1 or min(args.warmup,args.workers)<0:p.error('invalid counts')
    validate_device(args.device,args.precision);torch.set_num_threads(4);torch.manual_seed(20261024)
    train=ShardDataset(args.cache,'train');val=ShardDataset(args.cache,'validation');audit_split_isolation(train,val)
    model=FoldingModel(esm_dim=train.index['embedding']['dim'],hidden=128,pair_dim=64,pair_layers=3,geometry_layers=3,recycles=3).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    source=loader(train,args.batch_size,20261024,0,args.workers,args.device,True)
    iterator=iter(source)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    samples=0;started=None
    for i in range(args.warmup+args.updates):
        if i==args.warmup:synchronize(args.device);started=time.monotonic()
        try:cpu,meta=next(iterator)
        except StopIteration:iterator=iter(source);cpu,meta=next(iterator)
        batch=move(cpu,args.device);target=augment_target(batch,20261024+i)
        optimizer.zero_grad(set_to_none=True)
        x,r=predict(model,batch,args.precision);loss,_=invariant_loss(x,r,target)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite loss/gradient')
        optimizer.step()
        if i>=args.warmup:samples+=len(meta)
    synchronize(args.device);elapsed=time.monotonic()-started
    result=dict(device=args.device,device_name=torch.cuda.get_device_name() if args.device=='cuda' else args.device,
        torch=str(torch.__version__),cuda_build=torch.version.cuda,precision=args.precision,batch_size=args.batch_size,
        updates=args.updates,seconds_per_update=elapsed/args.updates,samples_per_second=samples/elapsed,
        peak_cuda_memory_GB=torch.cuda.max_memory_allocated()/1e9 if args.device=='cuda' else None,
        scope='Short execution/throughput check on train inputs; no generalization claim')
    args.out.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
