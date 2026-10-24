"""Train/validation only. Test evaluation lives in evaluate_test.py."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from foldflow.dataset import ShardDataset,audit_split_isolation,atomic_json,atomic_torch
from foldflow.folding import FoldingModel,invariant_loss,augment_target
from foldflow.runtime import validate_device,synchronize,loader,move,predict,evaluate,code_fingerprint,rng_state,restore_rng


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--warmup-updates',type=int,default=250)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--pair-dim',type=int,default=64)
    p.add_argument('--pair-layers',type=int,default=3)
    p.add_argument('--geometry-layers',type=int,default=3)
    p.add_argument('--recycles',type=int,default=3)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--max-updates',type=int,help='Benchmark cap on total updates; does not mark training complete')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for key in ['epochs','batch_size','cpu_threads','hidden','pair_dim','pair_layers','geometry_layers','recycles','save_every','log_every','patience']:
        if getattr(args,key)<1:p.error(f'{key} must be positive')
    if args.hidden%4 or args.workers<0 or args.warmup_updates<0 or args.lr<=0 or args.min_delta<0 or args.weight_decay<0:p.error('invalid dimensions/workers/lr')
    if args.max_updates is not None and args.max_updates<1:p.error('max updates must be positive')
    validate_device(args.device,args.precision)
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if args.device=='cuda':torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    train=ShardDataset(args.cache,'train');validation=ShardDataset(args.cache,'validation')
    audit_split_isolation(train,validation)
    model_config=dict(esm_dim=train.index['embedding']['dim'],hidden=args.hidden,pair_dim=args.pair_dim,
        pair_layers=args.pair_layers,geometry_layers=args.geometry_layers,recycles=args.recycles)
    model=FoldingModel(**model_config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    batches_per_epoch=math.ceil(len(train)/args.batch_size)
    total_updates=args.epochs*batches_per_epoch
    def schedule(update):
        if args.warmup_updates and update<args.warmup_updates:return (update+1)/args.warmup_updates
        phase=(update-args.warmup_updates)/max(1,total_updates-args.warmup_updates)
        return .1+.9*.5*(1+math.cos(math.pi*min(1,max(0,phase))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    contract=dict(model=model_config,train_fingerprint=train.fingerprint,validation_fingerprint=validation.fingerprint,
        embedding=train.index['embedding'],dataset_id=train.index['dataset_id'],seed=args.seed,
        epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,weight_decay=args.weight_decay,
        warmup_updates=args.warmup_updates,patience=args.patience,min_delta=args.min_delta,
        precision=args.precision,code_sha256=code_fingerprint())
    args.out.mkdir(parents=True,exist_ok=True)
    last=args.out/'last.pt'
    if not args.resume and any(args.out.iterdir()):raise ValueError('output must be empty; use --resume or new output')
    if (args.out/'TRAINING_COMPLETE.json').exists():raise ValueError('run already complete; frozen result will not be resumed')
    epoch,next_batch,step=0,0,0
    best,best_epoch,bad=-1.,None,0
    train_sum,train_count=0.,0
    if args.resume:
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['contract']!=contract:raise ValueError('resume config/data/code mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch,next_batch,step=saved['epoch'],saved['next_batch'],saved['step']
        best,best_epoch,bad=saved['best_score'],saved['best_epoch'],saved['bad_epochs']
        train_sum,train_count=saved['train_sum'],saved['train_count']
        restore_rng(saved['rng'])
    run_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    run_config.update(contract=contract,torch_version=str(torch.__version__),
        cuda_build=torch.version.cuda,train_count=len(train),validation_count=len(validation),
        model_parameters=sum(p.numel() for p in model.parameters()),
        device_name=torch.cuda.get_device_name() if args.device=='cuda' else args.device,
        scope='Fresh model; frozen ESM; train updates; validation checkpoint selection; test never loaded')
    atomic_json(args.out/'config.json',run_config)
    stop=[False]
    def request_stop(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def snapshot(cursor_epoch,cursor_batch,selected=False):
        return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),contract=contract,model_config=model_config,epoch=cursor_epoch,next_batch=cursor_batch,step=step,
            best_score=best,best_epoch=best_epoch,bad_epochs=bad,train_sum=train_sum,train_count=train_count,
            rng=rng_state(),selected_by='validation_mean_ca_lddt' if selected else None)
    print(json.dumps(run_config),flush=True)
    model.train();completed=False;reason='benchmark_cap';stop_run=False
    synchronize(args.device);start=time.monotonic();timed_samples=0;timed_steps=0
    train_loader=loader(train,args.batch_size,args.seed,epoch,args.workers,args.device,True)
    validation_loader=loader(validation,args.batch_size,0,0,args.workers,args.device,False)
    with (args.out/'history.jsonl').open('a' if args.resume else 'w') as history:
        while epoch<args.epochs:
            train_loader.batch_sampler.set_epoch(epoch)
            if len(train_loader)!=batches_per_epoch:raise ValueError('batch count mismatch')
            for batch_index,(cpu,metas) in enumerate(train_loader):
                if batch_index<next_batch:continue
                if args.max_updates is not None and step>=args.max_updates:
                    atomic_torch(last,snapshot(epoch,batch_index));stop_run=True;break
                batch=move(cpu,args.device)
                supervised=augment_target(batch,args.seed+step)
                optimizer.zero_grad(set_to_none=True)
                x,r=predict(model,batch,args.precision)
                value,terms=invariant_loss(x,r,supervised)
                if not torch.isfinite(value):raise FloatingPointError('nonfinite train loss')
                value.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite train gradient')
                optimizer.step();scheduler.step();step+=1
                next_batch=batch_index+1
                train_sum+=float(value.detach().cpu())*len(metas);train_count+=len(metas)
                timed_samples+=len(metas);timed_steps+=1
                if step%args.log_every==0 or step==1:
                    synchronize(args.device);elapsed=time.monotonic()-start
                    event=dict(event='train',epoch=epoch+1,step=step,batch=next_batch,loss=float(value.detach().cpu()),
                        learning_rate=optimizer.param_groups[0]['lr'],samples_per_s=timed_samples/max(elapsed,1e-9),
                        seconds_per_update=elapsed/max(timed_steps,1),elapsed_train_s=elapsed,
                        peak_cuda_memory_GB=torch.cuda.max_memory_allocated()/1e9 if args.device=='cuda' else None,
                        **{k:float(v.detach().cpu()) for k,v in terms.items()})
                    history.write(json.dumps(event)+'\n');history.flush();print(json.dumps(event),flush=True)
                if step%args.save_every==0 or stop[0]:atomic_torch(last,snapshot(epoch,next_batch))
                if stop[0]:reason='signal';stop_run=True;break
            if stop_run:break
            synchronize(args.device)
            before=time.monotonic()
            summary,rows=evaluate(model,validation,args.batch_size,args.device,args.precision,args.workers,data_loader=validation_loader)
            # Exclude validation wall time from training throughput.
            start+=time.monotonic()-before
            score=summary['mean_ca_lddt']
            improved=score>best+args.min_delta
            if improved:best=score;best_epoch=epoch+1;bad=0
            else:bad+=1
            epoch+=1;next_batch=0
            event=dict(event='validation',epoch=epoch,step=step,train_mean_loss=train_sum/max(train_count,1),
                improved=improved,best_score=best,best_epoch=best_epoch,**summary)
            atomic_json(args.out/f'validation_{epoch:04d}.json',dict(summary=event,records=rows))
            history.write(json.dumps(event)+'\n');history.flush();print(json.dumps(event),flush=True)
            train_sum,train_count=0.,0
            if improved:atomic_torch(args.out/'best.pt',snapshot(epoch,0,True))
            atomic_torch(last,snapshot(epoch,0))
            if bad>=args.patience:completed=True;reason='validation_early_stopping';break
        if epoch>=args.epochs:completed=True;reason='epoch_limit'
    status=dict(training_complete=completed,reason=reason,epoch=epoch,next_batch=next_batch,step=step,best_epoch=best_epoch,
        best_validation_ca_lddt=best,train_fingerprint=train.fingerprint,validation_fingerprint=validation.fingerprint)
    atomic_json(args.out/'status.json',status)
    if completed:atomic_json(args.out/'TRAINING_COMPLETE.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
