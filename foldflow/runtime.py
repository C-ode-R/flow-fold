"""Single-device training/evaluation utilities. Geometry stays float32."""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LengthBatches,collate_shards
from .folding import invariant_loss
from .training import structure_metrics


def validate_device(device,precision):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: check driver and CUDA-enabled PyTorch install')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    if precision=='bf16' and (device!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bf16 trunk precision requires a BF16-capable CUDA device')


def synchronize(device):
    if device=='cuda':torch.cuda.synchronize()
    elif device=='mps':torch.mps.synchronize()


def loader(dataset,batch_size,seed,epoch,workers,device,shuffle):
    sampler=LengthBatches(dataset.lengths,batch_size,seed,epoch,shuffle)
    kwargs=dict(dataset=dataset,batch_sampler=sampler,collate_fn=collate_shards,
        num_workers=workers,pin_memory=device=='cuda',generator=torch.Generator().manual_seed(seed+epoch))
    if workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
    return DataLoader(**kwargs)


def move(batch,device):
    return {k:v.to(device,non_blocking=device=='cuda') for k,v in batch.items()}


def predict(model,batch,precision):
    device=batch['x'].device.type
    context=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    with context:encoded=model.encode(batch['embedding'],batch['aa'],batch['mask'])
    encoded=tuple(t.float() for t in encoded)
    context=torch.autocast('cuda',enabled=False) if device=='cuda' else nullcontext()
    with context:return model(batch['embedding'],batch['aa'],batch['mask'],encoded=encoded)


def code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    names=['foldflow/model.py','foldflow/direct.py','foldflow/folding.py','foldflow/geometry.py',
        'foldflow/data.py','foldflow/dataset.py','foldflow/runtime.py','scripts/train_dataset.py']
    digest=hashlib.sha256()
    for name in names:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate(model,dataset,batch_size,device,precision,workers=0,data_loader=None):
    model.eval();rows=[];loss_sum=0.
    synchronize(device);start=time.monotonic()
    source=data_loader if data_loader is not None else loader(dataset,batch_size,0,0,workers,device,False)
    for cpu,metas in source:
        batch=move(cpu,device);x,r=predict(model,batch,precision)
        value,_=invariant_loss(x,r,batch)
        if not torch.isfinite(value) or not torch.isfinite(x).all() or not torch.isfinite(r).all():
            raise FloatingPointError('nonfinite evaluation output')
        loss_sum+=float(value.cpu())*len(metas)
        px=x.cpu();pr=r.cpu();target=cpu['x'];rot=cpu['r'];mask=cpu['frame_mask']
        for i,meta in enumerate(metas):
            valid=mask[i]
            metrics=structure_metrics(px[i],target[i],valid)
            a=px[i][valid].numpy().copy();b=target[i][valid].numpy().copy()
            a-=a.mean(0);b-=b.mean(0)
            u,_,vh=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vh)
            q=torch.tensor((u@fix@vh).T,dtype=pr.dtype)
            relative=(q@pr[i]).transpose(-1,-2)@rot[i]
            vee=torch.stack((relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]),-1)*.5
            cos=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
            angles=(torch.atan2(vee.norm(dim=-1),cos)[valid]*180/torch.pi)
            rows.append(dict(id=meta['id'],cluster_id=meta['cluster_id'],**metrics,
                mean_aligned_frame_rotation_error_deg=float(angles.mean()),max_aligned_frame_rotation_error_deg=float(angles.max())))
    synchronize(device)
    count=len(rows)
    if count!=len(dataset):raise ValueError('evaluation did not cover complete split')
    cluster={}
    for row in rows:cluster.setdefault(row['cluster_id'],[]).append(row['ca_lddt'])
    summary=dict(split=dataset.split,count=count,mean_loss=loss_sum/count,
        mean_ca_lddt=float(np.mean([r['ca_lddt'] for r in rows])),
        median_ca_lddt=float(np.median([r['ca_lddt'] for r in rows])),
        mean_ca_rmsd_A=float(np.mean([r['ca_rmsd_A'] for r in rows])),
        median_ca_rmsd_A=float(np.median([r['ca_rmsd_A'] for r in rows])),
        ca_rmsd_lt2_fraction=sum(r['ca_rmsd_A']<2 for r in rows)/count,
        cluster_macro_ca_lddt=float(np.mean([np.mean(v) for v in cluster.values()])),
        max_aligned_frame_rotation_error_deg=max(r['max_aligned_frame_rotation_error_deg'] for r in rows),
        elapsed_s=time.monotonic()-start)
    model.train()
    return summary,rows
