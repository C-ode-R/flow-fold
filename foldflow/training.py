import numpy as np
from scipy.spatial.transform import Rotation
import torch
from .geometry import so3_exp,local_vector,global_vector,pair_geometry,backbone_with_oxygen


def flow_sample(batch,rng):
    """CPU target construction: Kabsch proper rotation, Haar rotations and SO3 log."""
    x1=batch['x'].detach().cpu().numpy().copy();r1=batch['r'].detach().cpu().numpy().copy()
    masks=batch['mask'].cpu().numpy();valid=batch['frame_mask'].cpu().numpy()
    b,l,_=x1.shape
    x0=rng.normal(size=x1.shape).astype(np.float32)
    r0=Rotation.random(b*l,random_state=rng).as_matrix().reshape(b,l,3,3).astype(np.float32)
    atoms=batch['atoms'].cpu().numpy().copy()
    for i in range(b):
        m=masks[i];v=valid[i]
        x0[i]-=x0[i,m].mean(0);x0[i,~m]=0
        u,_,vh=np.linalg.svd(x1[i,v].T@x0[i,v])
        fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vh)
        row_rotation=(u@fix@vh).astype(np.float32)
        x1[i]=x1[i]@row_rotation
        r1[i]=row_rotation.T@r1[i]
        atoms[i]=atoms[i]@row_rotation
    omega=Rotation.from_matrix((np.swapaxes(r0,-1,-2)@r1).reshape(-1,3,3)).as_rotvec().reshape(b,l,3).astype(np.float32)
    device=batch['x'].device
    x0,x1,r0,omega,atoms=[torch.from_numpy(a).to(device) for a in (x0,x1,r0,omega,atoms)]
    t=torch.from_numpy(rng.uniform(.02,.98,b).astype(np.float32)).to(device)
    weight=t[:,None,None]
    xt=((1-weight)*x0+weight*x1)*batch['mask'][...,None]
    rt=r0@so3_exp(weight*omega)
    return dict(x=xt,r=rt,t=t,velocity=local_vector(rt,x1-x0),omega=omega,target_x=x1,
                target_r=torch.from_numpy(r1).to(device),target_atoms=atoms)


def mean_masked(value,mask):
    # Per-protein normalization, then batch mean: lengths do not dominate the loss.
    weighted=value*mask
    return (weighted.flatten(1).sum(-1)/mask.expand_as(value).flatten(1).sum(-1).clamp_min(1)).mean()


def loss(model,batch,state):
    v,w=model(batch['embedding'],batch['aa'],state['x'],state['r'],state['t'],batch['mask'])
    frame=batch['frame_mask']
    velocity=mean_masked((v-state['velocity']).square(),frame[...,None])
    angular=mean_masked((w-state['omega']).square(),frame[...,None])
    remaining=(1-state['t'])[:,None,None]
    estimated_x=state['x']+remaining*global_vector(state['r'],v)
    estimated_r=state['r']@so3_exp(remaining*w)
    pred_relative=pair_geometry(estimated_x,estimated_r)[...,:3]
    true_relative=pair_geometry(state['target_x'],state['target_r'])[...,:3]
    pair=frame[:,:,None]&frame[:,None,:]
    l=frame.shape[1];pair=pair&~torch.eye(l,device=frame.device,dtype=torch.bool)[None]
    fape=mean_masked((pred_relative-true_relative).norm(dim=-1).clamp(max=1.),pair)
    # Distances avoid global orientation ambiguity; all observed pairs, including across gaps.
    pd=(estimated_x[:,:,None]-estimated_x[:,None,:]).norm(dim=-1)
    td=(state['target_x'][:,:,None]-state['target_x'][:,None,:]).norm(dim=-1)
    distance=mean_masked((pd-td).square(),pair)
    predicted_atoms=backbone_with_oxygen(estimated_x,estimated_r)
    atom_mask=batch['observed']&frame[...,None]
    atom_mask=atom_mask.clone();atom_mask[:,:,3]&=batch['links']
    atom=mean_masked((predicted_atoms-state['target_atoms']).square(),atom_mask[...,None])
    cn=(predicted_atoms[:,:-1,2]-predicted_atoms[:,1:,0]).norm(dim=-1)
    peptide=mean_masked((cn-.133).square(),batch['links'][:,:-1])
    total=velocity+angular+.5*fape+.1*distance+.5*atom+.2*peptide
    return total,dict(flow_translation=velocity,flow_rotation=angular,fape=fape,
                     pair_distance=distance,atom=atom,peptide=peptide)


@torch.no_grad()
def generate(model,batch,steps,seed):
    rng=np.random.default_rng(seed);device=batch['x'].device
    shape=batch['x'].shape;b,l,_=shape
    x=torch.tensor(rng.normal(size=shape),dtype=torch.float32,device=device)
    mask=batch['mask'];x=(x-(x*mask[...,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None])*mask[...,None]
    r=torch.tensor(Rotation.random(b*l,random_state=rng).as_matrix().reshape(b,l,3,3),dtype=torch.float32,device=device)
    encoded=model.encode(batch['embedding'],batch['aa'],mask)
    for step in range(steps):
        t=torch.full((b,),step/steps,device=device)
        v,w=model(batch['embedding'],batch['aa'],x,r,t,mask,encoded=encoded)
        x=(x+global_vector(r,v)/steps)*mask[...,None]
        x=(x-(x*mask[...,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None])*mask[...,None]
        r=r@so3_exp(w/steps)
    return x,r


def structure_metrics(pred,true,valid):
    a=pred.detach().cpu().numpy()[valid.cpu().numpy()]*10
    b=true.detach().cpu().numpy()[valid.cpu().numpy()]*10
    a-=a.mean(0);b-=b.mean(0)
    u,_,vh=np.linalg.svd(a.T@b);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vh)
    aligned=a@(u@fix@vh)
    rmsd=float(np.sqrt(((aligned-b)**2).sum(-1).mean()))
    da=np.linalg.norm(a[:,None]-a[None,:],axis=-1)
    db=np.linalg.norm(b[:,None]-b[None,:],axis=-1)
    m=(db<15)&(db>1e-6)
    score=float(np.mean(np.stack([np.abs(da-db)[m]<t for t in (.5,1.,2.,4.)]))) if m.any() else 0.
    return dict(ca_rmsd_A=rmsd,ca_lddt=score)
