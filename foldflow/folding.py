"""Small ESMFold-inspired direct predictor; not an ESMFold reproduction."""
import torch
from torch import nn
from .direct import DirectFrames,center,rotation_6d
from .geometry import pair_geometry,global_vector,so3_exp,backbone_with_oxygen
from .training import mean_masked


class TriangleMultiplication(nn.Module):
    def __init__(self,pair_dim,channels=16):
        super().__init__()
        self.norm=nn.LayerNorm(pair_dim)
        self.a=nn.Linear(pair_dim,channels);self.b=nn.Linear(pair_dim,channels)
        self.out=nn.Linear(2*channels,pair_dim)
        self.gate=nn.Linear(pair_dim,pair_dim)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,z,mask):
        valid=(mask[:,:,None]&mask[:,None,:])[...,None]
        n=self.norm(z);a=self.a(n)*valid;b=self.b(n)*valid
        outgoing=torch.einsum('bikc,bjkc->bijc',a,b)
        incoming=torch.einsum('bkic,bkjc->bijc',a,b)
        scale=mask.sum(1).float().sqrt()[:,None,None,None].clamp_min(1)
        delta=self.out(torch.cat((outgoing,incoming),-1)/scale)
        return (z+torch.sigmoid(self.gate(n))*delta)*valid


class FoldingModel(DirectFrames):
    def __init__(self,recycles=3,**kwargs):
        super().__init__(**kwargs)
        self.config['recycles']=recycles
        h,p=self.config['hidden'],self.config['pair_dim']
        self.triangles=nn.ModuleList(TriangleMultiplication(p) for _ in range(self.config['pair_layers']))
        self.recycle_single=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h))
        self.recycle_pair=nn.Sequential(nn.LayerNorm(p),nn.Linear(p,p))
    def encode(self,embedding,aa,mask):
        s,z=super().encode(embedding,aa,mask)
        for triangle in self.triangles:z=triangle(z,mask)
        return s,z
    def forward(self,embedding,aa,mask,encoded=None):
        base_s,base_z=self.encode(embedding,aa,mask) if encoded is None else encoded
        initial=self.initial(base_s)
        x=center(initial[...,:3],mask);r=rotation_6d(initial[...,3:])
        s,z=base_s,base_z
        for recycle in range(self.config['recycles']):
            if recycle:
                s=base_s+self.recycle_single(s)
                z=base_z+self.recycle_pair(z)
            for block,update in zip(self.blocks,self.updates):
                s=block(s,z+self.geometry(pair_geometry(x,r)),x,r,mask)
                raw=update(s)*mask[...,None]
                x=center(x+global_vector(r,.2*torch.tanh(raw[...,:3])),mask)
                r=r@so3_exp(.5*torch.tanh(raw[...,3:]))
        return x,r


def smooth_distance(delta):
    # 0.01 A smoothing removes unstable norm subgradients at exact self-atom matches.
    return torch.sqrt(delta.square().sum(-1)+1e-6)-1e-3


def invariant_loss(x,r,batch):
    """Independent proper global transforms of prediction/target preserve this loss."""
    frame=batch['frame_mask'];tx,tr=batch['x'],batch['r']
    pair=frame[:,:,None]&frame[:,None,:]
    pair=pair&~torch.eye(frame.shape[1],device=x.device,dtype=torch.bool)[None]
    pg=pair_geometry(x,r);tg=pair_geometry(tx,tr)
    fape=mean_masked(smooth_distance(pg[...,:3]-tg[...,:3]).clamp(max=1),pair)
    rotation=mean_masked((pg[...,3:12]-tg[...,3:12]).square(),pair[...,None])
    distance=mean_masked((pg[...,-1]-tg[...,-1]).square(),pair)
    atoms=backbone_with_oxygen(x,r)
    atom_mask=batch['observed']&frame[...,None]
    atom_mask=atom_mask.clone();atom_mask[:,:,3]&=batch['links']
    # Every valid residue frame observes each supervised atom, not raw world coordinates.
    def local_atoms(points,positions,rotations):
        delta=points.flatten(1,2)[:,None]-positions[:,:,None]
        return torch.einsum('bijd,bidk->bijk',delta,rotations)
    atom_error=smooth_distance(local_atoms(atoms,x,r)-local_atoms(batch['atoms'],tx,tr)).clamp(max=1)
    full_mask=frame[:,:,None]&atom_mask.flatten(1,2)[:,None,:]
    atom_fape=mean_masked(atom_error,full_mask)
    cn=(atoms[:,:-1,2]-atoms[:,1:,0]).norm(dim=-1)
    peptide=mean_masked((cn-.133).square(),batch['links'][:,:-1])
    total=.5*fape+.2*rotation+.2*distance+.5*atom_fape+.1*peptide
    return total,dict(fape=fape,relative_rotation=rotation,pair_distance=distance,atom_fape=atom_fape,peptide=peptide)


def augment_target(batch,seed):
    """Transform all target atoms/frames together. Never an input to the model."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(seed);device=batch['x'].device;b=len(batch['x'])
    q=torch.tensor(Rotation.random(b,random_state=rng).as_matrix(),dtype=torch.float32,device=device)
    shift=torch.tensor(rng.normal(size=(b,3)),dtype=torch.float32,device=device)
    out=dict(batch)
    out['x']=torch.einsum('bij,blj->bli',q,batch['x'])+shift[:,None]
    out['r']=q[:,None]@batch['r']
    out['atoms']=torch.einsum('bij,blaj->blai',q,batch['atoms'])+shift[:,None,None]
    return out
