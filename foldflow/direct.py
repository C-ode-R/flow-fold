"""Deterministic sequence-only control. Same trunk/point attention, no flow path."""
import torch
from torch import nn
from .model import FrameFlow
from .geometry import so3_exp,pair_geometry,global_vector,backbone_with_oxygen
from .training import mean_masked


def rotation_6d(value):
    first,second=value[...,:3],value[...,3:]
    e1=first/first.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    second=second-(second*e1).sum(-1,keepdim=True)*e1
    e2=second/second.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    return torch.stack((e1,e2,torch.cross(e1,e2,dim=-1)),-1)


def center(x,mask):
    return (x-(x*mask[...,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None].clamp_min(1))*mask[...,None]


class DirectFrames(FrameFlow):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        h=self.config['hidden']
        del self.time
        del self.velocity
        self.initial=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,9))
        nn.init.normal_(self.initial[-1].weight,std=.02)
        with torch.no_grad():
            self.initial[-1].bias.zero_()
            self.initial[-1].bias[3:]=torch.tensor([1.,0.,0.,0.,1.,0.])
        self.updates=nn.ModuleList(nn.Linear(h,6) for _ in self.blocks)
        for layer in self.updates:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,embedding,aa,mask):
        single,pair=self.encode(embedding,aa,mask)
        initial=self.initial(single)
        x=center(initial[...,:3],mask)
        r=rotation_6d(initial[...,3:])
        for block,update in zip(self.blocks,self.updates):
            geometry_pair=pair+self.geometry(pair_geometry(x,r))
            single=block(single,geometry_pair,x,r,mask)
            raw=update(single)*mask[...,None]
            x=center(x+global_vector(r,.2*torch.tanh(raw[...,:3])),mask)
            r=r@so3_exp(.5*torch.tanh(raw[...,3:]))
        return x,r


def direct_loss(x,r,batch):
    frame=batch['frame_mask']
    position=mean_masked((x-batch['x']).square(),frame[...,None])
    rotation=mean_masked((r-batch['r']).square(),frame[...,None,None])
    pair=frame[:,:,None]&frame[:,None,:]
    pair=pair&~torch.eye(frame.shape[1],device=x.device,dtype=torch.bool)[None]
    pred_relative=pair_geometry(x,r)[...,:3]
    target_relative=pair_geometry(batch['x'],batch['r'])[...,:3]
    fape=mean_masked((pred_relative-target_relative).norm(dim=-1).clamp(max=1),pair)
    pd=(x[:,:,None]-x[:,None,:]).norm(dim=-1)
    td=(batch['x'][:,:,None]-batch['x'][:,None,:]).norm(dim=-1)
    distance=mean_masked((pd-td).square(),pair)
    atoms=backbone_with_oxygen(x,r)
    amask=batch['observed']&frame[...,None]
    amask=amask.clone();amask[:,:,3]&=batch['links']
    atom=mean_masked((atoms-batch['atoms']).square(),amask[...,None])
    cn=(atoms[:,:-1,2]-atoms[:,1:,0]).norm(dim=-1)
    peptide=mean_masked((cn-.133).square(),batch['links'][:,:-1])
    total=position+.5*rotation+.2*fape+.1*distance+.2*atom+.1*peptide
    return total,dict(position=position,rotation_chordal=rotation,fape=fape,pair_distance=distance,atom=atom,peptide=peptide)
