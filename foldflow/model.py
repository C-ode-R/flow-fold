import math
import torch
from torch import nn
from .geometry import pair_geometry,global_vector,local_vector


class PairBlock(nn.Module):
    def __init__(self,h,p,heads):
        super().__init__();self.heads=heads
        self.norm=nn.LayerNorm(h);self.qkv=nn.Linear(h,h*3)
        self.bias=nn.Linear(p,heads);self.out=nn.Linear(h,h)
        self.ff=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h*2),nn.SiLU(),nn.Linear(h*2,h))
        self.update=nn.Sequential(nn.LayerNorm(p+2*h),nn.Linear(p+2*h,p),nn.SiLU(),nn.Linear(p,p))
    def forward(self,s,z,mask):
        b,l,h=s.shape
        q,k,v=[t.reshape(b,l,self.heads,h//self.heads).transpose(1,2) for t in self.qkv(self.norm(s)).chunk(3,-1)]
        logits=q@k.transpose(-1,-2)/math.sqrt(h//self.heads)+self.bias(z).permute(0,3,1,2)
        attn=logits.masked_fill(~mask[:,None,None,:],-1e4).softmax(-1)
        s=(s+self.out((attn@v).transpose(1,2).reshape(b,l,h)))*mask[...,None]
        s=(s+self.ff(s))*mask[...,None]
        left=s[:,:,None].expand(-1,-1,l,-1);right=s[:,None].expand(-1,l,-1,-1)
        z=z+self.update(torch.cat((z,left+right,left-right),-1))
        return s,z*(mask[:,:,None]&mask[:,None,:])[...,None]


class GeometryBlock(nn.Module):
    def __init__(self,h,p,heads=4,points=2):
        super().__init__();self.heads=heads;self.points=points;self.h=h
        self.norm=nn.LayerNorm(h);self.qkv=nn.Linear(h,h*3)
        self.qpoints=nn.Linear(h,heads*points*3);self.kpoints=nn.Linear(h,heads*points*3)
        self.vpoints=nn.Linear(h,heads*points*3)
        self.pair_bias=nn.Linear(p,heads);self.scale=nn.Parameter(torch.zeros(heads))
        self.out=nn.Linear(h+heads*points*4,h)
        self.ff=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,2*h),nn.SiLU(),nn.Linear(2*h,h))
    def forward(self,s,z,x,r,mask):
        b,l,h=s.shape;heads,points=self.heads,self.points
        t=self.norm(s)
        q,k,v=[a.reshape(b,l,heads,h//heads).permute(0,2,1,3) for a in self.qkv(t).chunk(3,-1)]
        def project(layer):
            local=layer(t).reshape(b,l,heads,points,3)
            return torch.einsum('blij,blhpj->blhpi',r,local)+x[:,:,None,None,:]
        qp,kp,vp=project(self.qpoints),project(self.kpoints),project(self.vpoints)
        squared=(qp[:,:,None]-kp[:,None,:]).square().sum((-1,-2)).permute(0,3,1,2)
        logits=q@k.transpose(-1,-2)/math.sqrt(h//heads)+self.pair_bias(z).permute(0,3,1,2)
        logits=logits-torch.nn.functional.softplus(self.scale)[None,:,None,None]*squared
        attn=logits.masked_fill(~mask[:,None,None,:],-1e4).softmax(-1)
        scalar=(attn@v).permute(0,2,1,3).reshape(b,l,h)
        point=torch.einsum('bhij,bjhpc->bihpc',attn,vp)-x[:,:,None,None,:]
        point=torch.einsum('blij,blhpi->blhpj',r,point)
        update=torch.cat((scalar,point.flatten(-3),point.norm(dim=-1).flatten(-2)),-1)
        s=(s+self.out(update))*mask[...,None]
        return (s+self.ff(s))*mask[...,None]


class FrameFlow(nn.Module):
    def __init__(self,esm_dim=320,hidden=96,pair_dim=32,heads=4,pair_layers=2,geometry_layers=2):
        super().__init__()
        self.config=dict(esm_dim=esm_dim,hidden=hidden,pair_dim=pair_dim,heads=heads,pair_layers=pair_layers,geometry_layers=geometry_layers)
        self.esm=nn.Linear(esm_dim,hidden);self.aa=nn.Embedding(20,hidden)
        self.position=nn.Linear(16,hidden)
        self.relpos=nn.Embedding(65,pair_dim)
        self.pair_init=nn.Linear(hidden*2,pair_dim)
        self.pair_blocks=nn.ModuleList(PairBlock(hidden,pair_dim,heads) for _ in range(pair_layers))
        self.time=nn.Sequential(nn.Linear(9,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.geometry=nn.Sequential(nn.Linear(13,pair_dim),nn.SiLU(),nn.Linear(pair_dim,pair_dim))
        self.blocks=nn.ModuleList(GeometryBlock(hidden,pair_dim,heads) for _ in range(geometry_layers))
        self.velocity=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,6))
        nn.init.zeros_(self.velocity[-1].weight);nn.init.zeros_(self.velocity[-1].bias)
    def encode(self,embedding,aa,mask):
        l=aa.shape[1];idx=torch.arange(l,device=embedding.device)
        phases=idx[:,None].float()/(2.**torch.arange(8,device=embedding.device).float())[None,:]
        pos=torch.cat((phases.sin(),phases.cos()),-1)
        s=(self.esm(embedding)+self.aa(aa)+self.position(pos)[None])*mask[...,None]
        left=s[:,:,None].expand(-1,-1,l,-1);right=s[:,None].expand(-1,l,-1,-1)
        z=self.pair_init(torch.cat((left+right,left-right),-1))+self.relpos((idx[:,None]-idx[None,:]).clamp(-32,32)+32)[None]
        z=z*(mask[:,:,None]&mask[:,None,:])[...,None]
        for block in self.pair_blocks:s,z=block(s,z,mask)
        return s,z
    def forward(self,embedding,aa,x,r,t,mask,encoded=None):
        s,z=self.encode(embedding,aa,mask) if encoded is None else encoded
        phases=t[:,None]*t.new_tensor([1.,2.,4.,8.])[None]*math.pi*2
        tf=torch.cat((t[:,None],phases.sin(),phases.cos()),-1)
        s=s+self.time(tf)[:,None]
        z=z+self.geometry(pair_geometry(x,r))
        for block in self.blocks:s=block(s,z,x,r,mask)
        raw=self.velocity(s)*mask[...,None]
        v=global_vector(r,raw[...,:3])
        v=(v-(v*mask[...,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None].clamp_min(1))*mask[...,None]
        return local_vector(r,v),raw[...,3:]
