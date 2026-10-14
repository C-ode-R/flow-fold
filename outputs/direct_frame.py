"""Deterministic sequence-to-residue-frame structure baseline.

Run from outputs/ with the cluster-separated manifest and frozen ESM features.
"""
import argparse,json,math,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from flow_fold import ALPHABET,kabsch_rmsd,signed_local_volume
from se3_flow import load_frames


def rotation_6d(raw):
 a=F.normalize(raw[...,:3],dim=-1)
 b0=raw[...,3:6]
 b=F.normalize(b0-(a*b0).sum(-1,keepdim=True)*a,dim=-1)
 c=torch.cross(a,b,dim=-1)
 return torch.stack((a,b,c),dim=-1)

class PairBlock(nn.Module):
 def __init__(self,h=128,heads=4):
  super().__init__();self.n1=nn.LayerNorm(h);self.n2=nn.LayerNorm(h)
  self.attn=nn.MultiheadAttention(h,heads,batch_first=True)
  self.ff=nn.Sequential(nn.Linear(h,4*h),nn.SiLU(),nn.Linear(4*h,h))
 def forward(self,x,bias):
  z=self.n1(x);x=x+self.attn(z,z,z,attn_mask=bias,need_weights=False)[0]
  return x+self.ff(self.n2(x))

class DirectFrame(nn.Module):
 def __init__(self,h=128,heads=4,layers=4):
  super().__init__();self.esm=nn.Linear(320,h);self.aa=nn.Embedding(len(ALPHABET),h)
  self.pos=nn.Linear(9,h)
  self.pair_bias=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,heads))
  self.blocks=nn.ModuleList([PairBlock(h,heads) for _ in range(layers)])
  self.ca=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,3))
  self.rot=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,6))
  self.dist=nn.Sequential(nn.Linear(2*h+1,h),nn.SiLU(),nn.Linear(h,1))
  nn.init.zeros_(self.ca[-1].weight);nn.init.zeros_(self.ca[-1].bias)
  nn.init.zeros_(self.rot[-1].weight)
  with torch.no_grad():self.rot[-1].bias.copy_(torch.tensor([1.,0.,0.,0.,1.,0.]))
 def forward(self,features,aa):
  length=len(aa);device=features.device
  index=torch.arange(length,device=device).float()
  p=index/max(length-1,1)
  pos=torch.stack([p]+[f(p*2*math.pi*k) for k in (1,2,4,8) for f in (torch.sin,torch.cos)],-1)
  x=self.esm(features)+self.aa(aa)+self.pos(pos)
  signed=(index[:,None]-index[None,:])/max(length-1,1)
  pair=torch.stack((signed.abs(),signed,torch.sin(math.pi*signed)),-1)
  bias=self.pair_bias(pair).permute(2,0,1)
  x=x[None]
  for block in self.blocks:x=block(x,bias)
  x=x[0]
  ca=self.ca(x);ca=ca-ca[:1]
  rot=rotation_6d(self.rot(x))
  xi=x[:,None,:].expand(length,length,-1);xj=x[None,:,:].expand(length,length,-1)
  distance=F.softplus(self.dist(torch.cat((xi+xj,(xi-xj).abs(),signed.abs()[...,None]),-1)))[...,0]
  return ca,rot,distance

def load_protein(row,features,device):
 seq,ca,rot,temp,mask,scale=load_frames(Path(row['path']),row['chain'])
 if seq!=row['sequence']:raise ValueError(f"Sequence mismatch {row['id']}:{row['chain']}")
 ca=ca.to(device)*scale;rot=rot.to(device);temp=temp.to(device)*scale
 anchor=rot[0]
 target_ca=(ca-ca[:1])@anchor
 target_rot=anchor.T[None]@rot
 target_atoms=target_ca[:,None,:]+torch.einsum('lai,lji->laj',temp[:,:3],target_rot)
 ids=torch.tensor([ALPHABET.index(a) for a in seq],device=device)
 embed=features[f"{row['id']}:{row['chain']}"].to(device=device,dtype=torch.float32)
 if len(embed)!=len(seq):raise ValueError('ESM length mismatch')
 return dict(id=row['id'],chain=row['chain'],split=row['split'],seq=seq,aa=ids,embed=embed,
             ca=target_ca/10,rot=target_rot,atoms=target_atoms/10,local=temp[:,:3]/10)

def shared_local(proteins,device):
 sums=torch.zeros(len(ALPHABET),3,3,device=device);counts=torch.zeros(len(ALPHABET),device=device)
 for p in proteins:
  sums.index_add_(0,p['aa'],p['local']);counts.index_add_(0,p['aa'],torch.ones(len(p['aa']),device=device))
 global_mean=sums.sum(0)/counts.sum()
 return torch.where(counts[:,None,None]>0,sums/counts.clamp_min(1)[:,None,None],global_mean[None])

def losses(model,p,local_table,bond_weight=.1,ca_step_weight=0.,rot_weight=.1,chiral_weight=0.):
 ca,rot,distance=model(p['embed'],p['aa']);target_ca=p['ca']
 target_pair=torch.cdist(target_ca,target_ca)
 pred_pair=torch.cdist(ca,ca)
 local=local_table[p['aa']]
 atoms=ca[:,None,:]+torch.einsum('lai,lji->laj',local,rot)
 pos=F.smooth_l1_loss(ca,target_ca,beta=.5)
 pair=F.smooth_l1_loss(pred_pair,target_pair,beta=.5)
 dist=F.smooth_l1_loss(distance,target_pair,beta=.5)
 atom=F.smooth_l1_loss(atoms,p['atoms'],beta=.5)
 angle=(3-torch.einsum('lij,lij->l',rot,p['rot'])).mean()/3
 cn=(atoms[1:,0]-atoms[:-1,2]).norm(dim=-1)
 bond=((cn-.133)**2).mean()
 ca_step=((ca[1:]-ca[:-1]).norm(dim=-1)-.38).square().mean()
 chiral=F.smooth_l1_loss(signed_local_volume(ca[None]),signed_local_volume(target_ca[None]))
 total=pos+pair+0.25*dist+0.5*atom+rot_weight*angle+bond_weight*bond+ca_step_weight*ca_step+chiral_weight*chiral
 return total,dict(pos=float(pos.detach()),pair=float(pair.detach()),dist=float(dist.detach()),atom=float(atom.detach()),rot=float(angle.detach()),bond=float(bond.detach()),ca_step=float(ca_step.detach()),chiral=float(chiral.detach()))

@torch.no_grad()
def evaluate(model,group,local_table):
 model.eval();out=[]
 for p in group:
  ca,rot,_=model(p['embed'],p['aa']);ca_a=ca*10;target=p['ca']*10
  rmsd=float(kabsch_rmsd(ca_a-ca_a.mean(0),target-target.mean(0)))
  dist_err=float((torch.cdist(ca_a,ca_a)-torch.cdist(target,target)).abs().mean())
  atoms=ca[:,None,:]+torch.einsum('lai,lji->laj',local_table[p['aa']],rot)
  bond=float((atoms[1:,0]-atoms[:-1,2]).norm(dim=-1).mean()*10)
  ca_step=float((ca[1:]-ca[:-1]).norm(dim=-1).mean()*10)
  out.append(dict(id=p['id'],chain=p['chain'],residues=len(p['seq']),ca_rmsd_A=rmsd,pair_distance_mae_A=dist_err,peptide_CN_mean_A=bond,adjacent_ca_mean_A=ca_step))
 model.train();return out

def summarize(rows):
 return dict(n=len(rows),median_ca_rmsd_A=float(np.median([x['ca_rmsd_A'] for x in rows])),
             median_pair_distance_mae_A=float(np.median([x['pair_distance_mae_A'] for x in rows])),
             median_peptide_CN_mean_A=float(np.median([x['peptide_CN_mean_A'] for x in rows])),
             median_adjacent_ca_mean_A=float(np.median([x['adjacent_ca_mean_A'] for x in rows])))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--esm-cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--steps',type=int,default=12000);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--init-checkpoint',type=Path);ap.add_argument('--bond-weight',type=float,default=.1)
 ap.add_argument('--ca-step-weight',type=float,default=0.);ap.add_argument('--rot-weight',type=float,default=.1)
 ap.add_argument('--chiral-weight',type=float,default=0.)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 rows=json.loads(args.manifest.read_text())['proteins'];features=torch.load(args.esm_cache,map_location='cpu',weights_only=True)['features']
 groups={s:[load_protein(r,features,device) for r in rows if r['split']==s] for s in ('train','val','test')}
 local_table=shared_local(groups['train'],device)
 model=DirectFrame().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
 if args.init_checkpoint:
  ck=torch.load(args.init_checkpoint,map_location=device,weights_only=True)
  model.load_state_dict(ck['model']);local_table=ck['local_table'].to(device)
 rng=random.Random(args.seed);order=list(range(len(groups['train'])));history=[]
 for step in range(1,args.steps+1):
  if (step-1)%len(order)==0:rng.shuffle(order)
  p=groups['train'][order[(step-1)%len(order)]]
  loss,parts=losses(model,p,local_table,args.bond_weight,args.ca_step_weight,args.rot_weight,args.chiral_weight)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if step==1 or step%500==0 or step==args.steps:
   item=dict(step=step,loss=float(loss.detach()),**parts);history.append(item);print(json.dumps(item),flush=True)
 results={s:evaluate(model,g,local_table) for s,g in groups.items()}
 summary={s:summarize(v) for s,v in results.items()}
 record=dict(model='direct_residue_frames',manifest=str(args.manifest),esm_cache=str(args.esm_cache),init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,steps=args.steps,lr=args.lr,seed=args.seed,device=str(device),bond_weight=args.bond_weight,ca_step_weight=args.ca_step_weight,rot_weight=args.rot_weight,chiral_weight=args.chiral_weight,summary=summary,results=results,history=history)
 (args.out/'metrics.json').write_text(json.dumps(record,indent=2))
 torch.save(dict(model=model.state_dict(),local_table=local_table.cpu(),manifest=str(args.manifest),esm_cache=str(args.esm_cache),seed=args.seed),args.out/'checkpoint.pt')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
