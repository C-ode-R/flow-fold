"""Train a single torsion-flow model on a cluster-separated PDB manifest."""
import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from flow_fold import ALPHABET
from se3_flow import load_frames, reconstruct
from torsion_flow import internal_coordinates, source_torsions, wrap
from multi_torsion_flow import geometry, backbone, rmsd

class Model(nn.Module):
 def __init__(self,h=96,esm=False):
  super().__init__(); self.aa=nn.Linear(320,h) if esm else nn.Embedding(len(ALPHABET),h); self.role=nn.Embedding(3,h)
  self.seq=nn.ModuleList([nn.Conv1d(h,h,5,padding=2*d,dilation=d) for d in (1,2,4,8)])
  self.inp=nn.Sequential(nn.Linear(h+6,h),nn.SiLU(),nn.Linear(h,h))
  self.blocks=nn.ModuleList([nn.Conv1d(h,h,5,padding=2*d,dilation=d) for d in (1,2,4,8,16,32)])
  self.head=nn.Sequential(nn.SiLU(),nn.Linear(h,1)); nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
 def forward(self,theta,t,aa):
  b,n=theta.shape; a=torch.arange(3,n+3,device=theta.device); r=a//3
  seq=self.aa(aa).T[None]
  for conv in self.seq: seq=seq+F.silu(conv(seq))
  base=seq.transpose(1,2)[0][r]+self.role(a%3)
  time=t[:,None].expand_as(theta); rel=r.float()[None].expand_as(theta)/max(len(aa)-1,1)
  dyn=torch.stack((theta.sin(),theta.cos(),time,(2*math.pi*time).sin(),(2*math.pi*time).cos(),rel),-1)
  h=self.inp(torch.cat((base[None].expand(b,-1,-1),dyn),-1)).transpose(1,2)
  for conv in self.blocks: h=h+F.silu(conv(h))
  return self.head(h.transpose(1,2)).squeeze(-1)

def load(row,device,features=None):
 seq,ca,rot,temp,mask,scale=load_frames(Path(row['path']),row['chain'])
 xyz=reconstruct(ca,rot,temp)[:,:3].reshape(-1,3).to(device)*scale
 lens,angs,tors=internal_coordinates(xyz)
 ids=torch.tensor([ALPHABET.index(a) for a in seq],device=device)
 cond=features[f"{row['id']}:{row['chain']}"].to(device=device,dtype=torch.float32) if features is not None else ids
 if len(cond)!=len(seq): raise ValueError('ESM feature and structure sequence lengths differ')
 return dict(name=row['id'],chain=row['chain'],seq=seq,xyz=xyz,ca=xyz[1::3],ids=ids,cond=cond,lens=lens,angs=angs,tors=tors,cluster=row['cluster'])

@torch.no_grad()
def generate(model,p,g,steps,seed):
 torch.manual_seed(seed); theta=source_torsions(len(p['tors']),p['ids'].device,1); dt=1/steps
 for i in range(steps):
  t=torch.tensor([i*dt],device=theta.device); v=model(theta,t,p['cond'])
  theta=theta+dt*(v+model(theta+dt*v,t+dt,p['cond']))/2
 return backbone(theta,g)[0],theta[0]

@torch.no_grad()
def evaluate(model,group,g,steps,seeds,seed):
 out=[]
 for p in group:
  oracle=backbone(p['tors'][None],g)[0]; errs=[]; tors=[]
  for j in range(seeds):
   atoms,theta=generate(model,p,g,steps,seed+j)
   errs.append(rmsd(atoms[:,1],p['ca'])); tors.append(float(wrap(theta-p['tors']).abs().mean()))
  out.append(dict(id=p['name'],chain=p['chain'],cluster=p['cluster'],residues=len(p['seq']),oracle_ca_rmsd_A=rmsd(oracle[:,1],p['ca']),median_ca_rmsd_A=float(np.median(errs)),median_torsion_mae_rad=float(np.median(tors))))
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--esm-cache',type=Path)
 ap.add_argument('--structure-weight',type=float,default=0.0)
 ap.add_argument('--structure-every',type=int,default=8)
 ap.add_argument('--init-checkpoint',type=Path)
 ap.add_argument('--lr',type=float,default=5e-4)
 ap.add_argument('--steps',type=int,default=8000); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--ode-steps',type=int,default=40); ap.add_argument('--eval-seeds',type=int,default=3); ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); torch.manual_seed(args.seed); torch.set_num_threads(min(4,torch.get_num_threads()))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=json.loads(args.manifest.read_text())['proteins']
 features=torch.load(args.esm_cache,map_location='cpu',weights_only=True)['features'] if args.esm_cache else None
 groups={s:[load(r,device,features) for r in rows if r['split']==s] for s in ('train','val','test')}
 g=geometry(groups['train']); model=Model(esm=features is not None).to(device)
 if args.init_checkpoint:
  ck=torch.load(args.init_checkpoint,map_location=device,weights_only=True)
  model.load_state_dict(ck['model']); g=[x.to(device) for x in ck['geometry']]
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
 rng=random.Random(args.seed); order=list(range(len(groups['train']))); history=[]
 for step in range(1,args.steps+1):
  if (step-1)%len(order)==0: rng.shuffle(order)
  p=groups['train'][order[(step-1)%len(order)]]; theta0=source_torsions(len(p['tors']),device,args.batch)
  v=wrap(p['tors'][None]-theta0); t=torch.rand(args.batch,device=device)
  theta_t=theta0+t[:,None]*v; pred=model(theta_t,t,p['cond'])
  loss=F.mse_loss(pred,v)
  if args.structure_weight and step%args.structure_every==0:
   endpoint=theta_t[:2]+(1-t[:2,None])*pred[:2]
   predicted_ca=backbone(endpoint,g)[:,:,1]
   oracle_ca=backbone(p['tors'][None],g)[:,:,1].detach()
   loss=loss+args.structure_weight*(predicted_ca-oracle_ca).norm(dim=-1).mean()/10
  opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  if step==1 or step%500==0 or step==args.steps:
   history.append(dict(step=step,velocity_mse=float(loss.detach()))); print(step,float(loss.detach()),flush=True)
 evaluated={s:(group[:50] if s=='train' and len(group)>50 else group) for s,group in groups.items()}
 results={s:evaluate(model,group,g,args.ode_steps,args.eval_seeds,args.seed+1) for s,group in evaluated.items()}
 summary={s:dict(n=len(v),median_ca_rmsd_A=float(np.median([r['median_ca_rmsd_A'] for r in v])),median_oracle_ca_rmsd_A=float(np.median([r['oracle_ca_rmsd_A'] for r in v])),median_torsion_mae_rad=float(np.median([r['median_torsion_mae_rad'] for r in v]))) for s,v in results.items()}
 (args.out/'metrics.json').write_text(json.dumps(dict(manifest=str(args.manifest),esm_cache=str(args.esm_cache) if args.esm_cache else None,init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,lr=args.lr,structure_weight=args.structure_weight,structure_every=args.structure_every,steps=args.steps,batch=args.batch,ode_steps=args.ode_steps,eval_seeds=args.eval_seeds,device=str(device),dataset_counts={s:len(v) for s,v in groups.items()},summary=summary,results=results,history=history),indent=2))
 torch.save(dict(model=model.state_dict(),geometry=[x.cpu() for x in g],manifest=str(args.manifest),esm_cache=str(args.esm_cache) if args.esm_cache else None,seed=args.seed),args.out/'checkpoint.pt'); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
