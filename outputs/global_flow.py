"""Geometry-aware global residue attention for torsion flow matching."""
import argparse,json,math,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from flow_fold import ALPHABET
from torsion_flow import source_torsions,wrap
from multi_torsion_flow import geometry,backbone
from scale_train import load,evaluate

class AttentionBlock(nn.Module):
 def __init__(self,h=128,heads=4):
  super().__init__();self.norm1=nn.LayerNorm(h);self.norm2=nn.LayerNorm(h)
  self.attn=nn.MultiheadAttention(h,heads,batch_first=True)
  self.ff=nn.Sequential(nn.Linear(h,4*h),nn.SiLU(),nn.Linear(4*h,h))
 def forward(self,x,bias):
  y=self.norm1(x);y=self.attn(y,y,y,attn_mask=bias,need_weights=False)[0]
  x=x+y;return x+self.ff(self.norm2(x))

class GlobalVelocity(nn.Module):
 def __init__(self,g,esm=False,h=128,heads=4,layers=3):
  super().__init__();self.esm=esm;self.heads=heads
  self.aa=nn.Linear(320,h) if esm else nn.Embedding(len(ALPHABET),h)
  self.dynamic=nn.Sequential(nn.Linear(10,h),nn.SiLU(),nn.Linear(h,h))
  self.pair=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,heads))
  self.blocks=nn.ModuleList([AttentionBlock(h,heads) for _ in range(layers)])
  self.output=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,3))
  nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
  self.register_buffer('role_lengths',g[0].detach().clone());self.register_buffer('role_angles',g[1].detach().clone())
  self.register_buffer('nca',g[2].detach().clone());self.register_buffer('cac',g[3].detach().clone());self.register_buffer('first_angle',g[4].detach().clone())
 def forward(self,theta,t,aa):
  b,n=theta.shape;length=(n+3)//3
  expanded=theta.new_zeros(b,length,3);expanded[:,1:]=theta.reshape(b,length-1,3)
  time=t[:,None,None].expand(b,length,1)
  pos=torch.arange(length,device=theta.device).float()[None,:,None].expand(b,-1,-1)/max(length-1,1)
  dyn=torch.cat((expanded.sin(),expanded.cos(),time,(2*math.pi*time).sin(),(2*math.pi*time).cos(),pos),-1)
  x=self.aa(aa)[None]+self.dynamic(dyn)
  with torch.no_grad():
   g=(self.role_lengths,self.role_angles,self.nca,self.cac,self.first_angle)
   ca=backbone(theta,g)[:,:,1]
   dist=torch.cdist(ca,ca)/20
   index=torch.arange(length,device=theta.device)
   signed=(index[:,None]-index[None,:]).float()/max(length-1,1)
   pair=torch.stack((dist,signed.abs()[None].expand(b,-1,-1),signed[None].expand(b,-1,-1)),-1)
  bias=self.pair(pair).permute(0,3,1,2).reshape(b*self.heads,length,length)
  for block in self.blocks:x=block(x,bias)
  return self.output(x)[:,1:].reshape(b,n)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--esm-cache',type=Path);ap.add_argument('--steps',type=int,default=8000);ap.add_argument('--batch',type=int,default=8)
 ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--ode-steps',type=int,default=40);ap.add_argument('--eval-seeds',type=int,default=1)
 ap.add_argument('--seed',type=int,default=42);ap.add_argument('--resume',type=Path)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=json.loads(args.manifest.read_text())['proteins']
 features=torch.load(args.esm_cache,map_location='cpu',weights_only=True)['features'] if args.esm_cache else None
 groups={s:[load(r,device,features) for r in rows if r['split']==s] for s in ('train','val','test')}
 g=geometry(groups['train']);model=GlobalVelocity(g,esm=features is not None).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
 if args.resume:
  ck=torch.load(args.resume,map_location=device,weights_only=True);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
 rng=random.Random(args.seed);order=list(range(len(groups['train'])));history=[]
 for step in range(1,args.steps+1):
  if (step-1)%len(order)==0:rng.shuffle(order)
  p=groups['train'][order[(step-1)%len(order)]];source=source_torsions(len(p['tors']),device,args.batch)
  v=wrap(p['tors'][None]-source);t=torch.rand(args.batch,device=device)
  pred=model(source+t[:,None]*v,t,p['cond']);loss=F.mse_loss(pred,v)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if step==1 or step%500==0 or step==args.steps:
   history.append(dict(step=step,velocity_mse=float(loss.detach())));print(step,float(loss.detach()),flush=True)
 results={s:evaluate(model,group[:50] if s=='train' else group,g,args.ode_steps,args.eval_seeds,args.seed+1) for s,group in groups.items()}
 summary={s:dict(n=len(v),median_ca_rmsd_A=float(np.median([r['median_ca_rmsd_A'] for r in v])),median_torsion_mae_rad=float(np.median([r['median_torsion_mae_rad'] for r in v]))) for s,v in results.items()}
 record=dict(architecture='global_distance_attention',manifest=str(args.manifest),esm_cache=str(args.esm_cache) if args.esm_cache else None,steps=args.steps,batch=args.batch,lr=args.lr,ode_steps=args.ode_steps,eval_seeds=args.eval_seeds,dataset_counts={s:len(v) for s,v in groups.items()},summary=summary,results=results,history=history)
 (args.out/'metrics.json').write_text(json.dumps(record,indent=2))
 torch.save(dict(architecture='global_distance_attention',model=model.state_dict(),optimizer=opt.state_dict(),geometry=[x.cpu() for x in g],manifest=str(args.manifest),esm_cache=str(args.esm_cache) if args.esm_cache else None),args.out/'checkpoint.pt')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
