"""Frozen local torsion flow plus geometry-aware global attention correction."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torsion_flow import source_torsions,wrap
from multi_torsion_flow import geometry
from scale_train import Model,load,evaluate
from global_flow import GlobalVelocity

class HybridVelocity(nn.Module):
 def __init__(self,g):
  super().__init__();self.local=Model();self.global_correction=GlobalVelocity(g)
  for p in self.local.parameters():p.requires_grad_(False)
 def forward(self,theta,t,aa):
  with torch.no_grad(): base=self.local(theta,t,aa)
  return base+self.global_correction(theta,t,aa)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--local-checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--lr',type=float,default=1e-4)
 ap.add_argument('--ode-steps',type=int,default=40);ap.add_argument('--eval-seeds',type=int,default=1);ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=json.loads(args.manifest.read_text())['proteins']
 groups={s:[load(r,device) for r in rows if r['split']==s] for s in ('train','val','test')}
 ck=torch.load(args.local_checkpoint,map_location=device,weights_only=True);g=[x.to(device) for x in ck['geometry']]
 model=HybridVelocity(g).to(device);model.local.load_state_dict(ck['model']);opt=torch.optim.AdamW(model.global_correction.parameters(),lr=args.lr)
 rng=random.Random(args.seed);order=list(range(len(groups['train'])));history=[]
 for step in range(1,args.steps+1):
  if (step-1)%len(order)==0:rng.shuffle(order)
  p=groups['train'][order[(step-1)%len(order)]];source=source_torsions(len(p['tors']),device,args.batch)
  v=wrap(p['tors'][None]-source);t=torch.rand(args.batch,device=device)
  pred=model(source+t[:,None]*v,t,p['cond']);loss=F.mse_loss(pred,v)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.global_correction.parameters(),1.0);opt.step()
  if step==1 or step%500==0 or step==args.steps:
   history.append(dict(step=step,velocity_mse=float(loss.detach())));print(step,float(loss.detach()),flush=True)
 results={s:evaluate(model,group[:50] if s=='train' else group,g,args.ode_steps,args.eval_seeds,args.seed+1) for s,group in groups.items()}
 summary={s:dict(n=len(v),median_ca_rmsd_A=float(np.median([r['median_ca_rmsd_A'] for r in v])),median_torsion_mae_rad=float(np.median([r['median_torsion_mae_rad'] for r in v]))) for s,v in results.items()}
 record=dict(architecture='hybrid_global_correction',manifest=str(args.manifest),local_checkpoint=str(args.local_checkpoint),steps=args.steps,batch=args.batch,lr=args.lr,ode_steps=args.ode_steps,eval_seeds=args.eval_seeds,summary=summary,results=results,history=history)
 (args.out/'metrics.json').write_text(json.dumps(record,indent=2))
 torch.save(dict(architecture='hybrid_global_correction',model=model.state_dict(),geometry=[x.cpu() for x in g],manifest=str(args.manifest)),args.out/'checkpoint.pt')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
