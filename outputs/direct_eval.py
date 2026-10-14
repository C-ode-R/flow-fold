"""Compare direct-frame checkpoints on identical held-out chains."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flow_fold import kabsch_rmsd
from direct_frame import DirectFrame,load_protein

def mean(rows,key):return float(np.median([r[key] for r in rows]))

def evaluate(model,group,local_table):
 output=[];model.eval()
 with torch.no_grad():
  for p in group:
   ca,rot,_=model(p['embed'],p['aa']);ca=ca*10;target=p['ca']*10
   dc=torch.cdist(ca,ca);dt=torch.cdist(target,target)
   mask=(torch.arange(len(ca),device=ca.device)[:,None]-torch.arange(len(ca),device=ca.device)[None,:]).abs()>1
   mask=mask&(dt<15)
   error=(dc-dt).abs()
   lddt=sum(float((error[mask]<threshold).float().mean()) for threshold in (.5,1,2,4))/4
   local=local_table[p['aa']]
   atoms=(ca[:,None,:]+torch.einsum('lai,lji->laj',local,rot)*10)
   true_atoms=p['atoms']*10
   cn=(atoms[1:,0]-atoms[:-1,2]).norm(dim=-1)
   target_cn=(true_atoms[1:,0]-true_atoms[:-1,2]).norm(dim=-1)
   ca_adj=(ca[1:]-ca[:-1]).norm(dim=-1)
   target_adj=(target[1:]-target[:-1]).norm(dim=-1)
   centered=ca-ca.mean(0);target_centered=target-target.mean(0)
   rmsd=float(kabsch_rmsd(centered,target_centered))
   mirror=float(kabsch_rmsd(centered,target_centered*torch.tensor([-1.,1.,1.],device=ca.device)))
   output.append(dict(id=p['id'],chain=p['chain'],residues=len(p['seq']),ca_rmsd_A=rmsd,mirror_ca_rmsd_A=mirror,
                      pair_distance_mae_A=float(error.mean()),lddt_ca=lddt,
                      peptide_CN_mae_A=float((cn-target_cn).abs().mean()),peptide_CN_mean_A=float(cn.mean()),
                      peptide_CN_within_0p2A=float(((cn-target_cn).abs()<.2).float().mean()),
                      adjacent_ca_mae_A=float((ca_adj-target_adj).abs().mean()),adjacent_ca_mean_A=float(ca_adj.mean())))
 return output

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--esm-cache',type=Path,required=True)
 ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 torch.set_num_threads(min(4,torch.get_num_threads()));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 rows=json.loads(args.manifest.read_text())['proteins'];features=torch.load(args.esm_cache,map_location='cpu',weights_only=True)['features']
 groups={s:[load_protein(r,features,device) for r in rows if r['split']==s] for s in ('val','test')}
 runs={}
 for run in args.runs:
  ck=torch.load(run/'checkpoint.pt',map_location=device,weights_only=True)
  model=DirectFrame().to(device);model.load_state_dict(ck['model']);local_table=ck['local_table'].to(device)
  results={s:evaluate(model,group,local_table) for s,group in groups.items()}
  keys=('ca_rmsd_A','pair_distance_mae_A','lddt_ca','peptide_CN_mae_A','peptide_CN_mean_A','peptide_CN_within_0p2A','adjacent_ca_mae_A','adjacent_ca_mean_A')
  summary={s:{key:mean(v,key) for key in keys} for s,v in results.items()}
  summary={s:dict(n=len(results[s]),**summary[s]) for s in summary}
  runs[run.name]=dict(summary=summary,results=results)
  print(run.name,json.dumps(summary['test']),flush=True)
 args.out.write_text(json.dumps(dict(manifest=str(args.manifest),runs=runs),indent=2))
if __name__=='__main__':main()
