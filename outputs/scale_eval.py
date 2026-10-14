"""Re-evaluate a scale_train.py checkpoint over multiple starting torsions."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scale_train import Model,load,evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--ode-steps',type=int,default=40)
 args=ap.parse_args();torch.set_num_threads(min(4,torch.get_num_threads()))
 record=json.loads((args.run/'metrics.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ck=torch.load(args.run/'checkpoint.pt',map_location=device,weights_only=True)
 features=torch.load(record['esm_cache'],map_location='cpu',weights_only=True)['features'] if record.get('esm_cache') else None
 g=[x.to(device) for x in ck['geometry']]
 rows=json.loads(Path(record['manifest']).read_text())['proteins']
 if ck.get('architecture')=='global_distance_attention':
  from global_flow import GlobalVelocity
  model=GlobalVelocity(g,esm=features is not None).to(device)
 elif ck.get('architecture')=='hybrid_global_correction':
  from hybrid_flow import HybridVelocity
  model=HybridVelocity(g).to(device)
 else:
  model=Model(esm=features is not None).to(device)
 model.load_state_dict(ck['model']);model.eval()
 results={}
 for split in ('val','test'):
  group=[load(r,device,features) for r in rows if r['split']==split]
  results[split]=evaluate(model,group,g,args.ode_steps,args.seeds,43)
  print(split,len(group),float(np.median([r['median_ca_rmsd_A'] for r in results[split]])),flush=True)
 summary={s:dict(n=len(v),median_ca_rmsd_A=float(np.median([r['median_ca_rmsd_A'] for r in v])),median_torsion_mae_rad=float(np.median([r['median_torsion_mae_rad'] for r in v]))) for s,v in results.items()}
 (args.run/f'eval_{args.seeds}seeds.json').write_text(json.dumps(dict(seeds=args.seeds,ode_steps=args.ode_steps,summary=summary,results=results),indent=2))
if __name__=='__main__':main()
