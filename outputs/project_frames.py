"""Project predicted residue frames toward connected backbone geometry."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from flow_fold import kabsch_rmsd
from direct_frame import DirectFrame,load_protein,rotation_6d
from se3_flow import write_pdb


def atoms_from(ca,rot,local):
 return ca[:,None,:]+torch.einsum('lai,lji->laj',local,rot)

def angle(a,b,c):
 u=F.normalize(a-b,dim=-1);v=F.normalize(c-b,dim=-1)
 return torch.acos((u*v).sum(-1).clamp(-1+1e-6,1-1e-6))

def dihedral(a,b,c,d):
 axis=F.normalize(c-b,dim=-1)
 left=a-b;right=d-c
 left=F.normalize(left-(left*axis).sum(-1,keepdim=True)*axis,dim=-1)
 right=F.normalize(right-(right*axis).sum(-1,keepdim=True)*axis,dim=-1)
 return torch.atan2((torch.cross(axis,left,dim=-1)*right).sum(-1),(left*right).sum(-1))

def measures(ca,rot,p,local):
 with torch.no_grad():
  atoms=atoms_from(ca,rot,local)*10;target_atoms=p['atoms']*10
  cn=(atoms[1:,0]-atoms[:-1,2]).norm(dim=-1);true_cn=(target_atoms[1:,0]-target_atoms[:-1,2]).norm(dim=-1)
  x=ca*10;y=p['ca']*10
  dx=torch.cdist(x,x);dy=torch.cdist(y,y)
  index=torch.arange(len(x),device=x.device)
  separation=(index[:,None]-index[None,:]).abs()
  contact_mask=(separation>1)&(dy<15)
  lddt=sum(float(((dx-dy).abs()[contact_mask]<threshold).float().mean()) for threshold in (.5,1,2,4))/4
  clash_mask=torch.triu((separation>3)&(dx<3),diagonal=1)
  adjacent=(x[1:]-x[:-1]).norm(dim=-1);target_adj=(y[1:]-y[:-1]).norm(dim=-1)
  angle1=angle(atoms[:-1,1],atoms[:-1,2],atoms[1:,0])
  true_angle1=angle(target_atoms[:-1,1],target_atoms[:-1,2],target_atoms[1:,0])
  angle2=angle(atoms[:-1,2],atoms[1:,0],atoms[1:,1])
  true_angle2=angle(target_atoms[:-1,2],target_atoms[1:,0],target_atoms[1:,1])
  omega=dihedral(atoms[:-1,1],atoms[:-1,2],atoms[1:,0],atoms[1:,1])
  true_omega=dihedral(target_atoms[:-1,1],target_atoms[:-1,2],target_atoms[1:,0],target_atoms[1:,1])
  omega_error=torch.atan2((omega-true_omega).sin(),(omega-true_omega).cos()).abs()
  return dict(ca_rmsd_A=float(kabsch_rmsd(x-x.mean(0),y-y.mean(0))),pair_mae_A=float((dx-dy).abs().mean()),
              mirror_ca_rmsd_A=float(kabsch_rmsd(x-x.mean(0),(y-y.mean(0))*torch.tensor([-1.,1.,1.],device=x.device))),
              lddt_ca=lddt,nonlocal_ca_clashes=int(clash_mask.sum()),
              cn_mae_A=float((cn-true_cn).abs().mean()),cn_mean_A=float(cn.mean()),cn_within_0p2A=float(((cn-true_cn).abs()<.2).float().mean()),
              adjacent_ca_mae_A=float((adjacent-target_adj).abs().mean()),adjacent_ca_mean_A=float(adjacent.mean()),
              peptide_angle_mae_deg=float(torch.rad2deg(torch.cat(((angle1-true_angle1).abs(),(angle2-true_angle2).abs()))).mean()),
              peptide_omega_mae_deg=float(torch.rad2deg(omega_error).mean()))

def project(ca0,rot0,local,steps,lr,angle_weight=0.,omega_weight=0.):
 ca_raw=torch.nn.Parameter(ca0.detach().clone())
 raw6=torch.nn.Parameter(torch.cat((rot0[:,:,0],rot0[:,:,1]),-1).detach().clone())
 opt=torch.optim.Adam((ca_raw,raw6),lr=lr)
 original_pair=torch.cdist(ca0,ca0).detach()
 for i in range(steps):
  ca=ca_raw-ca_raw[:1];rot=rotation_6d(raw6);atoms=atoms_from(ca,rot,local)
  cn=(atoms[1:,0]-atoms[:-1,2]).norm(dim=-1)
  adj=(ca[1:]-ca[:-1]).norm(dim=-1)
  anchor=(ca-ca0).square().mean()
  pair=F.smooth_l1_loss(torch.cdist(ca,ca),original_pair,beta=.2)
  align=(3-torch.einsum('lij,lij->l',rot,rot0)).mean()/3
  bond=(cn-.133).square().mean();step=(adj-.38).square().mean()
  a1=angle(atoms[:-1,1],atoms[:-1,2],atoms[1:,0])
  a2=angle(atoms[:-1,2],atoms[1:,0],atoms[1:,1])
  angle_loss=(a1-math.radians(116.1)).square().mean()+(a2-math.radians(121.7)).square().mean()
  omega=dihedral(atoms[:-1,1],atoms[:-1,2],atoms[1:,0],atoms[1:,1])
  omega_error=torch.atan2((omega-math.pi).sin(),(omega-math.pi).cos())
  omega_loss=omega_error.square().mean()
  loss=2*anchor+.2*pair+.2*align+100*bond+30*step+angle_weight*angle_loss+omega_weight*omega_loss
  opt.zero_grad();loss.backward();opt.step()
 return (ca_raw-ca_raw[:1]).detach(),rotation_6d(raw6).detach()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--esm-cache',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--steps',type=int,default=250);ap.add_argument('--lr',type=float,default=.03)
 ap.add_argument('--angle-weight',type=float,default=0.);ap.add_argument('--omega-weight',type=float,default=0.)
 args=ap.parse_args();torch.set_num_threads(min(4,torch.get_num_threads()));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ck=torch.load(args.run/'checkpoint.pt',map_location=device,weights_only=True);model=DirectFrame().to(device);model.load_state_dict(ck['model']);model.eval()
 table=ck['local_table'].to(device);features=torch.load(args.esm_cache,map_location='cpu',weights_only=True)['features']
 rows=[r for r in json.loads(args.manifest.read_text())['proteins'] if r['split']=='test']
 if args.limit:rows=rows[:args.limit]
 results=[]
 for index,row in enumerate(rows):
  p=load_protein(row,features,device);local=table[p['aa']]
  with torch.no_grad():ca0,rot0,_=model(p['embed'],p['aa'])
  before=measures(ca0,rot0,p,local)
  ca,rot=project(ca0,rot0,local,args.steps,args.lr,args.angle_weight,args.omega_weight)
  after=measures(ca,rot,p,local)
  results.append(dict(id=p['id'],before=before,after=after))
  if index==0:
   atoms=atoms_from(ca,rot,local)*10
   pdb_atoms=torch.cat((atoms,torch.zeros_like(atoms[:,:1])),1).cpu()
   write_pdb(args.out.with_suffix('.pdb'),p['seq'],pdb_atoms,torch.zeros(len(p['seq']),dtype=torch.bool))
  if (index+1)%10==0:print(index+1,flush=True)
 keys=list(results[0]['before']);summary={stage:{key:float(np.median([r[stage][key] for r in results])) for key in keys} for stage in ('before','after')}
 args.out.write_text(json.dumps(dict(run=str(args.run),n=len(results),steps=args.steps,lr=args.lr,angle_weight=args.angle_weight,omega_weight=args.omega_weight,summary=summary,results=results),indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
