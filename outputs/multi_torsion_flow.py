"""Multi-protein torsion flow pilot; run from outputs/."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from flow_fold import ALPHABET, kabsch_rmsd
from se3_flow import load_frames, reconstruct, write_pdb
from torsion_flow import internal_coordinates, extend_chain, source_torsions, wrap

class Velocity(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.aa=nn.Embedding(len(ALPHABET),hidden); self.pos=nn.Embedding(128,hidden); self.role=nn.Embedding(3,hidden)
        self.input=nn.Sequential(nn.Linear(hidden+6,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.SiLU(),nn.Conv1d(hidden,hidden,1)) for _ in range(4)])
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,theta,t,aa):
        b,n=theta.shape; atom=torch.arange(3,n+3,device=theta.device); res=atom//3
        base=self.aa(aa)[res]+self.aa(aa).mean(0)+self.pos(res)+self.role(atom%3)
        time=t[:,None].expand_as(theta); relative=res.float()[None].expand_as(theta)/max(len(aa)-1,1)
        dyn=torch.stack((theta.sin(),theta.cos(),time,(2*math.pi*time).sin(),(2*math.pi*time).cos(),relative),-1)
        h=self.input(torch.cat((base[None].expand(b,-1,-1),dyn),-1)).transpose(1,2)
        for block in self.blocks: h=h+block(h)
        return self.head(h.transpose(1,2)).squeeze(-1)

def load(spec,folder,device):
    name,chain=spec.split(':'); seq,ca,rot,temp,mask,scale=load_frames(folder/f'{name}.cif',chain)
    xyz=reconstruct(ca,rot,temp)[:,:3].reshape(-1,3).to(device)*scale
    lens,angs,tors=internal_coordinates(xyz)
    return dict(name=name,chain=chain,seq=seq,xyz=xyz,ca=xyz[1::3],ids=torch.tensor([ALPHABET.index(a) for a in seq],device=device),lens=lens,angs=angs,tors=tors)

def geometry(proteins):
    lens=torch.stack([torch.cat([p['lens'][i::3] for p in proteins]).mean() for i in range(3)])
    angs=torch.stack([torch.cat([p['angs'][i::3] for p in proteins]).mean() for i in range(3)])
    nca=torch.cat([(p['xyz'][1::3]-p['xyz'][0::3]).norm(dim=-1) for p in proteins]).mean()
    cac=torch.cat([(p['xyz'][2::3]-p['xyz'][1::3]).norm(dim=-1) for p in proteins]).mean()
    first=torch.cat([p['angs'][1::3] for p in proteins]).mean()
    return lens,angs,nca,cac,first

def backbone(tors,g):
    lens,angs,nca,cac,first=g; z=torch.zeros((),device=tors.device)
    root=torch.stack((torch.stack((z,z,z)),torch.stack((nca,z,z)),torch.stack((nca-cac*first.cos(),cac*first.sin(),z))))
    role=torch.arange(tors.shape[1],device=tors.device)%3
    return extend_chain(root[None].expand(tors.shape[0],-1,-1),lens[role],angs[role],tors).reshape(tors.shape[0],-1,3,3)

def rmsd(a,b):
    return float(kabsch_rmsd(a-a.mean(0),b-b.mean(0)))

@torch.no_grad()
def sample(model,p,g,steps,seed):
    torch.manual_seed(seed); theta=source_torsions(len(p['tors']),p['ids'].device,1); dt=1/steps
    for i in range(steps):
        t=torch.tensor([i*dt],device=theta.device); v=model(theta,t,p['ids'])
        theta=theta+dt*(v+model(theta+dt*v,t+dt,p['ids']))/2
    return backbone(theta,g)[0],theta[0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=Path('multi_data')); ap.add_argument('--out',type=Path,default=Path('run_multi'))
    ap.add_argument('--train',nargs='+',default=['1CRN:A','1GB1:A','1VII:A','1FME:A']); ap.add_argument('--holdout',nargs='+',default=['1L2Y:A','2CI2:I'])
    ap.add_argument('--steps',type=int,default=6000); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--ode-steps',type=int,default=60)
    ap.add_argument('--eval-seeds',type=int,default=8); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); torch.manual_seed(args.seed); torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=[load(s,args.data,device) for s in args.train]; holdout=[load(s,args.data,device) for s in args.holdout]; g=geometry(train)
    model=Velocity().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); history=[]
    for step in range(1,args.steps+1):
        p=train[(step-1)%len(train)]; theta0=source_torsions(len(p['tors']),device,args.batch)
        velocity=wrap(p['tors'][None]-theta0); t=torch.rand(args.batch,device=device)
        loss=F.mse_loss(model(theta0+t[:,None]*velocity,t,p['ids']),velocity)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%500==0 or step==args.steps:
            history.append(dict(step=step,velocity_mse=float(loss.detach()))); print(step,float(loss.detach()),flush=True)
    results=[]
    for split,group in [('train',train),('holdout',holdout)]:
        for p in group:
            oracle=backbone(p['tors'][None],g)[0]; errs=[]; mirror=[]; tmae=[]; first=None
            for j in range(args.eval_seeds):
                atoms,theta=sample(model,p,g,args.ode_steps,args.seed+1+j)
                if first is None: first=atoms
                errs.append(rmsd(atoms[:,1],p['ca'])); mirror.append(rmsd(atoms[:,1],p['ca']*torch.tensor([-1.,1.,1.],device=device)))
                tmae.append(float(wrap(theta-p['tors']).abs().mean()))
            row=dict(id=p['name'],chain=p['chain'],split=split,residues=len(p['seq']),sequence=p['seq'],geometry_oracle_ca_rmsd_A=rmsd(oracle[:,1],p['ca']),median_ca_rmsd_A=float(np.median(errs)),min_ca_rmsd_A=min(errs),median_mirror_rmsd_A=float(np.median(mirror)),target_closer_count=sum(a<b for a,b in zip(errs,mirror)),median_torsion_mae_rad=float(np.median(tmae)),seed_ca_rmsd_A=errs)
            results.append(row); print(json.dumps({k:v for k,v in row.items() if k not in ('sequence','seed_ca_rmsd_A')}),flush=True)
            dummy=torch.cat((first,torch.zeros_like(first[:,:1])),1).cpu()
            write_pdb(args.out/f"{split}_{p['name']}_seed{args.seed+1}.pdb",p['seq'],dummy,torch.zeros(len(p['seq']),dtype=torch.bool))
    summary=dict(training_ids=args.train,holdout_ids=args.holdout,steps=args.steps,batch=args.batch,ode_steps=args.ode_steps,eval_seeds=args.eval_seeds,device=str(device),geometry='three backbone atom-role mean lengths and angles from training proteins only',source='generic alpha helix torsions plus Gaussian noise',results=results,history=history)
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2))
    torch.save(dict(model=model.state_dict(),geometry=[x.cpu() for x in g],training_ids=args.train,holdout_ids=args.holdout,seed=args.seed),args.out/'checkpoint.pt')
if __name__=='__main__': main()
