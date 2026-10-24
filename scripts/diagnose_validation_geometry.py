"""Validation-only long-range contacts and backbone geometry diagnostics."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foldflow.dataset import ShardDataset,atomic_json
from foldflow.folding import FoldingModel
from foldflow.runtime import loader,move,predict
from foldflow.geometry import backbone_with_oxygen

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    saved=torch.load(a.run/'best.pt',map_location='cpu',weights_only=True)
    ds=ShardDataset(a.cache,'validation');model=FoldingModel(**saved['model_config']).to('mps');model.load_state_dict(saved['model']);model.eval()
    rows=[]
    for cpu,meta in loader(ds,8,0,0,0,'mps',False):
        batch=move(cpu,'mps');px,pr=predict(model,batch,'fp32');atoms=backbone_with_oxygen(px,pr).cpu().numpy()*10
        px=px.cpu().numpy()*10;tx=cpu['x'].numpy()*10
        for i,m in enumerate(meta):
            valid=cpu['frame_mask'][i].numpy();ids=np.flatnonzero(valid);pred=px[i,valid];true=tx[i,valid]
            pd=np.linalg.norm(pred[:,None]-pred[None,:],axis=-1);td=np.linalg.norm(true[:,None]-true[None,:],axis=-1)
            sep=np.abs(ids[:,None]-ids[None,:]);tri=np.triu(np.ones_like(sep,dtype=bool),1)
            long=(sep>=12)&tri;true_contacts=long&(td<8);pred_contacts=long&(pd<8)
            tp=int((true_contacts&pred_contacts).sum());nt=int(true_contacts.sum());npred=int(pred_contacts.sum())
            def score(mask):
                e=np.abs(pd-td)[mask];return float(np.mean([np.mean(e<t) for t in [.5,1,2,4]])) if len(e) else None
            links=cpu['links'][i,:-1].numpy();cn=np.linalg.norm(atoms[i,:-1,2]-atoms[i,1:,0],axis=-1)[links]
            rgp=np.sqrt(((pred-pred.mean(0))**2).sum(-1).mean());rgt=np.sqrt(((true-true.mean(0))**2).sum(-1).mean())
            rows.append(dict(id=m['id'],length=len(valid.nonzero()[0]),long_range_true_contacts=nt,long_range_pred_contacts=npred,long_range_tp=tp,
                contact_recall=tp/nt if nt else None,contact_precision=tp/npred if npred else None,
                local_lddt=score(tri&(sep<12)&(td<15)),long_range_lddt=score(long&(td<15)),
                ca_clash_pairs=int((tri&(sep>1)&(pd<2)).sum()),rg_ratio=float(rgp/rgt),
                peptide_cn_outside_1p1_1p8_count=int(((cn<1.1)|(cn>1.8)).sum()),peptide_link_count=len(cn),
                peptide_cn_mean_abs_error_A=float(np.abs(cn-1.33).mean()) if len(cn) else None))
    def mean(key):return float(np.mean([r[key] for r in rows if r[key] is not None]))
    summary=dict(count=len(rows),long_range_contact_recall_macro=mean('contact_recall'),long_range_contact_precision_macro=mean('contact_precision'),
        long_range_contact_recall_micro=sum(r['long_range_tp'] for r in rows)/max(1,sum(r['long_range_true_contacts'] for r in rows)),
        long_range_contact_precision_micro=sum(r['long_range_tp'] for r in rows)/max(1,sum(r['long_range_pred_contacts'] for r in rows)),
        proteins_with_true_long_contacts=sum(r['long_range_true_contacts']>0 for r in rows),
        local_lddt_mean=mean('local_lddt'),long_range_lddt_mean=mean('long_range_lddt'),
        proteins_with_ca_clash=sum(r['ca_clash_pairs']>0 for r in rows),
        mean_rg_ratio=mean('rg_ratio'),median_rg_ratio=float(np.median([r['rg_ratio'] for r in rows])),
        peptide_cn_outside_fraction=sum(r['peptide_cn_outside_1p1_1p8_count'] for r in rows)/sum(r['peptide_link_count'] for r in rows),
        mean_per_protein_peptide_cn_abs_error_A=mean('peptide_cn_mean_abs_error_A'))
    clashes=0;bad=0;total=0
    for row in ds:
        coords=row['x'][row['frame_mask']]*10
        ids=torch.where(row['frame_mask'])[0];sep=(ids[:,None]-ids).abs()
        clashes+=int(((torch.cdist(coords,coords)<2)&(sep>1)).any())
        atoms=backbone_with_oxygen(row['x'][None],row['r'][None])[0]*10
        d=(atoms[:-1,2]-atoms[1:,0]).norm(dim=-1)[row['links'][:-1]]
        bad+=int(((d<1.1)|(d>1.8)).sum());total+=len(d)
    control=dict(proteins_with_ca_clash=clashes,peptide_cn_outside_fraction=bad/total,links=total)
    atomic_json(a.out,dict(native_ideal_frame_control=control,summary=summary,records=rows,test_read=False,definition='Long range: sequence separation >=12, contact distance <8A. CA clash: nonadjacent CA distance <2A. Peptide gate: ideal reconstructed C-N outside [1.1,1.8]A; diagnostic only.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
