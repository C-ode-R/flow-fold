import json
from pathlib import Path
import numpy as np
import torch

AA='ACDEFGHIKLMNPQRSTVWY'


def prepare_record(row,embedding):
    sequence=row['entity']['sequence'];length=len(sequence)
    atoms=np.zeros((length,4,3),dtype=np.float32);observed=np.zeros((length,4),dtype=bool)
    for a in row['ligand_metadata']['backbone_atoms']:
        i=a['label_seq_id']-1;j=['N','CA','C','O'].index(a['atom'])
        if not 0<=i<length:raise ValueError('residue index outside sequence')
        atoms[i,j]=a['xyz_A'];observed[i,j]=True
    valid=observed[:,:3].all(1)
    e1=atoms[:,2]-atoms[:,1];e1=e1/np.maximum(np.linalg.norm(e1,axis=-1,keepdims=True),1e-8)
    n=atoms[:,0]-atoms[:,1];e2=n-(n*e1).sum(-1,keepdims=True)*e1
    norm=np.linalg.norm(e2,axis=-1)
    valid &= norm>1e-4
    e2=e2/np.maximum(norm[:,None],1e-8)
    r=np.stack((e1,e2,np.cross(e1,e2)),axis=-1)
    r[~valid]=np.eye(3)
    center=atoms[valid,1].mean(0)
    atoms=(atoms-center)/10
    x=atoms[:,1].copy();x[~valid]=0
    links=np.zeros(length,dtype=bool)
    distance=np.linalg.norm(atoms[:-1,2]-atoms[1:,0],axis=-1)*10
    links[:-1]=valid[:-1]&valid[1:]&(distance>=1.1)&(distance<=1.8)
    if valid.sum()<3:raise ValueError('too few complete frames')
    return dict(id=row['sequence_id'],sequence=sequence,aa=torch.tensor([AA.index(a) for a in sequence]),
        embedding=embedding.float(),x=torch.from_numpy(x),r=torch.from_numpy(r),
        atoms=torch.from_numpy(atoms),observed=torch.from_numpy(observed),frame_mask=torch.from_numpy(valid),
        links=torch.from_numpy(links),split=row['split'])


def collate(records,device):
    b=len(records);l=max(len(r['aa']) for r in records);dim=records[0]['embedding'].shape[-1]
    result=dict(aa=torch.zeros(b,l,dtype=torch.long),embedding=torch.zeros(b,l,dim),
        x=torch.zeros(b,l,3),r=torch.eye(3).expand(b,l,3,3).clone(),
        atoms=torch.zeros(b,l,4,3),observed=torch.zeros(b,l,4,dtype=torch.bool),
        frame_mask=torch.zeros(b,l,dtype=torch.bool),links=torch.zeros(b,l,dtype=torch.bool),
        mask=torch.zeros(b,l,dtype=torch.bool))
    for i,row in enumerate(records):
        n=len(row['aa']);result['mask'][i,:n]=True
        for k in result:
            if k!='mask':result[k][i,:n]=row[k]
    return {k:v.to(device) for k,v in result.items()}
