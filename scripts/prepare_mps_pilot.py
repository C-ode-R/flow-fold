"""Select train-only overfit examples and cache frozen ESM2 embeddings on CPU or MPS."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer,AutoModel
from huggingface_hub import HfApi
from foldflow.data import prepare_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=Path('data/mps_pilot/train.jsonl'))
    p.add_argument('--out',type=Path)
    p.add_argument('--esm-model',choices=['8M','35M','150M','650M'],default='150M')
    p.add_argument('--embedding-device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261024)
    args=p.parse_args()
    if not 1<=args.count<=50:p.error('first overfit pilot requires count 1..50')
    if args.out is None:args.out=Path(f'data/mps_pilot/cache_{args.esm_model}_{args.count}.pt')
    if args.out.exists():raise ValueError('cache already exists; choose a new --out')
    if args.embedding_device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    rows=[json.loads(line) for line in args.train.open()]
    if any(r['split']!='train' for r in rows):raise ValueError('train-only input required')
    if args.count>len(rows):raise ValueError('not enough train records')
    selected=random.Random(args.seed).sample(sorted(rows,key=lambda r:r['sequence_id']),args.count)
    names={'8M':'esm2_t6_8M_UR50D','35M':'esm2_t12_35M_UR50D','150M':'esm2_t30_150M_UR50D','650M':'esm2_t33_650M_UR50D'}
    model_id='facebook/'+names[args.esm_model]
    revision=HfApi(token=False).model_info(model_id,token=False).sha
    print(f'Loading {model_id} at {revision} on {args.embedding_device}',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,token=False)
    esm=AutoModel.from_pretrained(model_id,revision=revision,token=False,add_pooling_layer=False).eval().to(args.embedding_device)
    esm.requires_grad_(False)
    revision=getattr(esm.config,'_commit_hash',None)
    records=[]
    with torch.no_grad():
        for i,row in enumerate(selected,1):
            sequence=row['entity']['sequence']
            encoded={k:v.to(args.embedding_device) for k,v in tokenizer(sequence,return_tensors='pt').items()}
            embedding=esm(**encoded).last_hidden_state[0,1:len(sequence)+1].cpu()
            if embedding.shape!=(len(sequence),esm.config.hidden_size):raise ValueError('ESM residue/token mismatch')
            records.append(prepare_record(row,embedding))
            print(f'Cached {i}/{len(selected)}: {row["sequence_id"]}, L={len(sequence)}',flush=True)
    payload=dict(records=records,provenance=dict(model=model_id,revision=revision,embedding_dim=esm.config.hidden_size,
        train_file=str(args.train.resolve()),train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(),
        seed=args.seed,selected_ids=[r['id'] for r in records],units='nm',schema=1))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.out)
    print(f'Saved {args.out}')


if __name__=='__main__':main()
