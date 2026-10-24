"""Build resumable split-specific frozen ESM caches. Test is opt-in, never default."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer,AutoModel
from foldflow.data import prepare_record
from foldflow.dataset import sha256,atomic_json,atomic_torch

SPECS={
 '150M':('facebook/esm2_t30_150M_UR50D','a695f6045e2e32885fa60af20c13cb35398ce30c'),
 '8M':('facebook/esm2_t6_8M_UR50D','c731040fcd8d73dceaa04b0a8e6329b345b0f5df')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('data/cache150'))
    p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation'])
    p.add_argument('--esm-model',choices=list(SPECS),default='150M')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--shard-size',type=int,default=64)
    args=p.parse_args()
    if min(args.batch_size,args.shard_size)<1:p.error('positive batch/shard size required')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    dataset=args.dataset.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (dataset/'COMPLETE').exists():raise ValueError('dataset missing COMPLETE marker')
    checksums=json.loads((dataset/'checksums.json').read_text())
    dataset_id=sha256(dataset/'checksums.json')
    model_id,revision=SPECS[args.esm_model]
    settings=dict(schema=2,dataset_id=dataset_id,model=model_id,revision=revision,shard_size=args.shard_size,units='nm')
    config=out/'config.json'
    if config.exists() and json.loads(config.read_text())!=settings:raise ValueError('cache settings/dataset changed; choose a new output')
    atomic_json(config,settings)
    tasks=[]
    for split in args.splits:
        source=dataset/f'{split}.jsonl'
        digest=sha256(source)
        if digest!=checksums[source.name]:raise ValueError(f'source checksum mismatch: {source}')
        folder=out/split;folder.mkdir(exist_ok=True)
        if (folder/'index.json').exists():
            from foldflow.dataset import ShardDataset
            existing=ShardDataset(out,split)
            if existing.index['source_sha256']!=digest:raise ValueError('existing cache/source mismatch')
            print(f'{split}: verified complete cache ({len(existing)} records)',flush=True);continue
        rows=[json.loads(line) for line in source.open()]
        if any(r['split']!=split for r in rows):raise ValueError('source split label mismatch')
        rows.sort(key=lambda r:(len(r['entity']['sequence']),r['sequence_id']))
        if len({r['sequence_id'] for r in rows})!=len(rows):raise ValueError('duplicate sequence IDs')
        tasks.append((split,folder,rows,digest))
    if not tasks:print('All requested splits already cached.');return
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,token=False)
    model=AutoModel.from_pretrained(model_id,revision=revision,token=False,add_pooling_layer=False).eval().to(args.device)
    model.requires_grad_(False)
    embedding=dict(model=model_id,revision=revision,dim=model.config.hidden_size)
    for split,folder,rows,digest in tasks:
        refs=[];shards=[]
        for start in range(0,len(rows),args.shard_size):
            selected=rows[start:start+args.shard_size]
            name=f'shard_{start//args.shard_size:05d}.pt';path=folder/name
            expected=[r['sequence_id'] for r in selected]
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved.get('source_sha256')!=digest or saved.get('embedding')!=embedding or [r['id'] for r in saved['records']]!=expected:
                    raise ValueError('incompatible saved shard')
                records=saved['records']
            else:
                records=[]
                with torch.no_grad():
                    for offset in range(0,len(selected),args.batch_size):
                        chunk=selected[offset:offset+args.batch_size]
                        sequences=[r['entity']['sequence'] for r in chunk]
                        tokens={k:v.to(args.device) for k,v in tokenizer(sequences,padding=True,return_tensors='pt').items()}
                        features=model(**tokens).last_hidden_state.cpu()
                        for i,row in enumerate(chunk):
                            length=len(sequences[i])
                            if int(tokens['attention_mask'][i].sum())!=length+2:raise ValueError('token/residue mismatch')
                            emb=features[i,1:length+1].clone()
                            if not torch.isfinite(emb).all():raise FloatingPointError('nonfinite ESM embedding')
                            record=prepare_record(row,emb)
                            record['cluster_id']=row['cluster_id']
                            record['entry_ids']=sorted({r['entry_id'] for r in row['observations']})
                            records.append(record)
                atomic_torch(path,dict(records=records,source_sha256=digest,embedding=embedding))
            for offset,r in enumerate(records):
                import hashlib
                refs.append(dict(id=r['id'],length=len(r['sequence']),shard=name,offset=offset,
                    sequence_sha256=hashlib.sha256(r['sequence'].encode()).hexdigest(),
                    cluster_id=r['cluster_id'],entry_ids=r['entry_ids']))
            shards.append(dict(file=name,sha256=sha256(path),count=len(records)))
            print(f'{split}: {min(start+args.shard_size,len(rows))}/{len(rows)} cached',flush=True)
        atomic_json(folder/'index.json',dict(schema=2,complete=True,split=split,dataset_id=dataset_id,
            source_sha256=digest,embedding=embedding,shards=shards,records=refs))
    print('Cache preparation complete. Test remains unprepared unless explicitly requested.')


if __name__=='__main__':main()
