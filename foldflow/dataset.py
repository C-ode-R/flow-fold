"""CPU shard dataset, reproducible length buckets, split isolation and fingerprints."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset,Sampler
from .data import collate


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def atomic_torch(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,temporary);temporary.replace(path)


class ShardDataset(Dataset):
    def __init__(self,root,split,verify=True):
        self.root=Path(root);self.split=split
        self.index_path=self.root/split/'index.json'
        self.index=json.loads(self.index_path.read_text())
        if self.index['split']!=split or not self.index.get('complete'):raise ValueError('incomplete or mismatched split cache')
        self.fingerprint=sha256(self.index_path)
        self.records=self.index['records'];self.lengths=[r['length'] for r in self.records]
        self.cache=OrderedDict()
        if not self.records:raise ValueError('empty split')
        if len({r['id'] for r in self.records})!=len(self.records):raise ValueError('duplicate IDs')
        if len({r['sequence_sha256'] for r in self.records})!=len(self.records):raise ValueError('duplicate exact sequences')
        for desc in self.index['shards']:
            path=self.root/split/desc['file']
            if verify and sha256(path)!=desc['sha256']:raise ValueError(f'cache checksum mismatch: {path}')
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        ref=self.records[index];name=ref['shard']
        if name not in self.cache:
            self.cache[name]=torch.load(self.root/self.split/name,map_location='cpu',weights_only=True)['records']
            if len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        row=self.cache[name][ref['offset']]
        if row['id']!=ref['id'] or row['split']!=self.split:raise ValueError('shard/manifest mismatch')
        return row


class LengthBatches(Sampler):
    def __init__(self,lengths,batch_size,seed,epoch,shuffle=True):
        if batch_size<1:raise ValueError('batch size must be positive')
        self.lengths=lengths;self.batch_size=batch_size;self.seed=seed;self.shuffle=shuffle
        self.set_epoch(epoch)
    def set_epoch(self,epoch):
        lengths,batch_size=self.lengths,self.batch_size
        ids=list(range(len(lengths)));rng=random.Random(self.seed+epoch)
        if self.shuffle:rng.shuffle(ids)
        batches=[]
        bucket=batch_size*20
        for start in range(0,len(ids),bucket):
            chunk=sorted(ids[start:start+bucket],key=lambda i:lengths[i])
            batches.extend(chunk[j:j+batch_size] for j in range(0,len(chunk),batch_size))
        if self.shuffle:rng.shuffle(batches)
        self.batches=batches
    def __iter__(self):return iter(self.batches)
    def __len__(self):return len(self.batches)


def collate_shards(records):
    return collate(records,'cpu'),[dict(id=r['id'],sequence=r['sequence'],cluster_id=r['cluster_id']) for r in records]


def audit_split_isolation(train,validation):
    if train.index['embedding']!=validation.index['embedding'] or train.index['dataset_id']!=validation.index['dataset_id']:
        raise ValueError('embedding/dataset revisions differ across splits')
    fields=('id','sequence_sha256','cluster_id')
    for field in fields:
        a={r[field] for r in train.records};b={r[field] for r in validation.records}
        if a&b:raise ValueError(f'cross-split overlap: {field}')
    a={e for r in train.records for e in r['entry_ids']}
    b={e for r in validation.records for e in r['entry_ids']}
    if a&b:raise ValueError('cross-split PDB entry overlap')
