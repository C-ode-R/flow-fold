"""CPU tests of split isolation, complete epoch accounting, resume and sealed test."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foldflow.dataset import ShardDataset,LengthBatches,audit_split_isolation,atomic_json,atomic_torch,sha256
from foldflow.geometry import so3_exp,backbone_with_oxygen

ROOT=Path(__file__).resolve().parents[1]


def fixture_split(root,split,count,start):
    folder=root/split;folder.mkdir(parents=True,exist_ok=True)
    records=[];refs=[]
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    for i in range(count):
        g=torch.Generator().manual_seed(start+i);length=7+i
        seq=''.join(alphabet[(start+i+j)%20] for j in range(length))
        x=torch.randn(1,length,3,generator=g);x-=x.mean(1,keepdim=True)
        r=so3_exp(torch.randn(1,length,3,generator=g))
        id=f'{split}_{i}'
        row=dict(id=id,sequence=seq,aa=torch.tensor([alphabet.index(a) for a in seq]),
            embedding=torch.randn(length,320,generator=g),x=x[0],r=r[0],
            atoms=backbone_with_oxygen(x,r)[0],observed=torch.ones(length,4,dtype=torch.bool),
            frame_mask=torch.ones(length,dtype=torch.bool),links=torch.zeros(length,dtype=torch.bool),
            split=split,cluster_id='cluster_'+id,entry_ids=['entry_'+id])
        records.append(row)
        refs.append(dict(id=id,length=length,shard='shard.pt',offset=i,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
                         cluster_id=row['cluster_id'],entry_ids=row['entry_ids']))
    atomic_torch(folder/'shard.pt',dict(records=records))
    atomic_json(folder/'index.json',dict(schema=2,complete=True,split=split,dataset_id='fixture',source_sha256='fixture',
        embedding=dict(model='synthetic_for_tests',revision='fixture',dim=320),records=refs,
        shards=[dict(file='shard.pt',count=count,sha256=sha256(folder/'shard.pt'))]))


class PipelineTests(unittest.TestCase):
 def test_bucket_covers_every_record_and_cache_integrity(self):
  lengths=[10,8,15,9,7,16,12]
  batches=list(LengthBatches(lengths,3,42,0))
  self.assertEqual(sorted(x for b in batches for x in b),list(range(len(lengths))))
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);fixture_split(root,'train',4,0);fixture_split(root,'validation',2,10)
   train=ShardDataset(root,'train');val=ShardDataset(root,'validation');audit_split_isolation(train,val)
   self.assertEqual(train[0]['embedding'].device.type,'cpu')
   val.records[0]['cluster_id']=train.records[0]['cluster_id']
   with self.assertRaisesRegex(ValueError,'overlap'):audit_split_isolation(train,val)
   with (root/'train/shard.pt').open('ab') as f:f.write(b'corrupt')
   with self.assertRaisesRegex(ValueError,'checksum'):ShardDataset(root,'train')

 def test_padding_does_not_change_valid_predictions(self):
  from foldflow.data import collate
  from foldflow.folding import FoldingModel
  torch.manual_seed(44)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);fixture_split(root,'train',2,0);dataset=ShardDataset(root,'train')
   m=FoldingModel(esm_dim=320,hidden=32,pair_dim=16,pair_layers=1,geometry_layers=1,recycles=2).eval()
   for layer in m.updates:torch.nn.init.normal_(layer.weight,std=.02)
   for triangle in m.triangles:torch.nn.init.normal_(triangle.out.weight,std=.02)
   batch=collate([dataset[0],dataset[1]],'cpu')
   x,r=m(batch['embedding'],batch['aa'],batch['mask'])
   for i in range(2):
    single=collate([dataset[i]],'cpu');xx,rr=m(single['embedding'],single['aa'],single['mask'])
    n=len(dataset[i]['sequence'])
    self.assertTrue(torch.allclose(x[i,:n],xx[0],atol=1e-5))
    self.assertTrue(torch.allclose(r[i,:n],rr[0],atol=1e-5))

 def test_resume_matches_full_run_and_test_is_separate(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cache=root/'cache';fixture_split(cache,'train',4,0);fixture_split(cache,'validation',2,10)
   base=[sys.executable,str(ROOT/'scripts/train_dataset.py'),'--cache',str(cache),'--device','cpu',
        '--epochs','2','--batch-size','2','--workers','0','--hidden','32','--pair-dim','16',
        '--pair-layers','1','--geometry-layers','1','--recycles','1','--warmup-updates','0',
        '--save-every','1','--log-every','1']
   def run(args,ok=True):
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    if ok and r.returncode:self.fail(r.stdout+r.stderr)
    if not ok:self.assertNotEqual(r.returncode,0)
    return r
   full=root/'full';resumed=root/'resumed'
   run(base+['--out',str(full)])
   # No test directory exists, proving training does not require native test data.
   self.assertFalse((cache/'test').exists())
   run(base+['--out',str(resumed),'--max-updates','1'])
   self.assertFalse((resumed/'TRAINING_COMPLETE.json').exists())
   result=run([sys.executable,str(ROOT/'scripts/evaluate_test.py'),'--run',str(resumed),
      '--cache',str(cache),'--out',str(root/'blocked_test'),'--device','cpu'],ok=False)
   self.assertIn('training not complete',result.stderr)
   run(base+['--out',str(resumed),'--resume'])
   a=torch.load(full/'last.pt',weights_only=True,map_location='cpu')
   b=torch.load(resumed/'last.pt',weights_only=True,map_location='cpu')
   self.assertEqual(a['step'],4);self.assertEqual(a['step'],b['step'])
   for k in a['model']:self.assertTrue(torch.allclose(a['model'][k],b['model'][k],atol=1e-7),k)
   self.assertEqual(a['scheduler']['last_epoch'],b['scheduler']['last_epoch'])
   self.assertEqual(a['best_epoch'],b['best_epoch'])
   fixture_split(cache,'test',2,16)
   run([sys.executable,str(ROOT/'scripts/evaluate_test.py'),'--run',str(full),'--cache',str(cache),
        '--out',str(root/'test_result'),'--device','cpu','--batch-size','2'])
   report=json.loads((root/'test_result/test.json').read_text())
   self.assertEqual(report['summary']['count'],2)
   self.assertTrue((full/'TEST_EVALUATED.json').exists())


if __name__=='__main__':unittest.main()
