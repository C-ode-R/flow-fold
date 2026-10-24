"""Evaluate the frozen validation-selected model once after training completes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from foldflow.dataset import ShardDataset,audit_split_isolation,sha256,atomic_json
from foldflow.folding import FoldingModel
from foldflow.runtime import validate_device,evaluate,code_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=0)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0:p.error('invalid batch size/workers')
    if not (args.run/'TRAINING_COMPLETE.json').exists():raise ValueError('training not complete; test must remain sealed')
    if (args.run/'TEST_EVALUATED.json').exists():raise ValueError('test already evaluated for this frozen run; read TEST_EVALUATED.json')
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('test output must be empty')
    state=torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)
    contract=state['contract']
    if state.get('selected_by')!='validation_mean_ca_lddt':raise ValueError('checkpoint was not selected by validation')
    if contract['code_sha256']!=code_fingerprint():raise ValueError('training code changed since checkpoint')
    validate_device(args.device,contract['precision']);torch.set_num_threads(4)
    train=ShardDataset(args.cache,'train');val=ShardDataset(args.cache,'validation');test=ShardDataset(args.cache,'test')
    if train.fingerprint!=contract['train_fingerprint'] or val.fingerprint!=contract['validation_fingerprint']:
        raise ValueError('train/validation cache changed')
    audit_split_isolation(train,test);audit_split_isolation(val,test)
    model=FoldingModel(**state['model_config']).to(args.device);model.load_state_dict(state['model'])
    summary,records=evaluate(model,test,args.batch_size,args.device,contract['precision'],args.workers)
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(checkpoint_sha256=sha256(args.run/'best.pt'),checkpoint_step=state['step'],
        selected_by=state['selected_by'],best_epoch=state['best_epoch'],test_cache_fingerprint=test.fingerprint,
        summary=summary,records=records,limitations=['C-alpha metrics; no claim of all-atom/side-chain accuracy.',
            'Sequence/entry split isolation checked; distant structural homology and PLM pretraining overlap not excluded.'])
    atomic_json(args.out/'test.json',result)
    atomic_json(args.run/'TEST_EVALUATED.json',dict(result=str((args.out/'test.json').resolve()),checkpoint_sha256=result['checkpoint_sha256']))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
