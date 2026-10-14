import json,sys,torch
from pathlib import Path
from transformers import AutoTokenizer,EsmModel
manifest=Path(sys.argv[1]); out=Path(sys.argv[2]); name='facebook/esm2_t6_8M_UR50D'
rows=json.loads(manifest.read_text())['proteins']; tokenizer=AutoTokenizer.from_pretrained(name)
model=EsmModel.from_pretrained(name,add_pooling_layer=False).eval(); torch.set_num_threads(4)
features={}
with torch.no_grad():
 for start in range(0,len(rows),16):
  batch=rows[start:start+16]; seq=[r['sequence'] for r in batch]
  inp=tokenizer(seq,return_tensors='pt',padding=True)
  hidden=model(**inp).last_hidden_state
  for r,h in zip(batch,hidden): features[f"{r['id']}:{r['chain']}"]=h[1:len(r['sequence'])+1].cpu().half()
  if start%160==0: print(start,flush=True)
torch.save(dict(model=name,features=features),out); print(out,flush=True)
