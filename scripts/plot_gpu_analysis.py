"""Plot exported validation histories without loading model weights or test data."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
h=[json.loads(s) for s in (a.run/'history.jsonl').read_text().splitlines()];v=[r for r in h if r['event']=='validation'];best=max(v,key=lambda r:r['mean_ca_lddt'])
r=json.loads((a.run/f"validation_{best['epoch']:04d}.json").read_text())['records'];meta={r['id']:r for r in json.loads((a.cache/'validation/index.json').read_text())['records']}
e=[r['epoch'] for r in v]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax[0,0].plot(e,[r['mean_ca_lddt'] for r in v],label='Sequence mean');ax[0,0].plot(e,[r['cluster_macro_ca_lddt'] for r in v],label='Cluster mean');ax[0,0].scatter([best['epoch']],[best['mean_ca_lddt']],color='red',zorder=3,label=f"Selected: epoch {best['epoch']}");ax[0,0].set(title='Validation local structure accuracy',xlabel='Epoch',ylabel='C-alpha lDDT',ylim=(0,1));ax[0,0].legend()
ax[0,1].plot(e,[r['mean_ca_rmsd_A'] for r in v],label='Mean');ax[0,1].plot(e,[r['median_ca_rmsd_A'] for r in v],label='Median');ax[0,1].set(title='Validation global coordinate error',xlabel='Epoch',ylabel='Aligned C-alpha RMSD (A)',ylim=(0,12));ax[0,1].legend()
ax[1,0].plot(e,[r['train_mean_loss'] for r in v],label='Train');ax[1,0].plot(e,[r['mean_loss'] for r in v],label='Validation');ax[1,0].set(title='Loss gap grows after early improvement',xlabel='Epoch',ylabel='Composite loss');ax[1,0].legend()
s=ax[1,1].scatter([meta[r['id']]['length'] for r in r],[r['ca_rmsd_A'] for r in r],c=[r['ca_lddt'] for r in r],cmap='viridis',vmin=0,vmax=1,s=25,alpha=.8);fig.colorbar(s,ax=ax[1,1],label='C-alpha lDDT');ax[1,1].set(title=f"Selected model: all {len(r)} validation sequences",xlabel='Sequence length (residues)',ylabel='Aligned C-alpha RMSD (A)')
for pane in ax.flat:pane.grid(alpha=.2)
fig.suptitle('Fold Flow first GPU run — validation only, one seed',fontsize=15)
fig.savefig(a.out/'learning_curves.png',dpi=160);fig.savefig(a.out/'learning_curves.pdf')
print(a.out/'learning_curves.png')
