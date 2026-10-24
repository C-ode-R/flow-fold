"""Exact sequence deduplication, exhaustive MMseqs2 similarity graph, frozen splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

from pilot_pdb import save


def seq_id(seq):
    return 's_'+hashlib.sha256(seq.encode()).hexdigest()[:24]


def rank(row):
    return (-row['backbone_coverage'],max(row['resolution_A']),max(row['rfree']),
            -row['oxygen_on_complete'],row['entry_id'],row['label_asym_id'])


class Components:
    def __init__(self,ids):self.parent={x:x for x in ids}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def allocate(groups,seed):
    rng=random.Random(seed)
    groups=list(groups)
    rng.shuffle(groups)
    groups.sort(key=len,reverse=True)
    total=sum(map(len,groups))
    targets=dict(train=total*.8,validation=total*.1,test=total*.1)
    counts=Counter()
    assignments={}
    for group in groups:
        # Largest-first capacity allocation keeps components intact.
        split=max(targets,key=lambda s:(targets[s]-counts[s])/max(targets[s],1))
        for x in group:assignments[x]=split
        counts[split]+=len(group)
    return assignments,dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/catalog/full_screen_v1/candidates.jsonl'))
    p.add_argument('--out',type=Path,default=Path('data/dataset_v1_final'))
    p.add_argument('--mmseqs',default='tools/mmseqs/bin/mmseqs')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261024)
    args=p.parse_args()
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output must be empty; frozen splits must not be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    unique={}
    members=defaultdict(list)
    all_rows=0
    with args.source.open() as stream,(out/'observations.jsonl').open('w') as observations:
        for line_num,line in enumerate(stream,1):
            row=json.loads(line)
            sequence=row['entity']['sequence']
            if not 40<=len(sequence)<=128 or set(sequence)-set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError('unexpected sequence in accepted candidates')
            if row['decision']!='candidate' or row['flags']:raise ValueError('noncandidate source row')
            id=seq_id(sequence)
            if id in unique and unique[id]['entity']['sequence']!=sequence:raise ValueError('hash collision')
            ref=dict(entry_id=row['entry_id'],entity_id=row['entity_id'],label_asym_id=row['label_asym_id'],source_line=line_num,
                     source_path=row['source_path'])
            members[id].append(ref)
            if id not in unique or rank(row)<rank(unique[id]):unique[id]=row
            observation=dict(sequence_id=id,**ref,ligand_metadata={k:v for k,v in row['ligand_metadata'].items() if k!='backbone_atoms'})
            observations.write(json.dumps(observation)+'\n')
            all_rows+=1
    if len(unique)<3:raise ValueError('not enough unique sequences')
    with (out/'sequences.fasta').open('w') as fasta:
        for id in sorted(unique):fasta.write(f'>{id}\n{unique[id]["entity"]["sequence"]}\n')
    print(f'Observations: {all_rows}; exact unique sequences: {len(unique)}',flush=True)
    mmseqs=str(Path(args.mmseqs).resolve()) if '/' in args.mmseqs else args.mmseqs
    version=subprocess.check_output([mmseqs,'version'],text=True).strip()
    command=[mmseqs,'easy-search',str(out/'sequences.fasta'),str(out/'sequences.fasta'),
        str(out/'all_pairs.tsv'),str(out/'mmseqs_tmp'),'--threads',str(args.threads),
        '--prefilter-mode','2','--min-seq-id','0.3','-c','0.8','--cov-mode','0',
        '--alignment-mode','3','--seq-id-mode','0','-e','1000000','--max-seqs',str(len(unique)+1),
        '--mask','0','--comp-bias-corr','0','--format-output','query,target,fident,qcov,tcov']
    save(out/'provenance.json',dict(source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        seed=args.seed,mmseqs_version=version,command=command,
        representative_order='backbone coverage desc, resolution asc, Rfree asc, observed oxygen desc, entry and chain lexicographic',
        identity_threshold=.3,query_and_target_coverage=.8,split_targets=dict(train=.8,validation=.1,test=.1)))
    with (out/'mmseqs.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    components=Components(unique)
    edges=[]
    with (out/'all_pairs.tsv').open() as stream:
        for line in stream:
            a,b,identity,qcov,tcov=line.split()
            if a not in unique or b not in unique:raise ValueError('unexpected MMseqs sequence ID')
            if a!=b and float(identity)>=.3 and min(float(qcov),float(tcov))>=.8:
                components.join(a,b);edges.append((a,b))
    # Also group distinct sequences from the same PDB entry, including nonrepresentative observations.
    by_entry=defaultdict(set)
    for id,refs in members.items():
        for r in refs:by_entry[r['entry_id']].add(id)
    for ids in by_entry.values():
        ids=sorted(ids)
        for id in ids[1:]:components.join(ids[0],id)
    groups=defaultdict(list)
    for id in sorted(unique):groups[components.find(id)].append(id)
    assignments,counts=allocate(groups.values(),args.seed)
    cross_hits=[(a,b) for a,b in edges if assignments[a]!=assignments[b]]
    if cross_hits:raise ValueError('cross-split hits remain')
    if any(len({assignments[x] for x in ids})!=1 for ids in by_entry.values()):raise ValueError('entry crosses splits')
    if set(assignments.values())!={'train','validation','test'}:raise ValueError('cannot form three nonempty splits')
    with (out/'manifest.jsonl').open('w') as manifest:
        handles={s:(out/(s+'.jsonl')).open('w') for s in counts}
        try:
            for id in sorted(unique):
                row=dict(unique[id],sequence_id=id,cluster_id=components.find(id),split=assignments[id],
                         duplicate_observation_count=len(members[id]),observations=members[id])
                line=json.dumps(row)+'\n';manifest.write(line);handles[assignments[id]].write(line)
        finally:
            for h in handles.values():h.close()
    summary=dict(source_chain_observations=all_rows,unique_sequences=len(unique),removed_duplicate_observations=all_rows-len(unique),
        original_entries=len(by_entry),components=len(groups),largest_component=max(map(len,groups.values())),split_counts=counts,
        detected_cross_split_sequence_hits=0,entries_crossing_splits=0,exact_sequences_crossing_splits=0,
        split_component_counts=dict(Counter(assignments[g[0]] for g in groups.values())),
        limitations=['30% identity and 80% both-sequence coverage under this alignment search; remote/structural homology not excluded.',
            'One structure per exact sequence; alternative ligand/conformational observations preserved separately.',
            'Connected components include same-entry links, so target split fractions are approximate.',
            'All accepted structures are preliminary; detailed wwPDB validation not yet applied.',
            'PLM pretraining sequence overlap is not evaluated.'])
    save(out/'summary.json',summary)
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    save(out/'checksums.json',checksums)
    (out/'COMPLETE').write_text('deduplication, graph split and detected-edge leakage audit completed\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
