"""Audit raw pilot candidates; no coordinates are removed or training split made."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

ROOT = Path(__file__).resolve().parents[1]
STANDARD = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())


def rows(d, prefix, fields):
    columns = [d.get(prefix + '.' + f, []) for f in fields]
    n = max(map(len, columns), default=0)
    return [dict(zip(fields, (c[i] if i < len(c) else '?' for c in columns))) for i in range(n)]


def number(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def audit(record):
    d = record.get('_parsed')
    if d is None:
        with gzip.open(ROOT / record['path'], 'rt') as stream:
            d = MMCIF2Dict(stream)
    methods = d.get('_exptl.method', [])
    resolutions = [number(x) for x in d.get('_refine.ls_d_res_high', []) + d.get('_em_3d_reconstruction.resolution', [])]
    resolutions = [x for x in resolutions if x is not None]
    rfree = [number(x) for x in d.get('_refine.ls_R_factor_R_free', [])]
    rfree = [x for x in rfree if x is not None]
    seq = defaultdict(dict)
    for r in rows(d, '_entity_poly_seq', ['entity_id', 'num', 'mon_id']):
        seq[r['entity_id']].setdefault(int(r['num']), set()).add(r['mon_id'])
    assemblies = {r['id']: r for r in rows(d, '_pdbx_struct_assembly', ['id', 'oligomeric_count', 'details'])}
    memberships = defaultdict(set)
    for r in rows(d, '_pdbx_struct_assembly_gen', ['assembly_id', 'asym_id_list']):
        for chain in r['asym_id_list'].split(','):
            memberships[chain].add(r['assembly_id'])
    atoms = rows(d, '_atom_site', ['label_asym_id', 'label_entity_id', 'label_seq_id', 'label_atom_id', 'label_alt_id', 'label_comp_id', 'occupancy', 'pdbx_PDB_model_num', 'Cartn_x', 'Cartn_y', 'Cartn_z'])
    models = sorted({r['pdbx_PDB_model_num'] for r in atoms}, key=lambda x: int(x) if x.isdigit() else 0)
    first_model = models[0]
    chains = defaultdict(list)
    ligands = set()
    for r in atoms:
        if r['pdbx_PDB_model_num'] != first_model:
            continue
        if r['label_seq_id'] in ('.', '?'):
            if r['label_comp_id'] != 'HOH':
                ligands.add(r['label_comp_id'])
        else:
            chains[(r['label_entity_id'], r['label_asym_id'])].append(r)
    output = []
    candidates = {x.split('_', 1)[1] for x in record['candidate_entities']}
    # struct_asym also includes completely unobserved chains.
    for chain_row in rows(d, '_struct_asym', ['id', 'entity_id']):
        entity, chain = chain_row['entity_id'], chain_row['id']
        if entity not in candidates:
            continue
        residues = seq[entity]
        chain_atoms = chains[(entity, chain)]
        by_res = defaultdict(list)
        for a in chain_atoms:
            if (number(a['occupancy']) or 0) > 0:
                by_res[int(a['label_seq_id'])].append(a)
        # Use one coherent altloc per residue; never combine A and B conformers.
        selected = {}
        for idx, aa in by_res.items():
            scores = defaultdict(float)
            for a in aa:
                if a['label_alt_id'] not in ('.', '?') and a['label_atom_id'] in ('N', 'CA', 'C', 'O'):
                    scores[a['label_alt_id']] += number(a['occupancy']) or 0
            chosen = sorted(scores, key=lambda x: (-scores[x], x))[0] if scores else None
            selected[idx] = {a['label_atom_id']: a for a in aa if a['label_alt_id'] in ('.', '?', chosen)}
        complete = {i for i, aa in selected.items() if {'N', 'CA', 'C'} <= aa.keys() and i in residues}
        oxygen = sum('O' in selected[i] for i in complete)
        breaks = []
        for i in sorted(complete):
            if i + 1 in complete:
                a, b = selected[i]['C'], selected[i + 1]['N']
                distance = math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')))
                if not 1.1 <= distance <= 1.8:
                    breaks.append({'after_label_seq_id': i, 'cn_distance_A': round(distance, 3)})
        assembly_rows = [assemblies[a] for a in sorted(memberships[chain]) if a in assemblies]
        counts = [number(a['oligomeric_count']) for a in assembly_rows]
        assembly_class = ('monomer_annotated' if counts and all(x == 1 for x in counts)
                          else 'multimer_annotated' if counts and all(x is not None and x > 1 for x in counts)
                          else 'ambiguous_or_missing')
        n = len(residues)
        flags = []
        if methods != ['X-RAY DIFFRACTION']:
            flags.append('method_not_initial_xray_scope')
        if not resolutions or max(resolutions) > 2.5:
            flags.append('resolution_missing_or_gt_2.5A')
        if not rfree or max(rfree) > 0.30:
            flags.append('rfree_missing_or_gt_0.30')
        if len(complete) / max(n, 1) < 0.95:
            flags.append('backbone_coverage_lt_95pct')
        if any(len(v) != 1 or not v <= STANDARD for v in residues.values()):
            flags.append('nonstandard_or_heterogeneous_sequence')
        if breaks:
            flags.append('peptide_geometry_review')
        if assembly_class != 'monomer_annotated':
            flags.append('assembly_review')
        output.append(dict(entry_id=record['entry_id'], entity_id=entity, label_asym_id=chain,
            sequence_length=n, complete_backbone_residues=len(complete),
            backbone_coverage=round(len(complete) / max(n, 1), 4), oxygen_on_complete=oxygen,
            missing_backbone_label_seq_ids=sorted(set(residues) - complete),
            method=methods, resolution_A=resolutions, rfree=rfree, model_count=len(models),
            altloc_atom_count=sum(a['label_alt_id'] not in ('.', '?') for a in chain_atoms),
            peptide_geometry_flags=breaks, assembly_class=assembly_class, assemblies=assembly_rows,
            entry_nonwater_nonpolymer_components=sorted(ligands),
            preliminary_status='review' if flags else 'eligible_pending_validation_and_context', flags=flags))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    args = p.parse_args()
    pilot = json.loads((args.catalog / 'pilot.json').read_text())
    results = []
    errors = []
    for record in pilot['records']:
        if record['status'] == 'downloaded':
            try:
                results.extend(audit(record))
            except Exception as exc:
                errors.append({'entry_id': record['entry_id'], 'error': str(exc)})
    report = dict(scope='20-entry connectivity pilot; not representative; chain counts include duplicates',
        chains=len(results), entries=len({r['entry_id'] for r in results}),
        statuses=dict(Counter(r['preliminary_status'] for r in results)),
        assembly_classes=dict(Counter(r['assembly_class'] for r in results)),
        flags=dict(Counter(f for r in results for f in r['flags'])), errors=errors, records=results)
    (args.catalog / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
