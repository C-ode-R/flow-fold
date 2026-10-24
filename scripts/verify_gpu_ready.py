"""Dependency-free integrity check of train/validation cache transfer."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/cache150'))
    p.add_argument('--ready',type=Path,default=Path('data/gpu_ready'))
    args=p.parse_args()
    marker=json.loads((args.ready/'READY.json').read_text())
    if digest(args.ready/'audit.json')!=marker['audit_sha256']:raise ValueError('audit report checksum mismatch')
    if digest(args.ready/'cache_checksums.json')!=marker['cache_checksum_manifest_sha256']:raise ValueError('cache manifest checksum mismatch')
    audit=json.loads((args.ready/'audit.json').read_text())
    if not audit['passed'] or audit['errors'] or marker['splits']!=['train','validation']:raise ValueError('QA not passed')
    files=json.loads((args.ready/'cache_checksums.json').read_text())
    for name,expected in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or (name!='config.json' and rel.parts[0] not in ('train','validation')):
            raise ValueError('unexpected transfer path')
        if digest(args.cache/rel)!=expected:raise ValueError(f'cache checksum mismatch: {name}')
    print(json.dumps(dict(verified=True,files=len(files),records=marker['records'],test_read=False),indent=2))


if __name__=='__main__':main()
