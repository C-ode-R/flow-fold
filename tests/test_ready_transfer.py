from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class TransferTests(unittest.TestCase):
 def test_dependency_free_check_rejects_changed_cache(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cache=root/'cache';ready=root/'ready';cache.mkdir();ready.mkdir()
   (cache/'config.json').write_text('{}')
   digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   (ready/'audit.json').write_text(json.dumps(dict(passed=True,errors=[])))
   (ready/'cache_checksums.json').write_text(json.dumps({'config.json':digest(cache/'config.json')}))
   (ready/'READY.json').write_text(json.dumps(dict(splits=['train','validation'],records=dict(train=1,validation=1),
      audit_sha256=digest(ready/'audit.json'),cache_checksum_manifest_sha256=digest(ready/'cache_checksums.json'))))
   cmd=[sys.executable,str(ROOT/'scripts/verify_gpu_ready.py'),'--cache',str(cache),'--ready',str(ready)]
   result=subprocess.run(cmd,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
   (cache/'config.json').write_text('{"changed":true}')
   result=subprocess.run(cmd,text=True,capture_output=True);self.assertNotEqual(result.returncode,0)
   self.assertIn('cache checksum mismatch',result.stderr)

if __name__=='__main__':unittest.main()
