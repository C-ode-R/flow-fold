from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_flow import roundtrip,oracle
from foldflow.geometry import so3_exp,backbone_with_oxygen

class DiagnosticTests(unittest.TestCase):
 def batch(self):
  torch.manual_seed(4)
  x=torch.randn(1,8,3);x-=x.mean(1,keepdim=True)
  r=so3_exp(torch.randn(1,8,3))
  atoms=backbone_with_oxygen(x,r)
  return dict(x=x,r=r,atoms=atoms,mask=torch.ones(1,8,dtype=torch.bool),
      frame_mask=torch.ones(1,8,dtype=torch.bool),observed=torch.ones(1,8,4,dtype=torch.bool),links=torch.ones(1,8,dtype=torch.bool))
 def test_roundtrip_and_oracle(self):
  b=self.batch()
  self.assertTrue(roundtrip(b)['passed'])
  self.assertTrue(oracle(b,128,5)['passed'])
 def test_bad_frame_is_detected(self):
  b=self.batch();b['r'][0,0,:,0]*=-1
  self.assertFalse(roundtrip(b)['passed'])

if __name__=='__main__':unittest.main()
