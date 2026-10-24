from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foldflow.direct import DirectFrames,direct_loss

class DirectTests(unittest.TestCase):
 def test_forward_has_no_target_inputs_and_gradients_finite(self):
  torch.manual_seed(1)
  model=DirectFrames(hidden=32,pair_dim=16,pair_layers=1,geometry_layers=1)
  embedding=torch.randn(1,6,320);aa=torch.randint(20,(1,6));mask=torch.ones(1,6,dtype=torch.bool)
  x,r=model(embedding,aa,mask)
  self.assertTrue(torch.allclose(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-5))
  self.assertTrue(torch.allclose(torch.linalg.det(r),torch.ones(1,6),atol=1e-5))
  batch=dict(x=torch.randn(1,6,3),r=torch.eye(3).expand(1,6,3,3),atoms=torch.randn(1,6,4,3),
      observed=torch.ones(1,6,4,dtype=torch.bool),frame_mask=mask,links=mask)
  total,_=direct_loss(x,r,batch);total.backward()
  self.assertTrue(torch.isfinite(total))
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
  xx,rr=model(embedding,aa,mask)
  self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(r,rr))

 def test_balanced_multi_sequence_schedule_is_resumable(self):
  from scripts.train_direct_overfit import schedule_index
  for epoch in range(3):
   ids=[schedule_index(epoch*15+i+1,15,42) for i in range(15)]
   self.assertEqual(sorted(ids),list(range(15)))
  uninterrupted=[schedule_index(i,15,42) for i in range(1,46)]
  resumed=[schedule_index(i,15,42) for i in range(23,46)]
  self.assertEqual(uninterrupted[22:],resumed)

if __name__=='__main__':unittest.main()
