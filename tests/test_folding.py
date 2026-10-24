from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foldflow.folding import FoldingModel,TriangleMultiplication,invariant_loss,augment_target
from foldflow.geometry import so3_exp,backbone_with_oxygen

class FoldingTests(unittest.TestCase):
 def batch(self):
  torch.manual_seed(9)
  x=torch.randn(2,6,3);r=so3_exp(torch.randn(2,6,3))
  mask=torch.ones(2,6,dtype=torch.bool);mask[1,-2:]=False
  return dict(x=x,r=r,atoms=backbone_with_oxygen(x,r),observed=torch.ones(2,6,4,dtype=torch.bool),
      links=mask.clone(),frame_mask=mask,mask=mask)
 def test_independent_target_and_prediction_rigid_invariance(self):
  b=self.batch();x=(b['x']+.1*torch.randn_like(b['x'])).requires_grad_()
  r=(b['r']@so3_exp(torch.randn(2,6,3)*.1)).requires_grad_()
  a,_=invariant_loss(x,r,b)
  rotated=augment_target(b,123)
  c,_=invariant_loss(x,r,rotated)
  self.assertTrue(torch.allclose(a,c,atol=2e-6))
  g1=torch.autograd.grad(a,(x,r),retain_graph=True)
  g2=torch.autograd.grad(c,(x,r),retain_graph=True)
  self.assertTrue(all(torch.allclose(u,v,atol=3e-6) for u,v in zip(g1,g2)))
  pred=augment_target(dict(b,x=x.detach(),r=r,atoms=backbone_with_oxygen(x.detach(),r)),321)
  d,_=invariant_loss(pred['x'],pred['r'],b)
  self.assertTrue(torch.allclose(a,d,atol=2e-6))
 def test_640d_recycling_backward(self):
  b=self.batch();m=FoldingModel(esm_dim=640,hidden=32,pair_dim=16,pair_layers=1,geometry_layers=1,recycles=2)
  x,r=m(torch.randn(2,6,640),torch.randint(20,(2,6)),b['mask'])
  total,_=invariant_loss(x,r,augment_target(b,22));total.backward()
  self.assertTrue(torch.isfinite(total))
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
  self.assertTrue(torch.equal(x[1,-2:],torch.zeros(2,3)))
 def test_triangle_mask(self):
  torch.manual_seed(3);t=TriangleMultiplication(8,4);torch.nn.init.normal_(t.out.weight,std=.1)
  z=torch.randn(1,5,5,8);mask=torch.tensor([[True,True,True,False,False]])
  a=t(z,mask);modified=z.clone();modified[:,3:]=100;modified[:,:,3:]=100
  c=t(modified,mask)
  self.assertTrue(torch.allclose(a[:,:3,:3],c[:,:3,:3],atol=1e-6))
  self.assertTrue(torch.equal(c[:,3:],torch.zeros_like(c[:,3:])))

if __name__=='__main__':unittest.main()
