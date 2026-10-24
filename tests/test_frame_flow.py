from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from foldflow.geometry import so3_exp,global_vector
from foldflow.model import FrameFlow
from foldflow.training import flow_sample,loss


class FlowTests(unittest.TestCase):
 def batch(self):
  b,l=2,6
  return dict(embedding=torch.randn(b,l,320),aa=torch.randint(20,(b,l)),x=torch.randn(b,l,3),r=torch.eye(3).expand(b,l,3,3).clone(),
      atoms=torch.randn(b,l,4,3),observed=torch.ones(b,l,4,dtype=torch.bool),frame_mask=torch.ones(b,l,dtype=torch.bool),
      links=torch.ones(b,l,dtype=torch.bool),mask=torch.ones(b,l,dtype=torch.bool))
 def test_so3_zero_and_finite_grad(self):
  w=torch.zeros(2,3,requires_grad=True);r=so3_exp(w)
  self.assertTrue(torch.allclose(r,torch.eye(3).expand(2,3,3)))
  r.sum().backward();self.assertTrue(torch.isfinite(w.grad).all())
 def test_equivariance_padding_and_backward(self):
  torch.manual_seed(7);m=FrameFlow(hidden=32,pair_dim=16,pair_layers=1,geometry_layers=1)
  # Enable nonzero outputs so equivariance check is substantive.
  torch.nn.init.normal_(m.velocity[-1].weight,std=.02)
  b=self.batch();b['mask'][1,-2:]=False
  r=so3_exp(torch.randn(2,6,3));x=b['x'];t=torch.tensor([.2,.8])
  v,w=m(b['embedding'],b['aa'],x,r,t,b['mask'])
  q=so3_exp(torch.tensor([.4,-.2,.1]));shift=torch.tensor([1.,2.,-1.])
  vv,ww=m(b['embedding'],b['aa'],x@q.T+shift,q@r,t,b['mask'])
  self.assertTrue(torch.allclose(v,vv,atol=1e-5));self.assertTrue(torch.allclose(w,ww,atol=1e-5))
  self.assertTrue(torch.equal(v[1,-2:],torch.zeros_like(v[1,-2:])))
  self.assertTrue(torch.allclose((global_vector(r,v)*b['mask'][...,None]).sum(1),torch.zeros(2,3),atol=1e-5))
  state=flow_sample(b,np.random.default_rng(1));total,_=loss(m,b,state);total.backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
