"""Column-vector residue frames. Positions use nm; exported atoms use Angstrom."""
import torch


def so3_exp(w):
    x,y,z=w.unbind(-1)
    zero=torch.zeros_like(x)
    k=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*w.shape[:-1],3,3)
    theta=w.norm(dim=-1)
    a=torch.sinc(theta/torch.pi)[...,None,None]
    b=(.5*torch.sinc(theta/(2*torch.pi)).square())[...,None,None]
    eye=torch.eye(3,device=w.device,dtype=w.dtype)
    return eye+a*k+b*(k@k)


def local_vector(r,v):return torch.einsum('...ji,...j->...i',r,v)
def global_vector(r,v):return torch.einsum('...ij,...j->...i',r,v)


def pair_geometry(x,r):
    delta=x[:,None,:,:]-x[:,:,None,:]
    local=torch.einsum('bijd,bidk->bijk',delta,r)
    relative=r[:,:,None].transpose(-1,-2)@r[:,None]
    return torch.cat((local,relative.flatten(-2),delta.norm(dim=-1,keepdim=True)),-1)


def backbone(x,r):
    # Ideal N/CA/C offsets in frame defined by C-CA and the N-CA plane, nm.
    offsets=x.new_tensor([[-.0525,.1363,0.],[0.,0.,0.],[.1526,0.,0.]])
    return x[...,None,:]+torch.einsum('blij,aj->blai',r,offsets)


def backbone_with_oxygen(x,r):
    atoms=backbone(x,r)
    ca,c=atoms[:,:,1],atoms[:,:,2]
    n_next=torch.cat((atoms[:,1:,0],atoms[:,-1:,0]),1)
    away=(ca-c)/(ca-c).norm(dim=-1,keepdim=True).clamp_min(1e-6)
    toward=(n_next-c)/(n_next-c).norm(dim=-1,keepdim=True).clamp_min(1e-6)
    direction=-(away+toward)
    direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    oxygen=c+.123*direction
    return torch.cat((atoms,oxygen[:,:,None,:]),2)
