"""Train-only flow diagnostics. Reads checkpoints/cache; never trains or picks best noise."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from foldflow.data import collate
from foldflow.geometry import so3_exp,local_vector,global_vector,backbone_with_oxygen
from foldflow.model import FrameFlow
from foldflow.training import flow_sample,generate,structure_metrics


def angular_error_deg(a,b,mask):
    # atan2 is stable near identity, unlike acos(trace) in float32.
    rel=a.transpose(-1,-2)@b
    vee=torch.stack((rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]),-1)*.5
    sine=vee.norm(dim=-1)
    cosine=((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
    return float((torch.atan2(sine,cosine)[mask]*180/torch.pi).max().cpu())


def source_path(batch,seed):
    state=flow_sample(batch,np.random.default_rng(seed))
    t=state['t'][:,None,None]
    x0=(state['x']-t*state['target_x'])/(1-t)
    r0=state['r']@so3_exp(-t*state['omega'])
    return x0,r0,state['target_x'],state['target_r'],state['omega']


def roundtrip(batch):
    x,r,atoms=batch['x'],batch['r'],batch['atoms']
    frame=batch['frame_mask']
    delta=atoms-x[:,:,None,:]
    local=torch.einsum('blij,blai->blaj',r,delta)
    reconstructed=x[:,:,None,:]+torch.einsum('blij,blaj->blai',r,local)
    amask=batch['observed']&frame[...,None]
    exact=(reconstructed-atoms).norm(dim=-1)[amask]*10
    ideal=backbone_with_oxygen(x,r)
    ideal_mask=amask.clone();ideal_mask[:,:,3]&=batch['links']
    ideal_error=(ideal-atoms).norm(dim=-1)
    per_atom={name:float(torch.sqrt((ideal_error[:,:,i][ideal_mask[:,:,i]]**2).mean()).cpu())*10
        if ideal_mask[:,:,i].any() else None for i,name in enumerate(('N','CA','C','O'))}
    orth=(r.transpose(-1,-2)@r-torch.eye(3,device=r.device)).abs()[frame].max()
    determinants=torch.linalg.det(r.cpu())[frame.cpu()]
    # Positive C x and N y confirm the intended frame convention.
    signs=bool((local[:,:,2,0][frame]>0).all() and (local[:,:,0,1][frame]>0).all())
    plane_error=max(float(local[:,:,2,1:][frame].abs().max().cpu()),float(local[:,:,0,2][frame].abs().max().cpu()))*10
    passed=bool(exact.max()<1e-3 and orth<1e-4 and ((determinants-1).abs()<1e-4).all() and signs and plane_error<1e-3)
    return dict(passed=passed,max_exact_roundtrip_error_A=float(exact.max().cpu()),
        max_orthogonality_error=float(orth.cpu()),min_det=float(determinants.min()),
        frame_axis_signs_correct=signs,max_frame_plane_error_A=plane_error,ideal_backbone_rms_error_A=per_atom,
        note='Exact roundtrip uses measured local coordinates; ideal N/C/O reconstruction is approximate and reported separately.')


def oracle(batch,steps=128,seed=1):
    x0,r0,x1,r1,omega=source_path(batch,seed)
    x,r=x0.clone(),r0.clone()
    delta=x1-x0
    for _ in range(steps):
        body=local_vector(r,delta)
        x=x+global_vector(r,body)/steps
        r=r@so3_exp(omega/steps)
    mask=batch['frame_mask']
    pos=float(((x-x1).norm(dim=-1)[mask]*10).max().cpu())
    angle=angular_error_deg(r,r1,mask)
    return dict(passed=pos<.01 and angle<.02,steps=steps,
        max_position_endpoint_error_A=pos,max_rotation_endpoint_error_deg=angle,
        note='Uses known global path displacement expressed in each current body frame. Target is used only for this oracle sanity test.')


@torch.no_grad()
def time_profile(model,batch,seed,trials):
    times=[0.,.01,.02,.1,.25,.5,.75,.9,.98,.99,1.]
    mask=batch['frame_mask']
    encoded=model.encode(batch['embedding'],batch['aa'],batch['mask'])
    values={t:[] for t in times}
    for trial in range(trials):
        x0,r0,x1,r1,omega=source_path(batch,seed+trial)
        delta=x1-x0
        for t in times:
            time=torch.full((len(x0),),t,device=x0.device)
            x=(1-t)*x0+t*x1;r=r0@so3_exp(t*omega)
            v,w=model(batch['embedding'],batch['aa'],x,r,time,batch['mask'],encoded)
            true_v=local_vector(r,delta)
            v_rmse=float(torch.sqrt((v-true_v).square()[mask].mean()).cpu())*10
            w_rmse=float(torch.sqrt((w-omega).square()[mask].mean()).cpu())
            estimated=x+(1-t)*global_vector(r,v)
            metrics=structure_metrics(estimated[0],x1[0],mask[0])
            values[t].append(dict(translation_component_rmse_A_per_time=v_rmse,
                rotation_component_rmse_rad_per_time=w_rmse,
                zero_velocity_baseline_rmse_A_per_time=float(torch.sqrt(true_v.square()[mask].mean()).cpu())*10,
                estimated_endpoint_ca_rmsd_A=metrics['ca_rmsd_A'],estimated_endpoint_ca_lddt=metrics['ca_lddt']))
    return [dict(t=t,inside_training_time_range=.02<=t<=.98,
        **{k:float(np.mean([r[k] for r in vv])) for k in vv[0]}) for t,vv in values.items()]


@torch.no_grad()
def ode_comparison(model,batch,steps,seed):
    results=[]
    for n in steps:
        x,r=generate(model,batch,n,seed)
        result=structure_metrics(x[0],batch['x'][0],batch['frame_mask'][0])
        result.update(steps=n,seed=seed)
        results.append(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/mps_pilot/cache1.pt'))
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/mps_overfit1/last.pt'))
    p.add_argument('--out',type=Path,default=Path('outputs/diagnostics_overfit1'))
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--ode-steps',type=int,nargs='+',default=[32,128,256])
    p.add_argument('--trials',type=int,default=4)
    args=p.parse_args()
    if args.trials<1 or not args.ode_steps or min(args.ode_steps)<1:p.error('positive trials and steps required')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if hashlib.sha256(args.cache.read_bytes()).hexdigest()!=state['cache_sha256']:raise ValueError('checkpoint/cache mismatch')
    if any(r['split']!='train' for r in cache['records']):raise ValueError('train-only diagnostics required')
    model=FrameFlow(**state['config']).to(args.device);model.load_state_dict(state['model']);model.eval()
    config_path=args.checkpoint.parent/'config.json'
    train_config=json.loads(config_path.read_text()) if config_path.exists() else {}
    seed=train_config.get('seed',20261024)
    report=dict(checkpoint=str(args.checkpoint),checkpoint_step=state['step'],device=args.device,
        scope='Train-only diagnostics; no retraining and no generalization claim.',records=[])
    args.out.mkdir(parents=True,exist_ok=True)
    for i,record in enumerate(cache['records']):
        batch=collate([record],args.device)
        result=dict(id=record['id'],length=len(record['sequence']),
            roundtrip=roundtrip(batch),oracle=oracle(batch,128,seed+i),
            time_profile=time_profile(model,batch,seed+i,args.trials),
            ode_comparison=ode_comparison(model,batch,args.ode_steps,seed+10000+i))
        report['records'].append(result)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    report['sanity_passed']=all(r['roundtrip']['passed'] and r['oracle']['passed'] for r in report['records'])
    report['interpretation']=[
        'Large exact roundtrip/oracle errors suggest geometry/path conventions or numerical implementation problems.',
        'Ideal backbone error need not be zero: ideal bond lengths/angles and oxygen construction are approximations.',
        'Better 128/256-step rollout than 32 suggests solver discretization contributes; no improvement does not rule out vector-field problems.',
        'Time-profile evaluates states on the true path. Rollouts may leave that path, so low teacher-path error does not guarantee accurate generation.',
        't=0/1 are outside sampled training range. At t=1 endpoint position is correct by construction regardless of predicted velocity.',
        'All ODE budgets use the same initial noise seed as training evaluation; no best-of-seed selection.']
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Sanity passed: {report["sanity_passed"]}; report: {args.out / "report.json"}')
    if not report['sanity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
