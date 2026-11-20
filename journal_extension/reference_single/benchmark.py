"""Reference single pendulum: frozen demonstrations, branch-aware chains.

Passive waiting is computed on the exact elliptic-function orbit, with
numerically localized intersections of the orbit and every candidate ball.
This is numerical continuous event detection, not a validated enclosure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ellipj, ellipk, ellipkinc
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from event_detection.ball_events import HermiteArc, verified_first_entry, unwrap_relative

ROOT = Path(__file__).resolve().parent
REF = ROOT / 'reference_inputs'
A, I = 19.62, 4.0
ES, RT, V0, HX = 39.24, 0.1, 0.001, 168.0
ELO, EHI = A * (1 + np.cos(RT)), ES + RT**2 / (2 * I)
OMEGA = np.sqrt(A / I)
L = A
# Exact supremum of ||grad H|| on H<=HX, obtained by maximizing in cos(q).
LV = np.sqrt(A*A + (HX-A)/2 + 1/16)
ENDPOINT_BUDGET = 1e-8  # conservative numerical reserve, not interval validation


@njit(cache=True)
def wrap(q):
    return (q + np.pi) % (2*np.pi) - np.pi


@njit(cache=True)
def energy(x):
    return x[1]**2/(2*I) + A*(1-np.cos(x[0]))


@njit(cache=True)
def score(x):
    """Signed Y-graph distance; this symmetric target covers all three arms."""
    e = energy(x)
    if e < ELO:
        return ELO-e
    if e > EHI:
        return e-EHI
    if e <= ES:
        return -min(e-ELO, ES-e+(EHI-ES))
    return -min(EHI-e, e-ES+min(ES-ELO, EHI-ES))


@njit(cache=True)
def distance(x, y):
    return np.sqrt(wrap(x[0]-y[0])**2+(x[1]-y[1])**2)


@njit(cache=True)
def target_gap(x):
    return np.sqrt(wrap(x[0]-np.pi)**2+x[1]**2)-RT


@njit(cache=True)
def step(x, u, h):
    q,p = x[0],x[1]
    a=p/I; b=-A*np.sin(q)+u
    c=(p+h*b/2)/I; d=-A*np.sin(q+h*a/2)+u
    e=(p+h*d/2)/I; f=-A*np.sin(q+h*c/2)+u
    g=(p+h*f)/I; k=-A*np.sin(q+h*e)+u
    return np.array([wrap(q+h*(a+2*c+2*e+g)/6),p+h*(b+2*d+2*f+k)/6])


@njit(cache=True)
def replay(x, control, lead, dt, subdivisions=4):
    y=x.copy(); maxe=energy(y); maxp=abs(y[1])
    for j in range(len(control)):
        h=(lead if j==0 else dt)/subdivisions
        for _ in range(subdivisions):
            y=step(y,control[j],h)
            maxe=max(maxe,energy(y)); maxp=max(maxp,abs(y[1]))
    return y,maxe,maxp


@njit(cache=True)
def replay_all(centers, flat, offsets, leads, dt, subdivisions=4):
    ends=np.empty_like(centers); maxe=np.empty(len(centers)); maxp=maxe.copy()
    for j in range(len(centers)):
        ends[j],maxe[j],maxp[j]=replay(centers[j],flat[offsets[j]:offsets[j+1]],leads[j],dt,subdivisions)
    return ends,maxe,maxp


def orbit_parameters(x):
    e=float(energy(x)); q=float(wrap(x[0])); p=float(x[1])
    if not 0 < e or abs(e-ES)<1e-12:
        raise ValueError('Equilibrium or numerical separatrix requires separate handling')
    if e<ES:
        k=np.sqrt(e/ES); m=k*k; K=ellipk(m)
        alpha=np.arctan2(np.sin(q/2)/k,p/(2*I*OMEGA*k))
        z=float(ellipkinc(alpha,m)%(4*K))
        return e,m,4*K,OMEGA,z,0
    k=np.sqrt(ES/e); m=k*k; K=ellipk(m); direction=1 if p>0 else -1
    z=float((direction*ellipkinc(q/2,m))%(2*K))
    return e,m,2*K,OMEGA/k,z,direction


def orbit_state(params, phase):
    e,m,period,rate,z,direction=params
    if direction==0:
        sn,cn,dn,am=ellipj(np.asarray(phase)%period,m)
        return np.stack([2*np.arcsin(np.clip(np.sqrt(m)*sn,-1,1)),2*I*OMEGA*np.sqrt(m)*cn],axis=-1)
    sn,cn,dn,am=ellipj(direction*(np.asarray(phase)%period),m)
    q=(2*am+np.pi)%(2*np.pi)-np.pi
    return np.stack([q,direction*2*I*OMEGA/np.sqrt(m)*dn],axis=-1)


def passive(x,t):
    par=orbit_parameters(x)
    return orbit_state(par,par[4]+par[3]*t)


@njit(cache=True)
def boundary_energy(q,p,r,theta):
    return (p+r*np.sin(theta))**2/(2*I)+A*(1-np.cos(q+r*np.cos(theta)))


@njit(cache=True)
def boundary_derivative(q,p,r,theta):
    return -A*np.sin(q+r*np.cos(theta))*np.sin(theta)+(p+r*np.sin(theta))*np.cos(theta)/I


@njit(cache=True)
def boundary_partitions(centers,radii,grid=64):
    """All numerically bracketed stationary angles; audited against finer grids."""
    n=len(radii); angles=np.full((n,10),np.nan); values=angles.copy(); counts=np.zeros(n,np.int64)
    for i in range(n):
        q,p=centers[i];r=radii[i];roots=np.empty(8);count=0
        prev=boundary_derivative(q,p,r,0.)
        for k in range(grid):
            lo=k*2*np.pi/grid;hi=(k+1)*2*np.pi/grid
            nex=boundary_derivative(q,p,r,hi)
            if prev*nex<0:
                a,b=lo,hi;fa=prev
                for _ in range(48):
                    mid=(a+b)/2;fm=boundary_derivative(q,p,r,mid)
                    if fa*fm<=0:b=mid
                    else:a=mid;fa=fm
                root=(a+b)/2
                if root>1e-12 and root<2*np.pi-1e-12 and (count==0 or root-roots[count-1]>1e-10):
                    if count>=8:raise ValueError('Boundary partition capacity exceeded')
                    roots[count]=root;count+=1
            elif abs(prev)<1e-13 and lo>1e-12 and (count==0 or lo-roots[count-1]>1e-10):
                if count>=8:raise ValueError('Boundary partition capacity exceeded')
                roots[count]=lo;count+=1
            prev=nex
        angles[i,0]=0
        for j in range(count):angles[i,j+1]=roots[j]
        angles[i,count+1]=2*np.pi
        counts[i]=count+2
        for j in range(count+2):values[i,j]=boundary_energy(q,p,r,angles[i,j])
    return angles,values,counts


@njit(cache=True)
def intersection_points(e,centers,radii,angles,values,counts,ids):
    points=np.empty((len(ids)*10,2)); labels=np.empty(len(ids)*10,np.int64);n=0
    for idx in ids:
        q,p=centers[idx];r=radii[idx]
        for k in range(counts[idx]-1):
            lo,hi=angles[idx,k],angles[idx,k+1]
            flo,fhi=values[idx,k]-e,values[idx,k+1]-e
            if flo*fhi>0:continue
            if abs(flo)<1e-13:root=lo
            elif abs(fhi)<1e-13:root=hi
            else:
                for _ in range(45):
                    mid=(lo+hi)/2;fm=boundary_energy(q,p,r,mid)-e
                    if flo*fm<=0:hi=mid
                    else:lo=mid;flo=fm
                root=(lo+hi)/2
            points[n,0]=wrap(q+r*np.cos(root));points[n,1]=p+r*np.sin(root);labels[n]=idx;n+=1
    return points[:n],labels[:n]


class Library:
    def __init__(self, payload, M, transported):
        self.full=payload
        mask=(payload['demo_ids']<M)&((payload['is_transport']==0)|transported)
        self.ids=np.flatnonzero(mask)
        self.centers=payload['centers'][mask];self.radii=payload['radii'][mask]
        self.angles=payload['angles'][mask];self.values=payload['values'][mask];self.counts=payload['counts'][mask]
        self.emin=np.nanmin(self.values,axis=1);self.emax=np.nanmax(self.values,axis=1)
        box=self.centers.copy();box[:,0]%=2*np.pi
        self.tree=cKDTree(box,boxsize=[2*np.pi,0.])
        self.rmax=float(self.radii.max())
        # Energy-image bounds are conservative to floating-point resolution.
        self.ec=(self.emin+self.emax)/2;self.ew=(self.emax-self.emin)/2
        levels=np.minimum(np.floor(-np.log(np.maximum(self.ew,1e-15)/self.ew.max())/np.log(8)).astype(int),7)
        self.energy_indices=[]
        for level in np.unique(levels):
            idx=np.flatnonzero(levels==level)
            self.energy_indices.append((idx,float(self.ew[idx].max()),cKDTree(self.ec[idx,None])))

    def inside(self,x):
        box=np.array([x[0]%(2*np.pi),x[1]])
        ids=np.asarray(self.tree.query_ball_point(box,self.rmax),dtype=int)
        if not len(ids):return -1
        d=self.centers[ids]-x;d[:,0]=(d[:,0]+np.pi)%(2*np.pi)-np.pi
        rho=np.linalg.norm(d,axis=1)/self.radii[ids]
        k=int(np.argmin(rho))
        return int(ids[k]) if rho[k]<=1 else -1

    def candidates(self,e,direction):
        hits=[]
        for ids,width,tree in self.energy_indices:
            near=np.asarray(tree.query_ball_point([e],width+2e-12),dtype=int)
            if len(near):hits.append(ids[near])
        if not hits:return np.empty(0,dtype=int)
        ids=np.concatenate(hits)
        keep=(self.emin[ids]<=e+2e-12)&(self.emax[ids]>=e-2e-12)
        if direction:keep&=(direction*self.centers[ids,1]+self.radii[ids]>=0)
        return ids[keep]

    def first_access(self,x):
        idx=self.inside(x)
        if idx>=0:return 0.,x.copy(),idx,0
        par=orbit_parameters(x);e,m,period,rate,z,direction=par
        ids=self.candidates(e,direction)
        if not len(ids):return np.inf,x.copy(),-1,0
        points,labels=intersection_points(e,self.centers,self.radii,self.angles,self.values,self.counts,ids)
        if direction:
            keep=direction*points[:,1]>0;points=points[keep];labels=labels[keep]
        if not len(labels):return np.inf,x.copy(),-1,len(ids)
        if direction:
            phases=(direction*ellipkinc(points[:,0]/2,m))%period
        else:
            alpha=np.arctan2(np.sin(points[:,0]/2)/np.sqrt(m),points[:,1]/(2*I*OMEGA*np.sqrt(m)))
            phases=ellipkinc(alpha,m)%period
        intervals=[]
        for idx in np.unique(labels):
            ps=np.unique(phases[labels==idx]);r=self.radii[idx]
            for k,a in enumerate(ps):
                b=ps[(k+1)%len(ps)]+(period if k+1==len(ps) else 0)
                mid=(a+b)/2
                if distance(orbit_state(par,mid),self.centers[idx])<=r:
                    entry=(a-z)%period/rate
                    intervals.append((entry,entry+(b-a)/(2*rate),int(idx)))
                elif distance(orbit_state(par,a),self.centers[idx])<=r:
                    # Exact numerical boundary membership also covers tangency.
                    entry=(a-z)%period/rate
                    intervals.append((entry,entry,int(idx)))
        for entry,inside,idx in sorted(intervals):
            r=self.radii[idx];low=entry;high=inside
            if distance(orbit_state(par,z+rate*low),self.centers[idx])<=r:
                # Search back a small amount to remove phase-conversion roundoff.
                low=max(0,entry-max(1e-12,1e-4*r/max(1,np.linalg.norm(x))))
            if distance(orbit_state(par,z+rate*high),self.centers[idx])>r:continue
            if distance(orbit_state(par,z+rate*low),self.centers[idx])>r:
                for _ in range(48):
                    mid=(low+high)/2
                    if distance(orbit_state(par,z+rate*mid),self.centers[idx])<=r:high=mid
                    else:low=mid
            else:high=low
            y=orbit_state(par,z+rate*high)
            if distance(y,self.centers[idx])>r:raise RuntimeError('Orbit entry lost membership')
            selected=self.inside(y)
            return high,y,selected if selected>=0 else idx,len(ids)
        return np.inf,x.copy(),-1,len(ids)


def prepare():
    start=time.perf_counter();a=np.load(REF/'assignments.npz');d=np.load(REF/'demonstrations.npz')
    centers=a['centers'].copy();radii=a['radii'].copy();flat=a['control_flat'][:,0];steps=a['control_steps'];leads=a['leads'];dt=float(a['dt'])
    offsets=np.r_[0,np.cumsum(steps)]
    ends,maxe,maxp=replay_all(centers,flat,offsets,leads,dt,4)
    ends2,_,_=replay_all(centers,flat,offsets,leads,dt,8)
    reintegration=np.array([distance(x,y) for x,y in zip(ends,ends2)])
    durations=leads+(steps-1)*dt
    margin=np.array([score(x)-score(y) for x,y in zip(centers,ends)])-V0*durations-ENDPOINT_BUDGET
    growth=np.exp(L*durations)
    rprogress=margin/(LV*(1+growth))
    # Bound nominal energy between substeps using |Hdot|<=|p| |u|/I.
    nominal_envelope=maxe+(maxp+(A+20)*dt/4)/I*20*dt/4
    rsafe=(HX-nominal_envelope)/(LV*growth)
    certified=np.minimum(np.minimum(radii,rprogress),rsafe)
    keep=certified>1e-10
    original_ids=np.flatnonzero(keep)
    original_count=len(original_ids)
    controls=[flat[offsets[j]:offsets[j+1]].copy() for j in original_ids]
    out_centers=list(centers[keep]);out_radii=list(certified[keep]);out_demo=list(a['demo_ids'][keep]);out_leads=list(leads[keep]);out_anchor=list(a['anchor_times'][keep]);out_parent=list(original_ids);out_shift=[0.]*original_count;out_trans=[0]*original_count
    transport_attempts=[]
    # Frozen construction: all retained original anchors, four fixed shifts.
    # This is chosen using construction coverage alone, before test rollouts.
    for demo in range(5):
        source=np.flatnonzero(keep&(a['demo_ids']==demo))
        for parent in source:
            u=flat[offsets[parent]:offsets[parent+1]]
            for shift in [-0.5,-0.25,0.25,0.5]:
                try:z=passive(centers[parent],shift)
                except ValueError:continue
                y,me,mp=replay(z,u,leads[parent],dt,4)
                y2,_,_=replay(z,u,leads[parent],dt,8)
                err=float(distance(y,y2));tau=durations[parent]
                eta=float(score(z)-score(y)-V0*tau-ENDPOINT_BUDGET)
                rr=min(eta/(LV*(1+np.exp(L*tau))), (HX-me-(mp+(A+20)*dt/4)/I*20*dt/4)/(LV*np.exp(L*tau)))
                accepted=rr>1e-10 and LV*err<ENDPOINT_BUDGET/10
                transport_attempts.append(dict(parent=int(parent),demo=demo,shift=shift,radius=float(rr),accepted=bool(accepted),reintegration_error=err))
                if accepted:
                    out_centers.append(z);out_radii.append(rr);out_demo.append(demo);out_leads.append(leads[parent]);out_anchor.append(a['anchor_times'][parent]);out_parent.append(parent);out_shift.append(shift);out_trans.append(1);controls.append(u.copy())
    out_centers=np.asarray(out_centers);out_radii=np.asarray(out_radii)
    angles,values,counts=boundary_partitions(out_centers,out_radii,64)
    angles2,values2,counts2=boundary_partitions(out_centers,out_radii,256)
    extrema_ok=bool(np.array_equal(counts,counts2) and np.nanmax(np.abs(angles-angles2))<1e-9)
    if not extrema_ok:raise RuntimeError('Boundary extrema grid convergence failed')
    payload=dict(centers=out_centers,radii=out_radii,control_flat=flat,offsets=offsets,leads=np.array(out_leads),dt=np.array(dt),demo_ids=np.array(out_demo),anchor_times=np.array(out_anchor),parent=np.array(out_parent),shift=np.array(out_shift),is_transport=np.array(out_trans),angles=angles,values=values,counts=counts)
    np.savez_compressed(ROOT/'library.npz',**payload)
    initial=np.load(REF/'rollouts.npz')['initial_states']
    rng=np.random.default_rng(2026);batch=rng.uniform([-np.pi,-np.sqrt(8*160)],[np.pi,np.sqrt(8*160)],size=(2000,2));sample=batch[np.array([energy(x)<=160 for x in batch])][:500]
    if not np.array_equal(initial,sample):raise RuntimeError('Reference initial sample mismatch')
    np.savez(ROOT/'initial_states.npz',states=initial)
    demo_audit=[]
    for j in range(5):
        x=d[f'states_{j}'][0];u=d[f'controls_{j}'][:,0]
        y,me,mp=replay(x,u,float(d[f'dt_{j}']),float(d[f'dt_{j}']),8)
        demo_audit.append(dict(name=str(d['names'][j]),endpoint_error=float(distance(y,d[f'states_{j}'][-1])),target_gap=float(target_gap(y)),energy=float(energy(y)),max_input=float(abs(u).max())))
    record=dict(reference_commit='7974f19e94975b96e40022a4878d4f45893b4f39',source_hashes={f:hashlib.sha256((REF/f).read_bytes()).hexdigest() for f in ['demonstrations.npz','assignments.npz','rollouts.npz']},original_assignments=len(centers),retained_original_assignments=original_count,transport_attempts=len(transport_attempts),transport_accepted=len(out_centers)-original_count,counts_by_demo=[dict(demo=j,original=int(np.sum((payload['demo_ids']==j)&(payload['is_transport']==0))),transported=int(np.sum((payload['demo_ids']==j)&(payload['is_transport']==1)))) for j in range(5)],LV=LV,L=L,HX=HX,endpoint_numerical_reserve=ENDPOINT_BUDGET,max_4vs8_substep_error=float(reintegration.max()),extrema_64vs256_agree=extrema_ok,initial_sample_exact_match=True,demo_audit=demo_audit,seconds=time.perf_counter()-start,numerical_status='Analytic perturbation/safety bounds evaluated with converged numerical endpoints; reserve is not a validated integration enclosure.')
    (ROOT/'construction.json').write_text(json.dumps(record,indent=2))
    (ROOT/'transport_attempts.json').write_text(json.dumps(transport_attempts,indent=2))
    print(json.dumps(record,indent=2),flush=True)


@njit(cache=True)
def primitive_trace(x,control,lead,dt,subdivisions):
    states=np.empty((len(control)*subdivisions+1,2));hs=np.empty(len(control)*subdivisions);us=hs.copy()
    states[0]=x;k=0
    for j in range(len(control)):
        h=(lead if j==0 else dt)/subdivisions
        for _ in range(subdivisions):
            states[k+1]=step(states[k],control[j],h);hs[k]=h;us[k]=control[j];k+=1
    return states,hs,us


def execute(x,control,lead,dt,subdivisions,remaining):
    states,hs,us=primitive_trace(x,control,lead,dt,subdivisions)
    ts=np.r_[0,np.cumsum(hs)]
    # Truncate a signal if the fixed evaluation horizon ends inside it.
    if ts[-1]>remaining:
        k=int(np.searchsorted(ts,remaining,side='right')-1)
        if remaining-ts[k]>1e-14:
            h=remaining-ts[k]
            states=np.vstack([states[:k+1],step(states[k],us[k],h)])
            hs=np.r_[hs[:k],h];us=us[:k+1]
        else:states=states[:k+1];hs=hs[:k];us=us[:k]
        ts=np.r_[0,np.cumsum(hs)]
    if not len(hs):return x.copy(),0.,0.,False,float(energy(x)),0
    fa=np.column_stack([states[:-1,1]/I,-A*np.sin(states[:-1,0])+us])
    fb=np.column_stack([states[1:,1]/I,-A*np.sin(states[1:,0])+us])
    arc=HermiteArc(states[:-1],states[1:],fa,fb,hs,angle_indices=(0,))
    centers=unwrap_relative(np.array([np.pi,0.]),arc.xa,angle_indices=(0,))
    def evaluate(row,s):
        return unwrap_relative(step(states[row],us[row],hs[row]*s),arc.xa[row],angle_indices=(0,))
    # RK4 subdivisions make the padding tiny relative to the physical target;
    # it is a numerical error allowance, independently checked by refinement.
    events=verified_first_entry(arc,np.arange(len(hs)),centers,np.full(len(hs),RT),evaluate,num_chords=4,interpolation_padding=1e-10)
    hit=np.flatnonzero(np.isfinite(events.s))
    if len(hit):
        j=int(hit[0]);duration=ts[j]+hs[j]*events.s[j];y=events.states[j].copy();y[0]=wrap(y[0])
        effort=float(np.dot(us[:j]**2,hs[:j])+us[j]**2*hs[j]*events.s[j])
        return y,duration,effort,True,max(float(np.max([energy(z) for z in states[:j+1]])),float(energy(y))),int(events.diagnostics['flow_evaluations'])
    return states[-1],float(ts[-1]),float(np.dot(us**2,hs)),False,float(np.max([energy(z) for z in states])),int(events.diagnostics['flow_evaluations'])


def target_library():
    c=np.array([[np.pi,0.]]);r=np.array([RT]);angles,values,counts=boundary_partitions(c,r)
    return Library(dict(centers=c,radii=r,angles=angles,values=values,counts=counts,demo_ids=np.array([0]),is_transport=np.array([0])),1,False)


def rollout(initial,lib,tlib,horizon=2000.,subdivisions=4,save_actions=False):
    x=initial.copy();t=active=effort=0.;actions=[];waits=[];number=0;nt=0;maxe=float(energy(x));reason='horizon';hit=False;queries=0;eventcalls=0;first_action=None
    if target_gap(x)<=0:return dict(success=True,time=0.,active=0.,idle=0.,effort=0.,actions=0,transport_actions=0,max_energy=maxe,reason='initial_target',final=x.tolist(),support_violation=0.),actions
    while t<horizon:
        terminal=score(x)<=0
        chosen=tlib if terminal else lib
        waiting,y,idx,nc=chosen.first_access(x);queries+=nc
        if not np.isfinite(waiting):
            reason='no_target_orbit_intersection' if terminal else 'no_support_orbit_intersection';break
        if t+waiting>horizon:reason='horizon_during_wait';break
        t+=waiting;waits.append(waiting);x=y
        if target_gap(x)<=0:
            hit=True;reason='passive_target';break
        if terminal:raise RuntimeError('Terminal access returned a state outside target')
        # Chain rule selects minimum normalized distance among all balls now
        # containing the actual state. The event's boundary label is a fallback
        # only when it itself passes exact numerical membership.
        selected=lib.inside(x)
        if selected>=0:idx=selected
        if distance(x,lib.centers[idx])>lib.radii[idx]:raise RuntimeError('Outside-support action')
        if first_action is None:first_action=float(t)
        source=int(lib.ids[idx]);p=lib.full;offsets=p['offsets'];parent=int(p['parent'][source]);u=p['control_flat'][offsets[parent]:offsets[parent+1]]
        before=x.copy();vbefore=float(score(x));rho=float(distance(x,lib.centers[idx])/lib.radii[idx]);duration=float(p['leads'][source]+(len(u)-1)*float(p['dt']))
        x,elapsed,cost,hit,peak,calls=execute(x,u,float(p['leads'][source]),float(p['dt']),subdivisions,horizon-t)
        t+=elapsed;active+=elapsed;effort+=cost;number+=1;nt+=int(p['is_transport'][source]);maxe=max(maxe,peak);eventcalls+=calls
        if save_actions:actions.append([t-elapsed,waiting,elapsed,source,rho,vbefore,float(score(x)),before[0],before[1],x[0],x[1],int(hit),int(elapsed>=duration-1e-10)])
        if hit:reason='active_target';break
        if elapsed>=duration-1e-10 and score(x)>vbefore-V0*duration+ENDPOINT_BUDGET:
            raise RuntimeError('Completed primitive violated required nominal rate')
        if maxe>HX:raise RuntimeError('Operating region exit')
        if number>100000:raise RuntimeError('Unexpected excessive action count')
    final_time=t if hit else horizon
    if not hit and t<horizon:
        try:x=passive(x,horizon-t)
        except ValueError:pass
    return dict(success=bool(hit),time=float(final_time),active=float(active),idle=float(final_time-active),effort=float(effort),actions=number,transport_actions=nt,max_energy=maxe,reason=reason,final=x.tolist(),max_completed_wait=float(max(waits,default=0)),open_wait=float(max(0,horizon-t)) if not hit else 0.,first_action=first_action,candidate_queries=queries,event_flow_calls=eventcalls),actions


def run(num=500,subdivisions=4,label='main',budgets=(1,2,3,4,5)):
    started=time.perf_counter();payload=dict(np.load(ROOT/'library.npz'));initial=np.load(ROOT/'initial_states.npz')['states'][:num];tlib=target_library();results=[];action_records=[]
    for M in budgets:
        for trans in [False,True]:
            lib=Library(payload,M,trans);mode='D+T' if trans else 'D';begin=time.perf_counter()
            for j,x in enumerate(initial):
                row,actions=rollout(x,lib,tlib,2000.,subdivisions,save_actions=True)
                row.update(M=M,variant=mode,index=j);results.append(row)
                action_records.extend([[M,int(trans),j]+a for a in actions])
            group=results[-num:]
            print(json.dumps(dict(M=M,variant=mode,n=num,successes=sum(g['success'] for g in group),max_time=max(g['time'] for g in group if g['success']) if any(g['success'] for g in group) else None,seconds=time.perf_counter()-begin)),flush=True)
            (ROOT/f'{label}_results.json').write_text(json.dumps(results,indent=2))
            np.savez_compressed(ROOT/f'{label}_actions.npz',records=np.array(action_records,dtype=float))
    meta=dict(n=num,subdivisions=subdivisions,max_dt=.005/subdivisions,horizons=[150.,500.,2000.],elapsed_seconds=time.perf_counter()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),library_sha256=hashlib.sha256((ROOT/'library.npz').read_bytes()).hexdigest(),event_module_sha256=hashlib.sha256((ROOT.parent/'event_detection/ball_events.py').read_bytes()).hexdigest(),action_columns=['M','transport_enabled','initial_index','start_time','preceding_wait','executed_duration','assignment_index','start_radius_ratio','V_start','V_end','q_start','p_start','q_end','p_end','target_hit','full_primitive'])
    (ROOT/f'{label}_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['prepare','run']);parser.add_argument('--num',type=int,default=500);parser.add_argument('--subdivisions',type=int,default=4);parser.add_argument('--label',default='main');parser.add_argument('--budgets',type=int,nargs='+',default=[1,2,3,4,5]);args=parser.parse_args()
    if args.task=='prepare':prepare()
    elif args.task=='run':run(args.num,args.subdivisions,args.label,args.budgets)
