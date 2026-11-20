"""Snippets and supports from the effort-minimal NMPC demonstrations (16 Nov 2026).

Algorithm 1 of the conference paper with the signed graph distance V in place
of the energy error.  The radius profile, its maximization over the snippet
duration, the anchor geometry and the first-exit search are the conference
routines (symplectic_ncp/chain/construction.py at commit 7974f19e); this file
only drives them.  At an anchor x_i = phi(s, x_j, u_j) and a duration t ending
on the demonstration grid,

    r_i(t) = (V(x_i) - V(phi(s + t, x_j, u_j)) - v0 t) / (L_H (1 + e^{L t})),

tau_i maximizes r_i, and the next anchor is the point where the demonstration
leaves the ball (advance at least 1e-4 s and at most tau_i), as before.  The
stored radius is the certified one of ../reference_single/benchmark.py:
r_i(tau_i) computed with a 1e-8 J numerical reserve, reduced where needed to
keep the replayed trajectories in X = {H <= 168 J}.  Assignments with a radius
below 1e-10 are not stored.

One rule is added for demonstrations that contain passive stretches.  An
anchor time at which the demonstration applies zero input moves to the next
instant with nonzero input: where the demonstration coasts, the policy waits
passively in any case, and a snippet that begins with a coast would only
repeat that wait.
"""
from common import *
import time
sys.path.insert(0,str(ROOT.parent/'reference_code_7974f19e'))
from symplectic_ncp.config import get_config
from symplectic_ncp.experts.demonstration import load_demonstrations
from symplectic_ncp.chain import construction as C

MIN_ADVANCE=1e-4
MIN_RADIUS=1e-10


def graph_distance(energy):
    """Vectorized ../reference_single/benchmark.py:score as a function of the energy."""
    e=np.atleast_1d(np.asarray(energy,float));out=np.where(e<b.ELO,b.ELO-e,np.where(e>b.EHI,e-b.EHI,0.))
    low=(e>=b.ELO)&(e<=b.ES);high=(e>b.ES)&(e<=b.EHI)
    out[low]=-np.minimum(e[low]-b.ELO,b.ES-e[low]+(b.EHI-b.ES))
    out[high]=-np.minimum(b.EHI-e[high],e[high]-b.ES+min(b.ES-b.ELO,b.EHI-b.ES))
    return out


class GraphTarget:
    """The conference target with V as its progress function."""
    def __init__(self,system,target):
        self.system=system;self.target=target;self.H_minus=(b.EHI-b.ELO)/2  # V >= -H_minus everywhere
    def energy_distance(self,x):return graph_distance(self.system.hamiltonian(x))
    def contains(self,x):return self.target.contains(x)


def extract(system,target,demo,demo_id):
    states,controls,dt,n=demo.states,demo.controls,demo.dt,demo.num_steps
    profiler=C._RadiusProfiler(system,GraphTarget(system,target),demo,b.V0,b.LV,b.L,'global')
    active=np.flatnonzero(controls[:,0]!=0);rows=[];signals=[];s=0.
    stats=dict(demo=demo_id,name=demo.name,anchors_tried=0,stored=0,rejected_radius=0,rejected_integration=0,binding_progress=0,binding_region=0,passive_jumps=0,stop='end of demonstration',max_substep_error=0.)
    while True:
        k=int(np.floor(s/dt+C._SNAP));theta=s/dt-k
        if theta<C._SNAP or k>=n:theta=0.
        if k>=n:break
        if controls[k,0]==0:
            j=int(np.searchsorted(active,k,side='right'))
            if j==len(active):stats['stop']='no further input';break
            s=float(active[j]*dt);stats['passive_jumps']+=1;continue
        x=states[k].copy() if theta==0 else C._flow(system,states[k],controls[k],theta*dt)
        if target.contains(x)[0]:stats['stop']='physical target';break
        stats['anchors_tried']+=1
        m,r=profiler.argmax(k,theta,x)
        if m==0:
            stats['stop']='no duration with positive radius';stats['stop_time']=float(s);stats['stop_V']=float(b.score(x));break
        lead=(1-theta)*dt;tau=(m-theta)*dt;u=np.ascontiguousarray(controls[k:k+m,0],dtype=float)
        y,me,mp=b.replay(x,u,lead,dt,4);fine,_,_=b.replay(x,u,lead,dt,8);err=float(b.distance(y,fine))
        growth=np.exp(b.L*tau);reserve=b.score(x)-b.score(y)-b.V0*tau-b.ENDPOINT_BUDGET
        rprogress=reserve/(b.LV*(1+growth));rsafe=(b.HX-me-(mp+(b.A+20)*dt/4)/b.I*20*dt/4)/(b.LV*growth)
        radius=float(min(r,rprogress,rsafe));stats['max_substep_error']=max(stats['max_substep_error'],err)
        if radius<=MIN_RADIUS:stats['rejected_radius']+=1;s+=min(MIN_ADVANCE,tau);continue
        if b.LV*err>=b.ENDPOINT_BUDGET/10:stats['rejected_integration']+=1;s+=min(MIN_ADVANCE,tau);continue
        stats['stored']+=1;stats['binding_region']+=int(rsafe<min(r,rprogress));stats['binding_progress']+=int(rprogress<=min(r,rsafe))
        rows.append((x,radius,lead,demo_id,s,tau));signals.append(u)
        # Line 14 of Algorithm 1: first exit of the demonstration from the stored ball, with the conference floor.
        floor=min(MIN_ADVANCE,tau)
        if float(system.distance(C._demo_state(system,demo,s+floor),x))>=radius:s+=floor;continue
        dist=system.distance(states[k+1:k+m+1],x[None,:]);outside=np.flatnonzero(dist>=radius)
        if outside.size:
            j=int(outside[0])+1
            start,start_time=(x,s) if j==1 else (states[k+j-1],(k+j-1)*dt)
            sigma=start_time+C._exit_time(system,start,controls[k+j-1],(k+j)*dt-start_time,x,radius)-s
        else:sigma=tau
        s+=min(max(sigma,MIN_ADVANCE),tau)
    return rows,signals,stats


def main():
    cfg=get_config('single_pendulum');system=cfg.make_system();target=cfg.make_target(system)
    demos=load_demonstrations(ROOT/'demonstrations.npz');start=time.perf_counter();rows=[];signals=[];report=[]
    L_H,L=system.lipschitz_constants(b.HX)
    assert abs(L-b.L)<1e-9 and L_H<=b.LV+1e-12 and b.LV-L_H<1e-2,(L_H,L)
    for j,demo in enumerate(demos):
        assert max(float(system.hamiltonian(demo.states).max()),cfg.H_bar)*1.05<=b.HX+1e-6
        r,u,stats=extract(system,target,demo,j);rows+=r;signals+=u
        radii=np.array([x[1] for x in r]);taus=np.array([x[5] for x in r]);energy=system.hamiltonian(np.array([x[0] for x in r]))
        stats.update(duration=demo.duration,radius_quantiles=np.quantile(radii,[0,.1,.5,.9,1]).tolist(),snippet_duration_quantiles=np.quantile(taus,[0,.1,.5,.9,1]).tolist(),
            anchor_energy_range=[float(energy.min()),float(energy.max())],seconds=time.perf_counter()-start)
        report.append(stats);print(json.dumps(stats),flush=True)
    n=len(rows);centers=np.array([x[0] for x in rows]);radii=np.array([x[1] for x in rows])
    payload=dict(centers=centers,radii=radii,leads=np.array([x[2] for x in rows]),demo_ids=np.array([x[3] for x in rows],dtype=int),anchor_times=np.array([x[4] for x in rows]),
        parent=np.arange(n),shift=np.zeros(n),is_transport=np.zeros(n,dtype=int),control_flat=np.concatenate(signals),offsets=np.r_[0,np.cumsum([len(u) for u in signals])],dt=np.array(demos[0].dt))
    angles,values,counts=b.boundary_partitions(centers,radii,64);fine=b.boundary_partitions(centers,radii,256)
    assert np.array_equal(counts,fine[2]) and np.nanmax(abs(angles-fine[0]))<1e-9
    payload.update(angles=angles,values=values,counts=counts)
    reach=C.max_energy_on_balls(system,centers,radii);assert reach<=b.HX,reach
    save_npz(ROOT/'original_library.npz',**payload)
    summary=dict(constants=dict(v0=b.V0,L=b.L,L_H=b.LV,H_X=b.HX,L_H_on_conference_grid=L_H,min_advance=MIN_ADVANCE,min_radius=MIN_RADIUS,reserve=b.ENDPOINT_BUDGET),
        assignments=n,by_demonstration=[int(np.sum(payload['demo_ids']==j)) for j in range(len(demos))],max_energy_on_supports=reach,demonstrations=report,seconds=time.perf_counter()-start)
    (ROOT/'extraction.json').write_text(json.dumps(summary,indent=1)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='demonstrations'}),flush=True);print('Complete',flush=True)

if __name__=='__main__':main()
