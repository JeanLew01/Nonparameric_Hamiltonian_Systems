"""Transported single-pendulum assignments by fiber realization (18 Nov 2026: whole pushes).

Third version.  A transported assignment replays the remainder of the push
that contains its anchor, not only the stored snippet: the demonstration input
from the anchor time to the end of the push (the chain of snippets that D
executes there without waiting), realized by one constant torque that matches
the energy at the end of the push.  With the stored snippets alone, a visit to
a shifted placement completed only part of a push in libration (the endpoint
lands ahead of the next placement), so K = 4 gave no gain there.

The construction is the variant ``direct_fiber`` of
../meeting_single_nmpc_fiber/construct.py, with the same solver, tolerances,
status codes and radius rule; here the original assignments are processed in
parallel and the results are assembled in the original order.

For every original assignment (x_i, u_i, tau_i) and every phase fraction
j/K, K in {2, 4, 8}, the placement is z = Phi_{j T_i / K}(x_i).  The endpoint
of the transported snippet must lie on the passive orbit of Phi_s(P_i x_i):
same energy, and same rotation direction above the separatrix.  The signal is
a constant torque c on the parent's hold schedule, |c| <= 20, with
H(endpoint) = H(P_i x_i) solved by a 161-point scan and bisection; the root of
least |c| is kept.  The support radius comes from the progress and
admissibility bounds evaluated on the implemented torque.
"""
from common import *
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import argparse, time

ENERGY_TOL=1e-9
MIN_RADIUS=1e-10
LEVELS=[(2,[.5]),(4,[.25,.75]),(8,[.125,.375,.625,.875])]


def passive_growth(s):
    w=np.sqrt(b.A/b.I);c=np.cosh(w*abs(s));v=np.sinh(w*abs(s))
    return float(np.linalg.norm(np.array([[c,v/(b.I*w)],[b.I*w*v,c]]),2))


@njit(cache=True)
def fiber_gap(x,c,n,lead,dt,goal_energy):
    y,_,_=b.replay(x,np.full(n,c),lead,dt,4)
    return b.energy(y)-goal_energy


@njit(cache=True)
def fiber_solve(x,n,lead,dt,goal_energy):
    """Least-|c| constant torque reaching goal_energy; nan if none in [-20,20]."""
    grid=np.linspace(-20.,20.,161);vals=np.empty(161)
    for k in range(161):vals[k]=fiber_gap(x,grid[k],n,lead,dt,goal_energy)
    best=np.nan
    for k in range(160):
        lo=grid[k];hi=grid[k+1];flo=vals[k];fhi=vals[k+1]
        if flo==0.:root=lo
        elif flo*fhi>0:continue
        else:
            for _ in range(80):
                mid=(lo+hi)/2;fm=fiber_gap(x,mid,n,lead,dt,goal_energy)
                if flo*fm<=0:hi=mid
                else:lo=mid;flo=fm
            root=(lo+hi)/2
        if np.isnan(best) or abs(root)<abs(best):best=root
    return best


OLD=None;DEMOS=None
def source():
    global OLD
    if OLD is None:OLD=dict(np.load(SOURCE/'original_library.npz'))
    return OLD


def push_remainder(demo_id,anchor_time,dt,stored):
    """Demonstration input from the anchor to the end of its push (samples on the 5 ms grid)."""
    global DEMOS
    if DEMOS is None:DEMOS=dict(np.load(SOURCE/'demonstrations.npz'))
    U=DEMOS[f'controls_{demo_id}'][:,0];k=int(np.floor(anchor_time/dt+1e-9));end=k
    while end<len(U) and U[end]!=0:end+=1
    assert end-k>=len(stored) and np.array_equal(U[k:k+len(stored)],stored),'stored snippet is not the start of the push remainder'
    return np.ascontiguousarray(U[k:end],dtype=float)


def place(ids):
    """Attempts and accepted placements for a block of original assignments, in order."""
    old=source();attempts=[];accepted=[];periods=[]
    for idx in ids:
        x=old['centers'][idx];parent=int(old['parent'][idx]);stored=old['control_flat'][old['offsets'][parent]:old['offsets'][parent+1]]
        lead=float(old['leads'][idx]);dt=float(old['dt'])
        u=push_remainder(int(old['demo_ids'][idx]),float(old['anchor_times'][idx]),dt,stored);tau=lead+(len(u)-1)*dt
        endpoint,_,_=b.replay(x,u,lead,dt,8)
        try:par=b.orbit_parameters(x);period=par[2]/par[3]
        except ValueError:par=None;period=np.nan
        periods.append(period)
        for K,fractions in LEVELS:
            for fraction in fractions:
                # status: 0 accepted, 1 nonperiodic, 4 no torque in [-20,20], 5 accuracy, 6 progress/admissibility,
                # 7 wrong rotation direction at the endpoint.
                status=0;error=np.nan;radius=0.;s=period*fraction
                domain_radius=float(old['radii'][idx]/passive_growth(s)) if par is not None else 0.
                if par is None:status=1
                else:
                    z=b.orbit_state(par,par[4]+par[2]*fraction)
                    try:goal=b.passive(endpoint,s)
                    except ValueError:status=1
                    if status==0:
                        c=fiber_solve(z,len(u),lead,dt,float(b.energy(goal)))
                        if np.isnan(c):status=4
                        else:signal=np.full(len(u),c)
                    if status==0:
                        y,me,mp=b.replay(z,signal,lead,dt,4);fine,_,_=b.replay(z,signal,lead,dt,8)
                        discrepancy=float(b.distance(y,fine));error=float(abs(b.energy(fine)-b.energy(goal)))
                        if b.energy(goal)>b.ES and np.sign(fine[1])!=np.sign(goal[1]):status=7
                        elif error>ENERGY_TOL or b.LV*discrepancy>b.ENDPOINT_BUDGET/10:status=5
                    if status==0:
                        growth=np.exp(b.L*tau)
                        reserve=b.score(z)-b.score(y)-b.V0*tau-b.ENDPOINT_BUDGET
                        radius=min(reserve/(b.LV*(1+growth)),(b.HX-me-(mp+(b.A+20)*dt/4)/b.I*20*dt/4)/(b.LV*growth))
                        if radius<=MIN_RADIUS:status=6
                    if status==0:
                        accepted.append((dict(centers=z,radii=radius,leads=lead,demo_ids=old['demo_ids'][idx],anchor_times=old['anchor_times'][idx],shift=s,is_transport=1,
                            introduced_K=K,phase_fraction=fraction,period=period,original_source=idx,endpoint_error=error,inverse_radius=domain_radius),signal))
                attempts.append([idx,K,fraction,s,status,error,radius,0])
    return attempts,accepted,periods


def construct(workers,limit=None):
    check_sources();start=time.perf_counter()
    old=source();ids=np.flatnonzero(old['is_transport']==0)
    if limit is not None:ids=ids[:limit]
    place(ids[:2])  # compile and cache before the pool starts
    blocks=[ids[k:k+400] for k in range(0,len(ids),400)]
    fields=['centers','radii','leads','demo_ids','anchor_times','parent','shift','is_transport']
    out={k:list(old[k][ids]) for k in fields}
    out.update(introduced_K=[1]*len(ids),phase_fraction=[0.]*len(ids),period=[],original_source=list(ids),endpoint_error=[0.]*len(ids),inverse_radius=[np.nan]*len(ids))
    original_periods=[]
    controls=[];attempts=[];max_error=0.;done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for att,acc,periods in pool.map(place,blocks):
            attempts+=att;original_periods+=periods
            for values,signal in acc:
                values['parent']=len(old['offsets'])-1+len(controls);controls.append(signal)
                for key,value in values.items():out[key].append(value)
                max_error=max(max_error,values['endpoint_error'])
            done+=len(periods)
            if done%20000<400:print(json.dumps(dict(anchors=done,accepted=len(controls),seconds=round(time.perf_counter()-start,1))),flush=True)
    n=len(ids);out['period']=original_periods+out['period'];payload={k:np.asarray(v) for k,v in out.items()}
    assert all(len(v)==n+len(controls) for v in payload.values())
    for k in ['parent','demo_ids','is_transport','introduced_K','original_source']:payload[k]=payload[k].astype(int)
    payload['control_flat']=np.concatenate([old['control_flat']]+controls)
    payload['offsets']=np.r_[old['offsets'],old['offsets'][-1]+np.cumsum([len(c) for c in controls])].astype(int);payload['dt']=old['dt']
    tail=b.boundary_partitions(payload['centers'][n:],payload['radii'][n:],64)
    fine=b.boundary_partitions(payload['centers'][n:],payload['radii'][n:],256)
    assert np.array_equal(tail[2],fine[2]) and np.nanmax(abs(tail[0]-fine[0]))<1e-9
    for k,v in zip(['angles','values','counts'],tail):payload[k]=np.concatenate([old[k][ids],v])
    att=np.asarray(attempts)
    report=dict(variant='direct_fiber',original_assignments=n,attempts=len(attempts),accepted=len(controls),max_endpoint_error=max_error,
        status_counts={str(k):int(np.sum(att[:,4]==k)) for k in range(8)},
        accepted_by_fraction={f'{f:g}':int(np.sum((att[:,2]==f)&(att[:,4]==0))) for f in sorted(set(att[:,2]))},
        seconds=time.perf_counter()-start,source_sha256=sha(__file__),
        rows=[dict(M=m,K=k,assignments=int(np.sum((payload['demo_ids']<m)&(payload['introduced_K']<=k)))) for m in range(1,6) for k in [1,2,4,8]])
    folder=ROOT/'direct_fiber';folder.mkdir(exist_ok=True)
    if limit is None:
        save_npz(folder/'library.npz',**payload)
        save_npz(folder/'construction_attempts.npz',records=att,columns=np.array(['original_source','K','fraction','shift','status','endpoint_error','radius','iterations']))
        (ROOT/'construction_direct_fiber.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True);print('Complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12);parser.add_argument('--limit',type=int)
    args=parser.parse_args();construct(args.workers,args.limit)
