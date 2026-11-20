"""Support-event executor for the corrected double-pendulum study (15 Nov 2026).

Copied from ../audit_double_repair/execute_split.py (itself the frozen executor
of ../meeting_double_10000 plus action end records, the randomized exploration
protocol and energy windows).  Two changes:

  * Stopping rule (interrupt=1).  A primitive is stopped when the energy
    reaches the centre E_* of the target band (58.86 J for up, 0.20 J for
    down), the rule the single-pendulum study already uses.  The crossing is
    localized by bisection on the actual partial step.  The earlier ablation
    stopped at the band edge, which left V = 0 to rounding and froze trials
    there; that variant is not used.
  * The exploration gain kappa is a job parameter (it was the constant 0.8).

  * Continuation guard (guard=1; used for transported libraries without
    exploration, as in the single-pendulum policy).  Before a transported
    action the policy predicts its stopped endpoint.  The action is executed
    only if it enters the target or the target band, or if passive motion
    from the endpoint reaches the target or an original support within
    GUARD_HORIZON seconds.  Otherwise transported assignments are skipped
    for GUARD_RETRY seconds and the original policy continues.

With interrupt=0, guard=0 and infinite windows the numerical map equals the
frozen executor (check_executor.py).
"""
from pathlib import Path
import os,json,time,argparse,concurrent.futures,hashlib
import numpy as np
from numba import njit
import benchmark as b
ROOT=b.ROOT

@njit(cache=True,nogil=True)
def flow_step(x,u,h,alpha=0.):
    """RK4, with fixed-q radial momentum projection for zero-work modes.

    An infeasible kinetic-energy correction triggers subdivision of the whole
    requested interval; it is never clamped to a different energy surface.
    This numerical projection is not an applied force or a volume-preserving
    integration claim. Event probes use this identical map.
    """
    if np.max(np.abs(u))>0:return b.step(x,u,h,alpha)
    E=b.energy(x)
    subdivisions=1
    for retry in range(20):
        z=x.copy();valid=True;dt=h/subdivisions
        for k in range(subdivisions):
            y=b.step(z,u,dt,alpha)
            U=b.G*(3-2*np.cos(y[0])-np.cos(y[1]));desired=E-U
            v=b.velocity(y);kinetic=.5*(y[2]*v[0]+y[3]*v[1])
            if desired<0 or kinetic<0 or (kinetic==0 and desired>0):valid=False;break
            if kinetic>0:y[2:]*=np.sqrt(desired/kinetic)
            z=y
        if valid:return z
        subdivisions*=2
    raise ValueError('Zero-work energy projection failed after subdivision')

@njit(cache=True,nogil=True)
def polyval(c,x,n):
    y=c[n]
    for j in range(n-1,-1,-1):y=y*x+c[j]
    return y

@njit(cache=True,nogil=True)
def polynomial_roots_unit(c,degree):
    """Real roots in [0,1], partitioning successively by derivative roots.

    This is a numerical root isolation routine, not interval arithmetic.
    It includes even-multiplicity roots at stationary partition boundaries.
    """
    while degree>0 and abs(c[degree])<1e-25:degree-=1
    old=np.empty(7);oldn=0
    if degree==0:return old,0
    for d in range(1,degree+1):
        p=np.zeros(7)
        for j in range(d+1):
            v=c[j+degree-d]
            for k in range(degree-d):v*=j+degree-d-k
            p[j]=v
        scale=np.sum(np.abs(p));tol=max(1e-25,scale*2e-14)
        cuts=np.empty(9);cuts[0]=0.
        for j in range(oldn):cuts[j+1]=old[j]
        cuts[oldn+1]=1.;new=np.empty(7);n=0
        for j in range(oldn+1):
            lo=cuts[j];hi=cuts[j+1];fl=polyval(p,lo,d);fh=polyval(p,hi,d)
            if abs(fl)<=tol and (n==0 or lo-new[n-1]>1e-10):new[n]=lo;n+=1
            if fl*fh<0:
                for k in range(48):
                    mid=(lo+hi)/2;fm=polyval(p,mid,d)
                    if fl*fm<=0:hi=mid;fh=fm
                    else:lo=mid;fl=fm
                z=(lo+hi)/2
                if n==0 or z-new[n-1]>1e-10:new[n]=z;n+=1
        if abs(polyval(p,1.,d))<=tol and (n==0 or 1-new[n-1]>1e-10):new[n]=1.;n+=1
        old=new;oldn=n
    return old,oldn

@njit(cache=True,nogil=True)
def ball_event(x,y,f0,f1,h,c,r,u,alpha):
    """Earliest verified membership, or -1. Hermite minima guide exact RK4 probes."""
    a=b.difference(x,c);delta=y-x
    for j in range(4):
        lo=min(a[j],a[j]+delta[j]);hi=max(a[j],a[j]+delta[j])
        # Cubic overshoot is bounded by coefficient deviations from its chord.
        dev=abs(h*f0[j]-delta[j])+abs(3*delta[j]-h*(2*f0[j]+f1[j]))+abs(-2*delta[j]+h*(f0[j]+f1[j]))
        if lo-dev>r or hi+dev<-r:return -1.
    C=np.empty((4,4));C[:,0]=a;C[:,1]=h*f0
    C[:,2]=3*delta-h*(2*f0+f1);C[:,3]=-2*delta+h*(f0+f1)
    P=np.zeros(7)
    for j in range(4):
        for k in range(4):
            for l in range(4):P[k+l]+=C[j,k]*C[j,l]
    P[0]-=r*r
    if P[0]<=0:return 0.
    D=np.zeros(7)
    for j in range(6):D[j]=(j+1)*P[j+1]
    roots,n=polynomial_roots_unit(D,5)
    candidates=np.empty(9)
    for j in range(n):candidates[j]=roots[j]
    candidates[n]=1.
    for j in range(n+1):
        z=candidates[j]
        if polyval(P,z,6)>1e-10:continue
        actual=flow_step(x,u,h*z,alpha);d=b.difference(actual,c)
        if np.dot(d,d)>r*r:
            # Retry actual flow around the interpolant minimum; never execute
            # an action merely because the interpolant reports membership.
            for retry in range(1,5):
                zz=min(1.,max(0.,z+((-1.)**retry)*retry*1e-5))
                trial=flow_step(x,u,h*zz,alpha);dd=b.difference(trial,c)
                if np.dot(dd,dd)<=r*r:z=zz;actual=trial;d=dd;break
        if np.dot(d,d)<=r*r:
            lo=0.;hi=z
            # Use the left stationary partition boundary when it is outside;
            # all preceding minima failed actual membership.
            if j>0:
                q=candidates[j-1];qq=b.difference(flow_step(x,u,h*q,alpha),c)
                if np.dot(qq,qq)>r*r:lo=q
            for k in range(40):
                mid=(lo+hi)/2;dd=b.difference(flow_step(x,u,h*mid,alpha),c)
                if np.dot(dd,dd)<=r*r:hi=mid
                else:lo=mid
            # Return an interior representable value, verified again by caller.
            return min(z,hi+1e-10)
    return -1.

@njit(cache=True,nogil=True)
def bin_id(x):
    i=int(((x[0]+np.pi)%(2*np.pi))*16/(2*np.pi))
    j=int(((x[1]+np.pi)%(2*np.pi))*16/(2*np.pi))
    k=int(np.floor((x[2]+32)/2));l=int(np.floor((x[3]+32)/2))
    if k<0 or k>=32 or l<0 or l>=32:return -1
    return ((i*16+j)*32+k)*32+l

def make_index(anchors,radii,keep=None):
    # At H<=110 and |u|inf<=20: ||v||<=24.01 and |pdot_i|<329.
    # A .005s interval moves each q by<.121 and p by<1.65 while safe.
    # Inflation .20/2.0 includes these bounds and all support radii.
    buckets={}
    for a,(c,r) in enumerate(zip(anchors,radii)):
        if keep is not None and not keep[a]:continue
        qs=[]
        for j in range(2):
            cc=(c[j]+np.pi)%(2*np.pi);w=r+.2
            qs.append(set(k%16 for k in range(int(np.floor((cc-w)*16/(2*np.pi))),int(np.floor((cc+w)*16/(2*np.pi)))+1)))
        ps=[range(max(0,int(np.floor((c[j]-r-2+32)/2))),min(31,int(np.floor((c[j]+r+2+32)/2)))+1) for j in (2,3)]
        for i in qs[0]:
            for j in qs[1]:
                for k in ps[0]:
                    for l in ps[1]:buckets.setdefault(((i*16+j)*32+k)*32+l,[]).append(a)
    offsets=np.zeros(16*16*32*32+1,dtype=np.int64)
    for k,v in buckets.items():offsets[k+1]=len(v)
    np.cumsum(offsets,out=offsets);rows=np.empty(offsets[-1],dtype=np.int64)
    for k,v in buckets.items():rows[offsets[k]:offsets[k+1]]=v
    return offsets,rows

@njit(cache=True,nogil=True)
def eligible(x,anchors,radii,offsets,rows,elo,ehi,E):
    k=bin_id(x);idx=-1;best=1.+1e-14
    if k<0:return idx
    for jj in range(offsets[k],offsets[k+1]):
        j=rows[jj]
        if E<elo[j] or E>ehi[j]:continue
        d=b.difference(x,anchors[j]);r=radii[j]
        if np.max(np.abs(d))>r:continue
        v=np.dot(d,d)/(r*r)
        if v<=1. and v<best:idx=j;best=v
    return idx

@njit(cache=True,nogil=True)
def support_entry(x,y,u,alpha,h,anchors,radii,offsets,rows,elo,ehi,E):
    k=bin_id(x);best=2.
    if k<0:return -1.
    f0=b.rhs(x,u,alpha);f1=b.rhs(y,u,alpha)
    for jj in range(offsets[k],offsets[k+1]):
        j=rows[jj]
        if E<elo[j] or E>ehi[j]:continue
        s=ball_event(x,y,f0,f1,h,anchors[j],radii[j],u,alpha)
        if s>=0 and s<best:best=s
    return best if best<=1 else -1.

@njit(cache=True,nogil=True)
def target_entry(x,u,alpha,h,target):
    # Physical angle-box entry cannot be inferred from energy alone. Four
    # subintervals and bisection check the actual partial RK4 trajectory.
    # Step-halving audits detect any practical missed narrow target excursion.
    for j in range(1,5):
        s=j/4.;z=flow_step(x,u,h*s,alpha)
        if b.target_residual(z,target)<=0:
            lo=(j-1)/4.;hi=s
            for k in range(38):
                mid=(lo+hi)/2
                if b.target_residual(flow_step(x,u,h*mid,alpha),target)<=0:hi=mid
                else:lo=mid
            return hi
    return -1.

GUARD_HORIZON=5.;GUARD_RETRY=.25

@njit(cache=True,nogil=True)
def continuation(x,j,target,anchors,coef,radii,durations,offsets0,rows0,elo,ehi,hmax):
    """Predicted outcome of assignment j from x: True if the target or the band is entered during
    the action (stop rule included), or if passive motion from its endpoint reaches the target or an
    original support within GUARD_HORIZON."""
    Estar=.2 if target==0 else 58.86;z=x.copy();side=1. if b.energy(z)<Estar else -1.
    dt=durations[j]/coef.shape[1];n=max(1,int(np.ceil(dt/hmax)));hh=dt/n
    for k in range(coef.shape[1]):
        for i in range(n):
            z=b.step(z,coef[j,k],hh)
            if b.target_residual(z,target)<=0 or side*(Estar-b.energy(z))<=0:return True
    if b.score(z,target)<=0:return True
    for i in range(2):z[i]=(z[i]+np.pi)%(2*np.pi)-np.pi
    E=b.energy(z);zero=np.zeros(2);steps=int(np.ceil(GUARD_HORIZON/hmax))
    for i in range(steps):
        z=flow_step(z,zero,hmax,0.)
        for q in range(2):z[q]=(z[q]+np.pi)%(2*np.pi)-np.pi
        if b.target_residual(z,target)<=0:return True
        if eligible(z,anchors,radii,offsets0,rows0,elo,ehi,E)>=0:return True
    return False

@njit(cache=True,nogil=True)
def simulate(x0,target,anchors,coef,radii,durations,offsets,rows,noise,explore,hmax,horizon,protocol,modes,dwell,interrupt,elo,ehi,guard,transported,offsets0,rows0):
    nout=int(round(horizon/.1))+1;trace=np.empty((nout,5));x=x0.copy();trace[0,:4]=x;trace[0,4]=b.score(x,target)
    t=0.;nextout=1;active=-1;elapsed=0.;stall=0.;actions=0;completed=0;progress_bad=0;outside=0
    first_action=-1.;first_band=-1.;hit=-1.;stop=0.;Emax=b.energy(x);acttime=0.;extime=0.;work=0.;effort=0.;vstart=0.
    tblock=-1.;guard_checks=0;guard_rejections=0;side=1.;Estar=.2 if target==0 else 58.86;starts=np.full((20000,10),np.nan);startn=0;zero=np.zeros(2);segend=-1.;pointer=0;segalpha=0.;interrupted=0;current=-1;Ewait=b.energy(x0)
    while t<horizon-1e-10:
        if b.target_residual(x,target)<=0:hit=t;break
        V=b.score(x,target)
        if V<=0 and first_band<0:first_band=t
        if active>=0 and elapsed>=durations[active]-1e-11:
            if vstart-V<.001*durations[active]-1e-8:progress_bad+=1
            starts[current,8]=t;starts[current,9]=V
            completed+=1;active=-1;elapsed=0.;stall=0.;Ewait=b.energy(x)
        if interrupt and active>=0 and side*(Estar-b.energy(x))<=0:
            starts[current,8]=t;starts[current,9]=V
            interrupted+=1;active=-1;elapsed=0.;stall=0.;Ewait=b.energy(x)
        if active<0 and V>0:
            if guard and t<tblock:j=eligible(x,anchors,radii,offsets0,rows0,elo,ehi,Ewait)
            else:j=eligible(x,anchors,radii,offsets,rows,elo,ehi,Ewait)
            if guard and j>=0 and transported[j]:
                guard_checks+=1
                if not continuation(x,j,target,anchors,coef,radii,durations,offsets0,rows0,elo,ehi,hmax):
                    guard_rejections+=1;tblock=t+GUARD_RETRY
                    j=eligible(x,anchors,radii,offsets0,rows0,elo,ehi,Ewait)
            if j>=0:
                active=j;elapsed=0.;vstart=V;actions+=1;side=1. if b.energy(x)<Estar else -1.
                if first_action<0:first_action=t
                d=b.difference(x,anchors[j]);ratio=np.dot(d,d)/radii[j]**2
                if ratio>1:outside+=1
                assert startn<len(starts), 'Action record capacity exceeded'
                starts[startn,:4]=x;starts[startn,4:8]=np.array([t,float(j),ratio,V]);current=startn;startn+=1;segend=-1.
        u=zero;alpha=0.;h=min(hmax,horizon-t,nextout*.1-t)
        if active>=0:
            dt=durations[active]/coef.shape[1];bin=min(int((elapsed+1e-12)/dt),coef.shape[1]-1)
            u=coef[active,bin];h=min(h,(bin+1)*dt-elapsed,durations[active]-elapsed)
        else:
            # Split exactly at exploration activation and noise changes.
            if explore and stall>=5.-1e-11 and protocol==0:
                ni=min(int((t+1e-11)/.5),len(noise)-1);alpha=noise[ni];h=min(h,(ni+1)*.5-t)
            elif explore and stall>=5.-1e-11:
                if segend<0 or t>=segend-1e-11:
                    assert pointer<len(modes), 'Exploration stream exhausted'
                    segalpha=modes[pointer];segend=t+dwell[pointer];pointer+=1
                alpha=segalpha;h=min(h,segend-t)
            elif explore:h=min(h,5.-stall)
        if guard and active<0 and t<tblock:h=min(h,tblock-t)
        if h<1e-12:
            if nextout<nout and nextout*.1-t<1e-10:trace[nextout,:4]=x;trace[nextout,4]=V;nextout+=1
            else:t+=1e-12
            continue
        y=flow_step(x,u,h,alpha)
        ev=-1.
        if active<0 and V>0:
            if guard and t<tblock:ev=support_entry(x,y,u,alpha,h,anchors,radii,offsets0,rows0,elo,ehi,Ewait)
            else:ev=support_entry(x,y,u,alpha,h,anchors,radii,offsets,rows,elo,ehi,Ewait)
        # Cheap necessary target bounds avoid four extra integrations far away.
        tar=-1.;a=0. if target==0 else np.pi
        close=True
        for j in range(2):
            d=abs((x[j]-a+np.pi)%(2*np.pi)-np.pi)
            if d>.2+25*h:close=False
        if close:tar=target_entry(x,u,alpha,h,target)
        chosen=1.
        if ev>=0:chosen=min(chosen,max(ev,1e-8))
        if tar>=0:chosen=min(chosen,tar)
        if interrupt and active>=0 and side*(Estar-b.energy(x))>0:
            # Localize the crossing of the band centre on the actual partial step.
            lo=0.;hi=-1.
            for j in range(1,5):
                if side*(Estar-b.energy(flow_step(x,u,h*chosen*j/4.,alpha)))<=0:lo=chosen*(j-1)/4.;hi=chosen*j/4.;break
            if hi>0:
                for k in range(38):
                    mid=(lo+hi)/2
                    if side*(Estar-b.energy(flow_step(x,u,h*mid,alpha)))<=0:hi=mid
                    else:lo=mid
                chosen=hi
        hh=h*chosen
        if chosen<1:y=flow_step(x,u,hh,alpha)
        # RK4 quadrature for force-work balance and effort.
        if active>=0 or alpha!=0:
            k1=b.rhs(x,u,alpha);z2=x+.5*hh*k1;k2=b.rhs(z2,u,alpha);z3=x+.5*hh*k2;k3=b.rhs(z3,u,alpha);z4=x+hh*k3
            for z,w in zip((x,z2,z3,z4),(1.,2.,2.,1.)):
                v=b.velocity(z);f=u
                if alpha!=0:
                    al=alpha/max(1.,abs(alpha)*max(abs(v[0]),abs(v[1]))/20.);f=np.array([-al*v[1],al*v[0]])
                work+=hh*w*np.dot(v,f)/6;effort+=hh*w*np.dot(f,f)/6
        if active>=0:elapsed+=hh;acttime+=hh
        else:
            stall+=hh
            if alpha!=0:extime+=hh
        x=y;t+=hh
        for j in range(2):x[j]=(x[j]+np.pi)%(2*np.pi)-np.pi
        Emax=max(Emax,b.energy(x))
        if b.energy(x)>=110:stop=-1.;break
        if nextout<nout and t>=nextout*.1-1e-10:
            trace[nextout,:4]=x;trace[nextout,4]=b.score(x,target);nextout+=1
    if hit<0 and b.target_residual(x,target)<=0:hit=t
    # Absorbing *recording convention* after an actual hit; no invariance claim.
    for k in range(nextout,nout):trace[k,:4]=x;trace[k,4]=b.score(x,target)
    assert nextout>=1 and nextout<=nout
    assert np.all(np.isfinite(trace)), 'Nonfinite trajectory output'
    metrics=np.array([1. if hit>=0 else 0.,hit if hit>=0 else horizon,first_action,first_band,actions,completed,progress_bad,outside,Emax,acttime,extime,effort,b.energy(x)-b.energy(x0)-work,stop,t,float(interrupted),float(pointer),float(guard_checks),float(guard_rejections)])
    return trace,metrics,x,starts[:startn]

ARMS={'D':(0,False),'D+E':(0,True),'D+T8':(8,False),'D+E+T8':(8,True),'D+E+T2':(2,True),'D+E+T4':(4,True)}
STEPS={'main':.0025,'fine':.00125,'finer':.000625}
H_MINUS,H_PLUS=.25,.75

def stream(trial,seed,kappa):
    """Per-trial randomized-protocol draws; shared by every policy and target."""
    rng=np.random.default_rng(np.random.SeedSequence([20261115,seed,trial]))
    n=int(10000./H_MINUS)+8
    return rng.choice(np.array([0.,kappa,-kappa]),size=n),rng.uniform(H_MINUS,H_PLUS,size=n)

def run_job(job):
    """job: dict(study,library,target,arm,clock,ids,horizon,protocol,seed,interrupt,kappa)."""
    study=job['study'];target=job['target'];arm=job['arm'];clock=job['clock'];horizon=job['horizon'];t0=time.monotonic()
    N,explore=ARMS[arm];name='down' if target==0 else 'up'
    libpath=Path(job['library'])/(name+'_library.npz')
    if not libpath.exists():libpath=Path(job['library'])/f'{name}_library_T{N}.npz'   # one self-contained library per transport count
    lib=dict(np.load(libpath));select=lib['shift_ids']<=N;A=lib['anchors'][select];C=lib['coefficients'][select];R=lib['radii'][select];D=lib['durations'][select]
    LO=lib['energy_lo'][select] if 'energy_lo' in lib else np.full(len(R),-np.inf);HI=lib['energy_hi'][select] if 'energy_hi' in lib else np.full(len(R),np.inf)
    guard=bool(job.get('guard',False)) and N>0 and not explore;transported=lib['shift_ids'][select]>0
    offsets,rows=make_index(A,R);offsets0,rows0=make_index(A,R,~transported) if guard else (offsets,rows);init=dict(np.load(ROOT/'initial_states.npz'));h=STEPS[clock]
    dest=ROOT/'results'/study/clock/name/arm;dest.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'benchmark.py',ROOT/'execute.py',ROOT/'initial_states.npz',libpath]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity=dict(source_hashes=hashes,study=study,target=target,arm=arm,transport_count=N,explore=explore,step=h,horizon=horizon,protocol=job['protocol'],seed=job['seed'],interrupt=job['interrupt'],dwell=[H_MINUS,H_PLUS],kappa=job['kappa'],guard=guard,guard_horizon=GUARD_HORIZON,guard_retry=GUARD_RETRY)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=[]
    for i in job['ids']:
        file=dest/(str(i)+'.npz')
        if file.exists():
            with np.load(file) as prior:
                assert str(prior['fingerprint'])==fingerprint, 'Stale trial must be archived before reuse: '+str(file)
                out.append((i,prior['metrics'].copy()))
            continue
        modes,dwell=stream(i,job['seed'],job['kappa']) if job['protocol']==1 else (np.zeros(1),np.ones(1))
        trace,metrics,end,starts=simulate(init['states'][i],target,A,C,R,D,offsets,rows,init['noise'][i]*(job['kappa']/.8),explore,h,horizon,job['protocol'],modes,dwell,job['interrupt'],LO,HI,guard,transported,offsets0,rows0)
        # Compact storage: the 0.1 s trace is thinned to 1 s (score only at 0.1 s).
        tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,score=trace[:,4].astype(np.float32),trace_1s=trace[::10],metrics=metrics,terminal=end,actions=starts,library_rows=np.flatnonzero(select),initial=init['states'][i],horizon=horizon,step=h,fingerprint=fingerprint,identity=json.dumps(identity,sort_keys=True),trial_index=i);tmp.replace(file)
        out.append((i,metrics))
    return dict(study=study,target=name,arm=arm,clock=clock,ids=job['ids'],hits=[int(m[0]) for _,m in out],seconds=time.monotonic()-t0)
