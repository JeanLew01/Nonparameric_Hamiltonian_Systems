"""Double-pendulum libraries built from the NMPC demonstrations (16 Nov 2026, RK4 version).

Copy of ../meeting_double_nmpc/construct.py with two changes: the 5 ms grid of
a demonstration is the stored RK4 simulation of demonstrations.py (not a
DOP853 resampling), and the passive shifts of the anchors are integrated
with RK4 at a 1 ms step (not DOP853).  The text below is that of the earlier studies.

Copy of ../meeting_double_transport_fix/construct.py.  Changes: the three
demonstrations are the NMPC rollouts of demonstrations.py (below, above,
down), only the variant `reuse` is built, and the support test of the rows runs in
parallel.  The anchors follow Algorithm 1 of the conference paper
(anchors_of): the next anchor is where the demonstration leaves the current
support ball.  The direction of progress required of a snippet is set by
its start energy; an NMPC demonstration is not monotone in energy, and a
snippet that moves the energy away from the band is rejected.  The text below is that of the earlier study.


The original demonstrations, their 0.1 s snippets and the nested passive
shifts are those of ../meeting_double_10000 (read, never modified).  Two things
change: which signal a shifted placement uses, and how a support is accepted.

Variants
  reuse_cap015 Ablation: as reuse, but with the radius capped at 0.15 as in the
               frozen study.
  reuse        Shifted start z = Phi_s(x_i).  Every stored signal u_j of the
               same target library is evaluated at z (eq. progress-reuse of the
               manuscript), and the one with the largest energy change towards
               the target band is kept.  The frozen study tried only j = i
               after a failed endpoint synthesis.
  frozen_rows  Ablation: the frozen rows (anchors and waveforms of
               ../meeting_double_10000, endpoint synthesis or unchanged
               replay), with the new support acceptance.

Support acceptance (both variants, originals included).  V depends on the
state only through H, and the executor stops a primitive when the energy
reaches the band centre E_*.  A start on one side of the band therefore makes
progress exactly when the work W = H(Pz) - H(z) of the full-duration primitive
has the sign that moves the energy towards the band.  A support is

    B(z, r)  intersected with  {H on the certified side of the band},

and r is halved from 2.4 (0.15 in the two ablation variants, the cap of the
frozen study) until, over the whole ball, the work keeps its sign
with margin 0.001 tau and the path energy stays below 110 J.  Whole-ball
bounds are first order at the anchor plus twice the largest second-order
residual seen on the probe set (the rule of ../audit_double_repair); 2,000
random interior points are then checked directly.  This is a derived
conservative test, not a validated enclosure.

Landing condition (both variants).  A ball spans only a few joules, and
near a band edge the energy gradient is small, so some energy levels outside
the band are covered by no support (for the down target, 0.40-0.62 J and
1.78-1.92 J at N_T = 8).  An action that ends on such a level could never be
followed by another one.  A start energy e is therefore admissible for a row
only if the whole landing interval (plus one 0.02 J grid cell on each side)
lies on covered levels.  The landing interval is [e + W_lo, e + W_hi]
intersected with the whole-ball range of the endpoint energy H o P; ending in the band, or beyond its
centre where the executor stops, is always allowed.  Covered levels are those
of the certified balls on that side of the band, on a 0.02 J grid, for each
nested library separately.  The rule is applied once.  The report states
whether every covered level keeps an admissible support afterwards (true for
N_T >= 4; the sparse libraries N_T <= 2 of the down target lose levels, so
trials can still stop making progress there).  A row whose admissible start
energies form several intervals is stored once per interval.
"""
from pathlib import Path
import os,json,hashlib,argparse,time,concurrent.futures
WORKERS=int(os.environ.get('HCP_WORKERS','8'))
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'_cache'))
import numpy as np
from numba import njit
import benchmark as b
FROZEN=ROOT.parent/'meeting_double_10000'
ELO,EHI=58.66,59.06;DOWN_HI=.4;RATE=.001;SAFE=110.;ALLOWANCE=2.;H_FINE=.001;RMAX=.15;LEVELS=11;RADIUS_START={'reuse':2.4,'reuse_cap015':.15,'frozen_rows':.15}
DEMOS={0:['down'],1:['below','above']};NAMES={0:'down',1:'up'}
INF=float('inf')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_sources():
    assert sha(ROOT/'benchmark.py')==sha(FROZEN/'benchmark.py'), 'kernel differs from the frozen study'
    return {k+'_original.npz':sha(ROOT/(k+'_original.npz')) for k in ['down','below','above']}

@njit(cache=True,nogil=True)
def grad_energy(x):
    f=b.rhs(x,np.zeros(2))
    return np.array([-f[2],-f[3],f[0],f[1]])

@njit(cache=True,nogil=True)
def triple(x,coef,duration):
    """Start energy, endpoint energy and path-maximum energy of one primitive."""
    z,mx=b.rollout(x,coef,duration,H_FINE)
    return np.array([b.energy(x),b.energy(z),mx])

@njit(cache=True,nogil=True)
def centre(a,coef,duration):
    F=triple(a,coef,duration);J=np.zeros((3,4));e=1e-6
    for i in range(4):
        d=np.zeros(4);d[i]=e
        J[:,i]=(triple(a+d,coef,duration)-triple(a-d,coef,duration))/(2*e)
    J[0]=grad_energy(a)
    return F,J

@njit(cache=True,nogil=True)
def probes(a,coef,duration,r,dirs):
    out=np.empty((len(dirs),3))
    for k in range(len(dirs)):out[k]=triple(a+r*dirs[k],coef,duration)
    return out

@njit(cache=True,nogil=True)
def best_signal(z,signals,taus,side):
    """Stored signal with the largest work in the direction `side` at z (safe path only)."""
    best=-1;value=0.
    for j in range(len(signals)):
        F=triple(z,signals[j],taus[j])
        if F[2]>=SAFE:continue
        w=side*(F[1]-F[0])
        if w>value:value=w;best=j
    return best,value

def bounds(F,J,P,dirs,r):
    """Whole-ball [lo,hi] for H, W and Emax (first order plus sampled curvature)."""
    lin=r*(dirs@J.T);res=P-F[None,:]-lin
    c={'H':(F[0],J[0],res[:,0]),'W':(F[1]-F[0],J[1]-J[0],res[:,1]-res[:,0]),'H1':(F[1],J[1],res[:,1]),'Emax':(F[2],J[2],res[:,2])}
    out={}
    for k,(v,g,q) in c.items():
        w=r*float(np.linalg.norm(g))+ALLOWANCE*float(np.max(np.abs(q)));out[k]=(v-w,v+w)
    return out

def window(target,side):
    if target==0:return DOWN_HI,INF
    return (-INF,ELO) if side>0 else (EHI,INF)

def pointwise(P,m,side,lo,hi):
    """Worst signed-work margin over probes inside the window (None if none is inside)."""
    worst=None
    for H0,H1,mx in P:
        if H0<=lo or H0>=hi:continue
        if mx>=SAFE:return -INF
        d=side*(H1-H0)-m;worst=d if worst is None else min(worst,d)
    return worst

def certify(z,coef,tau,target,side,base,dense,rmax=RMAX):
    """Largest dyadic radius whose whole ball keeps the sign of the work; None if rejected."""
    m=RATE*tau;F,J=centre(z,coef,tau);lo,hi=window(target,side)
    extra=[]
    for g in (J[0],J[1],J[1]-J[0],J[2]):
        n=np.linalg.norm(g)
        if n>0:extra+=[g/n,-g/n,.9*g/n,-.9*g/n]
    D=np.r_[base,np.array(extra)];r=rmax;reason='sign'
    for level in range(LEVELS+int(round(np.log2(rmax/RMAX)))):
        P=probes(z,coef,tau,r,D);B=bounds(F,J,P,D,r);Hlo,Hhi=B['H'];Wlo,Whi=B['W'];H1lo,H1hi=B['H1']
        if B['Emax'][1]>=SAFE:reason='safety'
        elif Hlo>=hi or Hhi<=lo:return None,'no_activation_state',F
        elif (Wlo if side>0 else -Whi)<m:reason='sign'
        else:
            pw=pointwise(P,m,side,lo,hi);Q=probes(z,coef,tau,r,dense);dw=pointwise(Q,m,side,lo,hi)
            if (pw is not None and pw<0) or (dw is not None and dw<0):reason='probe'
            else:
                inside=float(np.mean((Q[:,0]>lo)&(Q[:,0]<hi)))
                e=Q[(Q[:,0]>lo)&(Q[:,0]<hi),0];span=[float(e.min()),float(e.max())] if len(e) else None
                if span is None:return None,'no_activation_state',F
                return dict(radius=r,lo=lo,hi=hi,W=[float(Wlo),float(Whi)],H=[float(Hlo),float(Hhi)],H1=[float(min(H1lo,Q[:,1].min())),float(max(H1hi,Q[:,1].max()))],sampled_energy=span,min_dense_margin=dw,fraction_of_ball_admissible=inside),'accepted',F
        r*=.5
    return None,reason,F

GRID=.02
def intervals(mask,g):
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    cuts=np.flatnonzero(np.diff(idx)>1)
    return [(float(g[i]),float(g[j])) for i,j in zip(np.r_[idx[0],idx[cuts+1]],np.r_[idx[cuts],idx[-1]])]

def landing(target,certs,sides):
    """Admissible start-energy intervals per row, and the covered levels before and after."""
    g=np.arange(0.,SAFE+GRID,GRID);n=len(g);edge_hi=DOWN_HI if target==0 else EHI
    span=np.array([c['sampled_energy'] for c in certs]);W=np.array([c['W'] for c in certs]);H1=np.array([c['H1'] for c in certs]);sides=np.array(sides)
    base=[(g>=span[k,0])&(g<=span[k,1])&((g>edge_hi) if sides[k]<0 else (g<ELO)) for k in range(len(certs))]
    adm=[a.copy() for a in base];report={}
    for side in (-1.,1.):
        rows=np.flatnonzero(sides==side)
        if not len(rows):continue
        covered0=np.zeros(n,bool)
        for k in rows:covered0|=base[k]
        if not covered0.any():
            for k in rows:adm[k][:]=False
            report['above' if side<0 else 'below']=dict(uncovered_levels=[],levels_lost_by_the_rule=[],every_covered_level_kept=True);continue
        centre=.2 if target==0 else 58.86
        ok=covered0|((g<=edge_hi+1e-9) if side<0 else (g>=ELO-1e-9))
        # Beyond the band edge towards the centre every level is allowed: the action is stopped at the centre.
        bad=np.r_[0,np.cumsum(~ok)];covered=np.zeros(n,bool)
        for k in rows:
            idx=np.flatnonzero(base[k]);keep=np.zeros(n,bool)
            if len(idx):
                # Endpoint energy: start energy plus work, and also inside the whole-ball range of H o P.
                lo=np.maximum(g[idx]+W[k,0],H1[k,0])-GRID;hi=np.minimum(g[idx]+W[k,1],H1[k,1])+GRID
                # The work keeps its sign over the ball, so the padding never extends back past the start level.
                if side<0:lo=np.maximum(lo,centre);hi=np.minimum(hi,g[idx])
                else:hi=np.minimum(hi,centre);lo=np.maximum(lo,g[idx])
                a=np.clip(np.floor(lo/GRID).astype(int),0,n-1);z=np.clip(np.ceil(hi/GRID).astype(int),0,n-1)
                keep[idx[(bad[z+1]-bad[a])==0]]=True
            adm[k]=keep;covered|=keep
        lim=(g>edge_hi+GRID)&(g<=SAFE) if side<0 else (g<ELO-GRID)
        report['above' if side<0 else 'below']=dict(uncovered_levels=intervals(lim&~covered0&(g<=g[covered0].max())&(g>=g[covered0].min()) if side>0 else lim&~covered0&(g<=g[covered0].max()),g),
            levels_lost_by_the_rule=intervals(covered0&~covered,g),every_covered_level_kept=bool(not np.any(covered0&~covered)))
    covered=np.zeros(n,bool)
    for a in adm:covered|=a
    # A grid point stands for the cell around it: extend each interval by one cell, up to the band edge.
    runs=[]
    for k,a in enumerate(adm):
        if sides[k]<0:runs.append([(max(lo-GRID,edge_hi),hi+GRID) for lo,hi in intervals(a,g)])
        else:runs.append([(lo-GRID,min(hi+GRID,ELO)) for lo,hi in intervals(a,g)])
    return runs,intervals(covered,g),report

def demo_file(kind,variant):
    """reuse: swing-up demonstration from below (demonstrations.py); ablations: the frozen quasi-static one."""
    return ROOT/(kind+'_original.npz')

SUB=5;WINDOW=20;STEP=.025/SUB   # the demonstration on a 5 ms grid; a snippet is the next 0.1 s of its input signal

def fine_demo(kind):
    """States and held inputs of one NMPC demonstration on the 5 ms grid: the stored RK4 simulation, recomputed as a check."""
    o=np.load(demo_file(kind,'reuse'));X=o['states'];U=o['inputs'];fine=[X[0].copy()]
    for k in range(len(U)):
        z=X[k].copy()
        for j in range(SUB):z=b.step(z,U[k],.025/SUB);fine.append(z.copy())
        assert np.array_equal(fine[-1],X[k+1])
    assert np.array_equal(np.array(fine),o['fine_states'])
    return np.array(fine),np.repeat(U,SUB,axis=0)

def required_side(target,E,own):
    """Direction of the energy change that decreases V at start energy E (own direction inside the band)."""
    if target==0:return -1.
    if E<ELO:return 1.
    if E>EHI:return -1.
    return own

def anchors_of(job):
    """Algorithm 1 of the conference paper on one demonstration, with V in place of Delta H.

    Anchor x_i = phi(s) on the demonstration; snippet u_i = the following 0.1 s
    of its input (zero after its end); support radius r_i = the largest tested
    ball on which the progress condition V(end) <= V(start) - v0 tau holds
    (certify).  The next anchor is the last grid point of the demonstration
    that is still inside B(x_i, r_i), at least one grid step ahead and at most
    tau ahead, so consecutive supports overlap.  An anchor without an
    admissible ball is skipped by one grid step.
    """
    kind,target,d=job;base,dense=direction_sets();X,U=fine_demo(kind);n=len(U);out=[];tried=0;s=0
    while s<n:
        coef=np.zeros((WINDOW,2));m=min(WINDOW,n-s);coef[:m]=U[s:s+m];z=X[s].copy();z[:2]=(z[:2]+np.pi)%(2*np.pi)-np.pi
        F0=triple(z,coef,WINDOW*STEP);side=required_side(target,float(F0[0]),1. if F0[1]>F0[0] else -1.)
        cert,reason,F=certify(z,coef,WINDOW*STEP,target,side,base,dense,RADIUS_START['reuse']);tried+=1
        if cert is None:s+=1;continue
        out.append((z,coef,WINDOW*STEP,d,s));nxt=s+1
        while nxt+1<=min(s+WINDOW,n) and np.linalg.norm(b.difference(X[nxt+1].copy(),X[s].copy()))<=cert['radius']:nxt+=1
        s=nxt
    return out,tried,n

def load_originals(target,variant):
    """Original assignments: the accepted anchors of Algorithm 1 on every demonstration of this target."""
    cache=ROOT/(NAMES[target]+'_originals.npz')
    if cache.exists():
        o=np.load(cache);return o['A'],o['C'],o['T'],o['demo'],o['snip']
    jobs=[(kind,target,d) for d,kind in enumerate(DEMOS[target])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as pool:res=list(pool.map(anchors_of,jobs))
    rows=[r for out,_,_ in res for r in out]
    A=np.array([r[0] for r in rows]);C=np.array([r[1] for r in rows]);T=np.array([r[2] for r in rows]);demo=np.array([r[3] for r in rows]);snip=np.array([r[4] for r in rows])
    np.savez_compressed(cache,A=A,C=C,T=T,demo=demo,snip=snip,tried=np.array([t for _,t,_ in res]),grid_points=np.array([n for _,_,n in res]),accepted=np.array([len(o) for o,_,_ in res]))
    return A,C,T,demo,snip

def placements(target,variant):
    """Passive shifts of every original anchor (cached; RK4 with a 1 ms step, forward or backward)."""
    file=ROOT/(NAMES[target]+('_placements.npz' if variant=='reuse' else '_placements_frozen_demo.npz'))
    A,C,T,demo,snip=load_originals(target,variant)
    if file.exists():
        z=np.load(file)['states'];assert z.shape==(len(A),len(b.SHIFTS),4);return z
    z=np.array([[drift_rk4(a,float(s)) for s in b.SHIFTS] for a in A])
    for k in range(len(A)):z[k,:,:2]=(z[k,:,:2]+np.pi)%(2*np.pi)-np.pi
    np.savez_compressed(file,states=z,shifts=b.SHIFTS);return z

@njit(cache=True,nogil=True)
def drift_rk4(x,t):
    """Passive flow over the signed time t with RK4 steps of 1 ms."""
    z=x.copy();n=int(round(abs(t)/H_FINE))
    if n==0:return z
    h=t/n;u=np.zeros(2)
    for _ in range(n):z=b.step(z,u,h)
    return z

def direction_sets():
    rng=np.random.default_rng(20261115);base=rng.normal(size=(104,4));base/=np.linalg.norm(base,axis=1)[:,None]
    base=np.r_[base,np.eye(4),-np.eye(4)]
    dense=np.random.default_rng(20261116).normal(size=(2000,4));dense/=np.linalg.norm(dense,axis=1)[:,None]
    dense*=np.random.default_rng(20261117).uniform(size=(2000,1))**.25
    return base,dense

def _certify(args):return certify(*args)

def construct(variant):
    hashes=check_sources();base,dense=direction_sets();report=dict(variant=variant,sources=hashes,targets={});start=time.perf_counter()
    for target in (0,1):
        name=NAMES[target];A,C,T,demo,snip=load_originals(target,variant);rows=[];logs=[]
        if variant.startswith('reuse'):
            Z=placements(target,variant)
            for i in range(len(A)):
                F0=triple(A[i],C[i],T[i]);own=1. if F0[1]>F0[0] else -1.
                for k,s in enumerate(b.SHIFTS):
                    z=Z[i,k];E=float(b.energy(z))
                    side=required_side(target,E,own)
                    if k==0:j=i
                    else:
                        j,value=best_signal(z,C,T,side)
                        if j<0:logs.append(dict(source=i,shift_id=k,accepted=False,reason='no_signal'));continue
                    rows.append((z,C[j],T[j],k,int(demo[i]),i,int(j),side))
        else:
            lib=np.load(FROZEN/(name+'_library.npz'))
            for r in range(len(lib['radii'])):
                z=lib['anchors'][r];coef=lib['coefficients'][r];tau=float(lib['durations'][r]);F0=triple(z,coef,tau)
                rows.append((z,coef,tau,int(lib['shift_ids'][r]),int(lib['demo_ids'][r]),int(lib['source_ids'][r]),-1-int(lib['realization_modes'][r]),1. if F0[1]>F0[0] else -1.))
        keep=[]
        certify(rows[0][0],rows[0][1],float(rows[0][2]),target,rows[0][7],base,dense,RADIUS_START[variant])   # compile before the pool starts
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
            results=list(pool.map(_certify,[(z,coef,float(tau),target,side,base,dense,RADIUS_START[variant]) for (z,coef,tau,k,d,i,j,side) in rows],chunksize=8))
        for n,(z,coef,tau,k,d,i,j,side) in enumerate(rows):
            cert,reason,F=results[n]
            log=dict(source=int(i),shift_id=int(k),signal=int(j),side=float(side),start_energy=float(F[0]),work=float(F[1]-F[0]),accepted=cert is not None,reason=reason)
            if cert is not None:log.update(cert);keep.append((n,cert))
            logs.append(log)
            if n%400==0:print(variant,name,n,'/',len(rows),'accepted',len(keep),round(time.perf_counter()-start,1),flush=True)
        # Landing condition, separately for each nested library N_T in {0,2,4,8} so that every policy is self-contained.
        sizes={};coverage={};trimmed={}
        for N in [0,2,4,8]:
            sel=[(n,c) for n,c in keep if rows[n][3]<=N]
            runs,cov,edges=landing(target,[c for _,c in sel],[rows[n][7] for n,_ in sel])
            flat=[(n,c,lo,hi) for (n,c),rr in zip(sel,runs) for lo,hi in rr]
            lib=dict(anchors=np.array([rows[n][0] for n,_,_,_ in flat]),coefficients=np.array([rows[n][1] for n,_,_,_ in flat]),durations=np.array([rows[n][2] for n,_,_,_ in flat]),
                 shift_ids=np.array([rows[n][3] for n,_,_,_ in flat]),demo_ids=np.array([rows[n][4] for n,_,_,_ in flat]),source_ids=np.array([rows[n][5] for n,_,_,_ in flat]),
                 signal_ids=np.array([rows[n][6] for n,_,_,_ in flat]),sides=np.array([rows[n][7] for n,_,_,_ in flat]),radii=np.array([c['radius'] for _,c,_,_ in flat]),
                 energy_lo=np.array([lo for _,_,lo,_ in flat]),energy_hi=np.array([hi for _,_,_,hi in flat]),certified_rows=np.array([n for n,_,_,_ in flat]))
            (ROOT/variant).mkdir(exist_ok=True);np.savez_compressed(ROOT/variant/f'{name}_library_T{N}.npz',**lib)
            sizes[str(N)]=int(sum(1 for r in runs if r));coverage[str(N)]=cov
            trimmed[str(N)]=dict(edges=edges,rows_without_admissible_energy=int(sum(1 for r in runs if not r)),rows_split=int(sum(1 for r in runs if len(r)>1)),
                rows_cut=int(sum(1 for (n,c),r in zip(sel,runs) if r and (r[0][0]>c['sampled_energy'][0]+GRID or r[-1][1]<c['sampled_energy'][1]-GRID or len(r)>1))))
        out=dict(shift_ids=np.array([rows[n][3] for n,_ in keep]),radii=np.array([c['radius'] for _,c in keep]))
        acc=[l for l in logs if l['accepted']];N=[0,2,4,8]
        report['targets'][name]=dict(requests=len(logs),accepted=len(acc),
            certified_sizes={str(n):int(np.sum(out['shift_ids']<=n)) for n in N},library_sizes=sizes,covered_energy=coverage,landing=trimmed,
            accepted_by_shift={str(k):int(np.sum(out['shift_ids']==k)) for k in range(9)},
            rejected_by_reason={r:int(sum(1 for l in logs if not l['accepted'] and l['reason']==r)) for r in sorted(set(l['reason'] for l in logs if not l['accepted']))},
            radius_counts={f'{r:g}':int(np.sum(out['radii']==r)) for r in sorted(set(out['radii']),reverse=True)},
            transported_abs_work_quantiles=[float(q) for q in np.quantile([abs(l['work']) for l in acc if l['shift_id']>0] or [0.],[.1,.5,.9])],
            original_abs_work_quantiles=[float(q) for q in np.quantile([abs(l['work']) for l in acc if l['shift_id']==0] or [0.],[.1,.5,.9])],
            own_signal_kept=int(sum(1 for l in acc if l['shift_id']>0 and l['signal']==l['source'])),
            min_dense_margin=min((l['min_dense_margin'] for l in acc if l['min_dense_margin'] is not None),default=None),rows=logs)
        print(json.dumps({k:v for k,v in report['targets'][name].items() if k!='rows'}),flush=True)
    report['seconds']=time.perf_counter()-start;report['source_sha256']=sha(__file__)
    (ROOT/f'construction_{variant}.json').write_text(json.dumps(report,indent=1,default=float)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['reuse','reuse_cap015','frozen_rows']);construct(ap.parse_args().variant)
