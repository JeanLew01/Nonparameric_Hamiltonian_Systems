"""Meeting-driven physical-target double-pendulum experiment (new, frozen cohort).

Run with Python 3.12.  Numerical checks are empirical, not
validated enclosures or an ergodicity certificate.  Existing experiments are
never modified by this module.
"""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,concurrent.futures
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'_cache'))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from numba import njit

G=9.81; SHIFTS=np.array([0.,.25,-.25,.5,-.5,1.,-1.,2.,-2.])
ENERGIES=np.array([5.,15.,30.,45.,60.,90.])

@njit(cache=True,nogil=True)
def velocity(x):
    c=np.cos(x[0]-x[1]);d=2-c*c
    return np.array([(x[2]-c*x[3])/d,(2*x[3]-c*x[2])/d])

@njit(cache=True,nogil=True)
def energy(x):
    v=velocity(x)
    return .5*(x[2]*v[0]+x[3]*v[1])+G*(3-2*np.cos(x[0])-np.cos(x[1]))

@njit(cache=True,nogil=True)
def rhs(x,u,alpha=0.):
    v=velocity(x);a=np.sin(x[0]-x[1])*v[0]*v[1]
    f=u.copy()
    if alpha!=0.:
        z=alpha/max(1.,abs(alpha)*max(abs(v[0]),abs(v[1]))/20.)
        f=np.array([-z*v[1],z*v[0]])
    return np.array([v[0],v[1],-a-2*G*np.sin(x[0])+f[0],a-G*np.sin(x[1])+f[1]])

@njit(cache=True,nogil=True)
def step(x,u,h,alpha=0.):
    k1=rhs(x,u,alpha);k2=rhs(x+.5*h*k1,u,alpha)
    k3=rhs(x+.5*h*k2,u,alpha);k4=rhs(x+h*k3,u,alpha)
    return x+h*(k1+2*k2+2*k3+k4)/6

@njit(cache=True,nogil=True)
def difference(x,y):
    z=x-y
    for j in range(2):z[j]=(z[j]+np.pi)%(2*np.pi)-np.pi
    return z

@njit(cache=True,nogil=True)
def score(x,target):
    E=energy(x)
    if target==0:return E-.4
    return max(58.66-E,E-59.06)

@njit(cache=True,nogil=True)
def target_residual(x,target):
    a=0. if target==0 else np.pi
    z=max(abs((x[0]-a+np.pi)%(2*np.pi)-np.pi),abs((x[1]-a+np.pi)%(2*np.pi)-np.pi))-.2
    return max(z,score(x,target))

@njit(cache=True,nogil=True)
def rollout(x,coef,duration,h=.0025):
    z=x.copy();dt=duration/len(coef);N=max(1,int(np.ceil(dt/h)));hh=dt/N
    maxE=energy(z)
    for u in coef:
        for k in range(N):z=step(z,u,hh);maxE=max(maxE,energy(z))
    return z,maxE

def exact_path(kind):
    if kind=='below':qa,qb=.02,np.pi-.02
    elif kind=='above':qa,qb=np.pi,3*np.pi-.02
    else:qa,qb=90/22,.005
    def speed(q):
        if kind=='below':return .5*np.sin(q)
        E=58.86+31.14*((3*np.pi-q)/(2*np.pi))**2 if kind=='above' else 22*q
        return (1 if kind=='above' else -1)*np.sqrt(max(0.,(E-29.43*(1-np.cos(q)))/2.5))
    def force(q):
        if kind=='below':
            a=.25*np.sin(q)*np.cos(q)
            return np.array([3*a+19.62*np.sin(q),2*a+9.81*np.sin(q)])
        ep=-62.28*(3*np.pi-q)/(4*np.pi**2) if kind=='above' else 22.
        return np.array([.6*ep+1.962*np.sin(q),.4*ep-1.962*np.sin(q)])
    def end(t,y):return (y[0]-qb)*(1 if kind!='down' else -1)
    end.terminal=True;end.direction=1
    sol=solve_ivp(lambda t,y:[speed(y[0])],(0,100),[qa],rtol=2e-12,atol=2e-13,events=end,dense_output=True,max_step=.02)
    T=float(sol.t[-1])
    def state(t):
        q=float(sol.sol(t)[0]);v=speed(q)
        return np.array([q,q,3*v,2*v])
    return T,state,lambda t:force(float(sol.sol(t)[0]))

def adaptive(x,coef,duration):
    z=x.copy();dt=duration/len(coef);mx=energy(z)
    for u in coef:
        s=solve_ivp(lambda t,y:rhs(y,u),[0,dt],z,method='DOP853',rtol=2e-11,atol=2e-12,max_step=.02)
        z=s.y[:,-1];mx=max(mx,max(energy(v) for v in s.y.T))
    return z,mx

def drift(x,t):
    if t==0:return x.copy()
    s=solve_ivp(lambda t,y:rhs(y,np.zeros(2)),[0,t],x,method='DOP853',rtol=2e-12,atol=2e-13,max_step=.02)
    return s.y[:,-1]

def prepare():
    p=dict(model='Full unit-mass unit-length double pendulum; canonical momenta; g=9.81; generalized forces bounded componentwise by20',
        targets={'down':{'angle_center':[0,0],'angle_halfwidth':.2,'energy':[0,.4]},'up':{'angle_center':[float(np.pi)]*2,'angle_halfwidth':.2,'energy':[58.66,59.06]}},
        energies=ENERGIES.tolist(),states_per_energy=12,initial_seed=20261121,noise_seed=20261122,
        initial_sampling='For each energy, reject uniform angles on the torus until potential<E; draw a uniform momentum-direction angle and normalize p by .5 p^T M^-1 p=E-U. Reuse identical states across all target/policy choices; no ergodicity or representative-measure claim.',
        horizon=2000.,output_dt=.1,steps={'main':.0025,'fine':.00125},transport_times=SHIFTS.tolist(),added_transport_counts=[0,2,4,8],
        integrator='Classical RK4 on active primitive segments. Zero-work passive/exploration intervals use RK4 followed by fixed-q radial momentum projection to interval-start energy; infeasible kinetic correction triggers subdivision/retry, never clamping. Identical projected partial-step map in all event probes. No volume-preservation or long chaotic path accuracy claim.',
        numerical_amendment='Initial .005s unprojected method was archived as an invariant-drift pilot after up to .371J work-balance error at90J/2000s, comparable to .4J target bandwidth. Before final evaluation, projection plus halved primary/fine steps was fixed for all policies; no target, cohort, support, force, or exploration parameter was changed.',
        refinement='Predeclared indices0and6 within each energy group, both targets and all four primary policies (96paired trials); same noise and source/library hashes. Local event fixtures additionally compare independent DOP853 and step halving.',
        exploration={'alpha':.8,'dwell':.5,'passive_grace':5.,'law':'fixed per-trial i.i.d. symmetric signs on the global half-second clock; reset grace after each primitive; u=alpha R v/max(1,abs(alpha)*norminf(v)/20)'},
        demonstrations={'up':['below','above'],'down':['down']},snippet_max_duration=.1,control_segments=4,
        support={'max_radius':.15,'boundary_directions':40,'minimum_progress_rate':.001,'method':'Empirical boundary probes plus center, dyadic radius reduction. Fresh checks for every synthesized transport; never a mathematical certificate.'},
        transport='Fresh bounded endpoint synthesis initialized by source waveform; exact passive-shift endpoints. For every endpoint residual>1e-6, uniformly try unchanged-source replay as a progress-only realization, with fresh independent progress/support checks. Store realization type and failed endpoint residual; reject failed progress/support without replacement.',
        construction_amendment='Before any closed-loop evaluation, short-duration endpoint feasibility checks motivated the uniformly applied unchanged-waveform progress-only fallback; no trajectories or outcomes were examined.',
        primary_arms=['D','D+E','D+T8','D+E+T8'],scaling_arms=['D+E','D+E+T2','D+E+T4','D+E+T8'],
        event_detection='Between-step cubic Hermite support crossing: all derivative roots isolated recursively, actual partial-RK4 membership required. Physical target checked on subdivided RK4 intervals and bracketed. Numerical event localization, not a validated continuous-time certificate.',
        horizon_rationale='2000s is fixed before evaluation, with40/150/500/2000s summaries and full0.1s trajectories; nonhits are right-censored, not called impossible.',
        safety_energy=110.,freeze='Protocol fixed before evaluation; construction failures and negative results retained.')
    path=ROOT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2)+'\n')
    rng=np.random.default_rng(p['initial_seed']);xs=[]
    for E in ENERGIES:
        for k in range(12):
            while True:
                q=rng.uniform(-np.pi,np.pi,2);U=G*(3-2*np.cos(q[0])-np.cos(q[1]))
                if U<E:break
            theta=rng.uniform(-np.pi,np.pi);b=np.array([np.cos(theta),np.sin(theta)])
            z=np.r_[q,b];z[2:]*=np.sqrt((E-U)/(.5*b@velocity(z)));xs.append(z)
    xs=np.array(xs)
    noise=np.random.default_rng(p['noise_seed']).choice(np.array([-.8,.8]),size=(len(xs),4001))
    np.savez_compressed(ROOT/'initial_states.npz',states=xs,energies=np.repeat(ENERGIES,12),noise=noise)
    return p

def construct():
    p=prepare();rng=np.random.default_rng(20261123)
    dirs=rng.normal(size=(32,4));dirs/=np.linalg.norm(dirs,axis=1)[:,None];dirs=np.r_[dirs,np.eye(4),-np.eye(4)]
    logs=[]
    for target,kinds in [(0,['down']),(1,['below','above'])]:
        anchors=[];endpoints=[];coeffs=[];radii=[];durations=[];shiftids=[];demoids=[];sourceids=[];modes=[]
        for demo_id,kind in enumerate(kinds):
            T,state,ufun=exact_path(kind);N=int(np.ceil(T/.1));times=np.linspace(0,T,N+1)
            ts=np.linspace(0,T,int(np.ceil(T/.001))+1)
            np.savez_compressed(ROOT/(kind+'_analytic_design.npz'),time=ts,state=np.array([state(t) for t in ts]),input=np.array([ufun(t) for t in ts]))
            original_anchors=[];original_ends=[];original_inputs=[];a=state(0.)
            for si,(ta,tb) in enumerate(zip(times[:-1],times[1:])):
                duration=tb-ta;desired=state(tb)
                # The stored original signal is a four-bin midpoint hold, corrected
                # once by bounded shooting to follow the supplied analytic path.
                seed=np.array([ufun(ta+(j+.5)*duration/4) for j in range(4)])
                def residual(z,x0,goal):return difference(rollout(x0,z.reshape(4,2),duration,.001)[0],goal)
                fit=least_squares(residual,seed.ravel(),args=(a,desired),bounds=(-20,20),method='dogbox',max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-12)
                original=fit.x.reshape(4,2);b,mx=adaptive(a,original,duration)
                if np.linalg.norm(difference(b,desired))>1e-6:raise RuntimeError('Original endpoint fit failed '+kind+str(si))
                original_anchors.append(a.copy());original_ends.append(b.copy());original_inputs.append(original.copy())
                for shift_id,s in enumerate(SHIFTS):
                    aa=drift(a,s);bb=drift(b,s)
                    if shift_id==0:coef=original.copy()
                    else:
                        fit=least_squares(residual,original.ravel(),args=(aa,bb),bounds=(-20,20),method='dogbox',max_nfev=80,ftol=1e-10,xtol=1e-10,gtol=1e-11)
                        coef=fit.x.reshape(4,2)
                    end,mx=adaptive(aa,coef,duration);err=np.linalg.norm(difference(end,bb));endpoint_err=err;mode=0 if shift_id==0 else 1
                    if err>1e-6 and shift_id>0:
                        coef=original.copy();end,mx=adaptive(aa,coef,duration);mode=2
                    drop=score(aa,target)-score(end,target)
                    accepted=False;r=.15
                    if (err<=1e-6 or mode==2) and drop>.001*duration and mx<110:
                        for reduction in range(11):
                            ok=True
                            for d in dirs:
                                z=aa+r*d;zz,HH=rollout(z,coef,duration)
                                if score(z,target)-score(zz,target)<=.001*duration or HH>=110:ok=False;break
                            if ok:accepted=True;break
                            r*=.5
                    logs.append(dict(target=target,demo=kind,snippet=si,shift_id=shift_id,shift=float(s),duration=duration,endpoint_error=endpoint_err,realization=['original','endpoint','progress_replay'][mode],center_drop=float(drop),accepted=accepted,radius=r if accepted else None))
                    if accepted:
                        anchors.append(aa);endpoints.append(end);coeffs.append(coef);radii.append(r);durations.append(duration);shiftids.append(shift_id);demoids.append(demo_id);sourceids.append(si);modes.append(mode)
                if si%30==0:print(kind,si,'/',N,'accepted',len(anchors),flush=True)
                a=b.copy()
            np.savez_compressed(ROOT/(kind+'_original.npz'),time=times,anchors=np.array(original_anchors),endpoints=np.array(original_ends),coefficients=np.array(original_inputs),duration=np.diff(times))
        np.savez_compressed(ROOT/(('down' if target==0 else 'up')+'_library.npz'),anchors=np.array(anchors),endpoints=np.array(endpoints),coefficients=np.array(coeffs),radii=np.array(radii),durations=np.array(durations),shift_ids=np.array(shiftids),demo_ids=np.array(demoids),source_ids=np.array(sourceids),realization_modes=np.array(modes),directions=dirs)
        (ROOT/'construction.json').write_text(json.dumps(logs,indent=2)+'\n')
    print('Construction complete',len(logs),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','construct']);args=ap.parse_args()
    globals()[args.command]()
