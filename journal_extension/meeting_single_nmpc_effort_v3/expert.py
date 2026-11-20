"""Effort-minimal NMPC expert for the single pendulum (16 Nov 2026, second version: new start states of demonstrations 4 and 5).

The expert of the conference pipeline (symplectic_ncp/experts) is kept as a
closed-loop NMPC rollout on the same plant (RK4, 5 ms) with the same control
period (20 ms), prediction horizon (2 s = 100 intervals), start states and
stopping rule.  Only the optimal control problem differs.  At every control
instant the expert solves

    min   | pi(x_N) - pi_* |  +  eta * sum_k (1 + 0.05 k/N) h |u_k|
    s.t.  x_{k+1} = RK4(x_k, u_k),   |u_k| <= u_E,

where pi(x) = sqrt(2 I H(x)) is the peak momentum of the passive orbit through
x (its momentum at the bottom), pi_* is its value at the centre E_* of the
projected target interval, and the second term is the torque impulse spent.
Both terms have the unit N m s.  The absolute values are handled with slack
variables, so the problem is smooth.  The expert torque bound u_E = 2 N m is a
tenth of the actuator bound of the plant (20 N m).

Because a torque impulse d at momentum p changes pi by d |p| / pi, spending
impulse pays off only where |p| >= eta * pi.  The optimal input is therefore a
push at the bound near the bottom of each swing and zero elsewhere (a
"hands-off" control), and the last push is cut so that pi reaches pi_*.  After
that the input is zero and the pendulum coasts into the physical target.
Inputs below 1e-4 N m in magnitude are applied as exactly zero.

Third version (18 Nov 2026): one push per period.  In the libration
demonstrations the torque has one sign (TORQUE_SIGN), so the expert pushes at
one pass through the bottom per period, as the braking expert of the rotation
demonstrations already does.  With pushes at both passes, the half-period
placement of transport only duplicated the demonstration.  From rest at the
bottom the first push is applied by rule (reset), because the one-sided problem
is degenerate at the equilibrium.

Demonstrations 1-3 start as in the conference configuration (rotations from
160 J in both directions, swing-up from rest at the bottom).  Demonstrations 4
and 5 start at rest at q = +1.5 and -1.5 rad; the conference states (+-3, 0)
lie next to the target and give one push each under this expert.
"""
from pathlib import Path
import dataclasses,json,os,sys,time
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(_v,'1')
import numpy as np
import casadi as ca
ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'reference_code_7974f19e'  # conference repository at commit 7974f19e (git archive)
sys.dont_write_bytecode=True;sys.path.insert(0,str(REFERENCE))
from symplectic_ncp.config import get_config
from symplectic_ncp.experts.demonstration import Demonstration,save_demonstrations
from symplectic_ncp.experts.generate import rollout_expert,demonstration_summary

A,I=19.62,4.0
RT=0.1
ELO,EHI=A*(1+np.cos(RT)),2*A+RT**2/(2*I)
E_STAR=(ELO+EHI)/2
PI_STAR=float(np.sqrt(2*I*E_STAR))
U_EXPERT=2.0
ETA=0.95
GROWTH=0.05  # the impulse weight rises linearly from eta to (1+GROWTH) eta along the horizon, so a push is never postponed
HORIZON=100
SUBSTEPS=4
ZERO_BELOW=1e-4
MAX_DURATION=200.0


def rk4(x,u,h):
    f=lambda z:ca.vertcat(z[1]/I,-A*ca.sin(z[0])+u)
    k1=f(x);k2=f(x+h/2*k1);k3=f(x+h/2*k2);k4=f(x+h*k3)
    return x+h/6*(k1+2*k2+2*k3+k4)


class EffortNMPC:
    """Callable x -> u (shape (1,)) with reset(x0), as rollout_expert expects.

    The nonlinear program is built once (multiple shooting, CasADi SX, IPOPT) and
    re-solved at every control instant with the measured state as parameter.
    The initial guess is the previous solution shifted by one interval, with the
    states re-simulated from the measured state.
    """
    def __init__(self,period,u_expert=U_EXPERT,eta=ETA,horizon=HORIZON,sign=0):
        self.period=float(period);self.u_expert=float(u_expert);self.eta=float(eta);self.N=N=int(horizon);self.sign=int(sign)
        x=ca.SX.sym('x',2);u=ca.SX.sym('u');y=x
        for _ in range(SUBSTEPS):y=rk4(y,u,self.period/SUBSTEPS)
        self.step=ca.Function('step',[x,u],[y])
        X=ca.SX.sym('X',2,N+1);U=ca.SX.sym('U',N);S=ca.SX.sym('S',N);e=ca.SX.sym('e');x0=ca.SX.sym('x0',2)
        g=[X[:,0]-x0]+[X[:,k+1]-self.step(X[:,k],U[k]) for k in range(N)]
        end=X[:,N];peak=ca.sqrt(2*I*(end[1]**2/(2*I)+A*(1-ca.cos(end[0])))+1e-9)
        g+=[U-S,-U-S,peak-PI_STAR-e,PI_STAR-peak-e]
        self.lbg=np.r_[np.zeros(2*(N+1)),np.full(2*N+2,-np.inf)];self.ubg=np.zeros(2*(N+1)+2*N+2)
        lo=0. if self.sign>0 else -self.u_expert;hi=0. if self.sign<0 else self.u_expert   # one-sided torque if sign != 0
        self.lbx=np.r_[np.full(2*(N+1),-np.inf),np.full(N,lo),np.zeros(N),0.]
        self.ubx=np.r_[np.full(2*(N+1),np.inf),np.full(N,hi),np.full(N,self.u_expert),np.inf]
        nlp=dict(x=ca.vertcat(ca.vec(X),U,S,e),p=x0,f=e+self.eta*self.period*ca.dot(ca.DM(1+GROWTH*np.arange(N)/N),S),g=ca.vertcat(*[ca.vec(v) for v in g]))
        self.solver=ca.nlpsol('effort_nmpc','ipopt',nlp,{'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.tol':1e-9,'ipopt.max_iter':2000})
        self.guess=None;self.solves=0;self.failures=0;self.iterations=0

    def reset(self,x0):
        """First guess: a push over the first 0.2 s that moves pi toward pi_*, then zero input.

        At rest the problem is symmetric in u and zero input is a stationary
        point; the guess pushes in the positive direction there.
        """
        x0=np.asarray(x0,float);direction=float(self.sign) if self.sign else (1.0 if x0[1]>=0 else -1.0)
        peak=np.sqrt(2*I*(x0[1]**2/(2*I)+A*(1-np.cos(x0[0])))+1e-9)
        self.guess=np.zeros(self.N);self.guess[:10]=direction*self.u_expert*np.sign(PI_STAR-peak)
        # One-sided expert at the equilibrium: every start time of the first push costs the same, and the
        # optimizer postpones it from solve to solve.  The first push is then applied by its known form
        # (the bound while |p| >= eta pi), and the optimization takes over when it ends.
        self.kick=bool(self.sign) and peak<1e-3;self.direction=direction

    def __call__(self,x):
        x=np.asarray(x,float).reshape(2);U=self.guess;N=self.N
        if getattr(self,'kick',False):
            peak=np.sqrt(2*I*(x[1]**2/(2*I)+A*(1-np.cos(x[0])))+1e-12)
            if peak<1e-3 or abs(x[1])>=self.eta*peak:
                self.guess=np.r_[self.guess[1:],0.0];return np.array([self.direction*self.u_expert])
            self.kick=False
        X=np.empty((2,N+1));X[:,0]=x
        for k in range(N):X[:,k+1]=np.asarray(self.step(X[:,k],U[k])).ravel()
        peak=np.sqrt(2*I*(X[1,-1]**2/(2*I)+A*(1-np.cos(X[0,-1])))+1e-9)
        w0=np.r_[X.ravel(order='F'),U,np.abs(U)+1e-3,abs(peak-PI_STAR)+1e-3]
        sol=self.solver(x0=w0,p=x,lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg);stats=self.solver.stats()
        self.solves+=1;self.iterations+=int(stats['iter_count']);self.failures+=int(not stats['success'])
        u=np.asarray(sol['x']).ravel()[2*(N+1):2*(N+1)+N]
        self.guess=np.r_[u[1:],0.0]
        first=float(np.clip(u[0],0. if self.sign>0 else -self.u_expert,0. if self.sign<0 else self.u_expert))
        return np.array([0.0 if abs(first)<ZERO_BELOW else first])


def one(name):
    """One closed-loop rollout; finished rollouts are kept in rollouts/ so that a rerun reuses them."""
    cfg=get_config('single_pendulum');cfg=dataclasses.replace(cfg,nmpc=dataclasses.replace(cfg.nmpc,max_duration=MAX_DURATION))
    system=cfg.make_system();target=cfg.make_target(system);x0=starts(cfg)[name]
    sign=TORQUE_SIGN.get(name,0);cache=ROOT/'rollouts'/f'{name}.npz';tag=json.dumps(dict(settings(cfg),torque_sign=sign) if sign else settings(cfg),sort_keys=True)
    if cache.exists():
        with np.load(cache) as d:
            if str(d['settings'])==tag:
                row=json.loads(str(d['row']));print(json.dumps(dict(row,reused=True)),flush=True)
                return Demonstration(name,d['states'],d['controls'],float(d['dt']),float(d['control_period'])),row
    expert=EffortNMPC(cfg.control_period,sign=sign);start=time.perf_counter()
    demo=rollout_expert(system,target,expert,x0,name,cfg)
    row=demonstration_summary(system,target,demo);u=demo.controls[:,0]
    row.update(active_time=float(np.count_nonzero(u)*demo.dt),pushes=int(np.count_nonzero((u[1:]!=0)&(u[:-1]==0))+(u[0]!=0)),impulse=float(np.abs(u).sum()*demo.dt),
        solves=expert.solves,failed_solves=expert.failures,mean_iterations=expert.iterations/max(1,expert.solves),seconds=time.perf_counter()-start)
    cache.parent.mkdir(exist_ok=True);np.savez(cache,states=demo.states,controls=demo.controls,dt=demo.dt,control_period=demo.control_period,settings=np.array(tag),row=np.array(json.dumps(row)))
    print(json.dumps(row),flush=True)
    return demo,row


TORQUE_SIGN={'libration_rest':1,'rest_p1.5':1,'rest_m1.5':-1}   # torque sign allowed in the libration demonstrations
PEAK_160=float(np.sqrt(2*I*160.0))
CANDIDATES={'rest_p0.5':(0.5,0.0),'rest_m0.5':(-0.5,0.0),'rest_p1.0':(1.0,0.0),'rest_m1.0':(-1.0,0.0),'rest_p1.5':(1.5,0.0),'rest_m1.5':(-1.5,0.0),
    'rotation_cw_bottom':(0.0,-PEAK_160),'rotation_ccw_bottom':(0.0,PEAK_160)}


def starts(cfg):
    """Start states by name: demonstrations 1-3 of the conference configuration and the candidates for 4 and 5."""
    out={e.name:tuple(e.x0) for e in cfg.experts[:3]};out.update(CANDIDATES);return out


def settings(cfg):
    return dict(u_expert=U_EXPERT,eta=ETA,weight_growth=GROWTH,horizon=HORIZON,control_period=cfg.control_period,sim_dt=cfg.sim_dt,substeps=SUBSTEPS,zero_below=ZERO_BELOW,E_star=E_STAR,pi_star=PI_STAR,
        energy_eps=cfg.energy_eps,demo_delta=cfg.demo_delta,max_duration=MAX_DURATION,first_guess='push over the first 0.2 s, then zero')


DEMONSTRATIONS=('rotation_cw','rotation_ccw','libration_rest','rest_p1.5','rest_m1.5')  # 4 and 5 chosen with explore_candidates.py


def main(names):
    """Without arguments: the five demonstrations of the study.  With names: only those rollouts (candidates)."""
    from concurrent.futures import ProcessPoolExecutor
    chosen=names or list(DEMONSTRATIONS)
    with ProcessPoolExecutor(max_workers=5) as pool:results=list(pool.map(one,chosen))
    if not names:
        cfg=get_config('single_pendulum');cfg=dataclasses.replace(cfg,nmpc=dataclasses.replace(cfg.nmpc,max_duration=MAX_DURATION));settings_=settings(cfg)
        save_demonstrations(ROOT/'demonstrations.npz',[r[0] for r in results],fingerprint=json.dumps(settings_,sort_keys=True))
        (ROOT/'expert_report.json').write_text(json.dumps(dict(settings=settings_,demonstrations=[r[1] for r in results]),indent=1)+'\n')
    print('Complete',flush=True)

if __name__=='__main__':main(sys.argv[1:])
