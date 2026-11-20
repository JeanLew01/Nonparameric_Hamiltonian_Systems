"""NMPC demonstrations for the double pendulum (16 Nov 2026; RK4 plant, as for the single pendulum).

Copy of ../meeting_double_nmpc/demonstrations.py.  One change: the closed-loop
plant is integrated with the classical RK4 at a 5 ms step (five steps per
control interval), the simulation of the single-pendulum expert, instead of
DOP853.  The 5 ms states are stored and are the grid of construct.py.

As in the conference version (Ouyang, Liu, Mallada; Section IV and the code
symplectic_ncp/experts), every demonstration is a closed-loop rollout of a
nonlinear model predictive controller, computed offline with CasADi and
do-mpc.  The controller is used only here.  The chain policy later replays
pieces of the recorded input signals and never calls it.

  min  sum_k e(x_k)' Q e(x_k) + R |u_k|^2 + e(x_N)' Q_N e(x_N),   |u_i| <= 20,

with the chordal angle error e_q^2 = 2 (1 - cos(q - q*)), the weights of the
conference pendulum (Q = 10 on angles and 0.1 on momenta, Q_N = 10 Q,
R = 1e-3), a prediction horizon of 2 s and orthogonal collocation.  A soft
constraint H(x) <= 100 J (penalty 1e4, as the energy cap of the conference
code) keeps the demonstrations inside the operating region H <= 110 J.  The
input is recomputed every 0.025 s and held in between.  The plant is the model of
benchmark.py integrated with RK4 (step 5 ms).  A rollout stops at the first multiple of
0.1 s at which the state lies in the physical target, which is checked on the
recorded trajectory.

Start states (those of the earlier hand-made demonstrations):
  below  rest near the hanging state, q = (0.02, 0.02)            -> up-up
  above  upright configuration moving along the diagonal at 90 J  -> up-up
  down   q = (90/22, 90/22) moving along the diagonal at 90 J     -> down-down
"""
from pathlib import Path
import json,time,warnings
import numpy as np
import casadi as ca
with warnings.catch_warnings():
    warnings.simplefilter('ignore');import do_mpc
import benchmark as b
ROOT=Path(__file__).resolve().parent
DT=.025;SUB=5;HORIZON=80;Q=np.array([10.,10.,.1,.1]);QN=10*Q;R=1e-3;UMAX=20.;MAX_DURATION=20.;ENERGY_CAP=100.;CAP_PENALTY=1e4
def diagonal(q,E,sign):
    v=sign*np.sqrt((E-29.43*(1-np.cos(q)))/2.5);return np.array([q,q,3*v,2*v])
STARTS={'below':(np.array([.02,.02,0.,0.]),1),'above':(diagonal(np.pi,90.,1.),1),'down':(diagonal(90/22,90.,-1.),0)}

def plant(x,u):
    """One control interval: SUB RK4 steps of DT/SUB; returns the SUB new states."""
    out=[];z=x.copy()
    for _ in range(SUB):z=b.step(z,u,DT/SUB);out.append(z.copy())
    return out

def controller(target):
    centre=np.array([np.pi,np.pi,0.,0.]) if target==1 else np.zeros(4)
    m=do_mpc.model.Model('continuous','SX');x=m.set_variable('_x','x',shape=(4,1));u=m.set_variable('_u','u',shape=(2,1))
    c=ca.cos(x[0]-x[1]);d=2-c*c;v1=(x[2]-c*x[3])/d;v2=(2*x[3]-c*x[2])/d;a=ca.sin(x[0]-x[1])*v1*v2
    m.set_rhs('x',ca.vertcat(v1,v2,-a-2*b.G*ca.sin(x[0])+u[0],a-b.G*ca.sin(x[1])+u[1]))
    m.set_expression('H',.5*(x[2]*v1+x[3]*v2)+b.G*(3-2*ca.cos(x[0])-ca.cos(x[1])));m.setup()
    mpc=do_mpc.controller.MPC(m);s=mpc.settings
    s.n_horizon=HORIZON;s.t_step=DT;s.n_robust=0;s.state_discretization='collocation';s.collocation_type='radau';s.collocation_deg=3;s.collocation_ni=1;s.store_full_solution=False;s.supress_ipopt_output()
    e=ca.vertcat(2*(1-ca.cos(x[0]-centre[0])),2*(1-ca.cos(x[1]-centre[1])),(x[2]-centre[2])**2,(x[3]-centre[3])**2)
    mpc.set_objective(lterm=ca.dot(ca.DM(Q),e)+R*ca.dot(u,u),mterm=ca.dot(ca.DM(QN),e));mpc.set_rterm(u=0.)
    mpc.bounds['lower','_u','u']=-UMAX;mpc.bounds['upper','_u','u']=UMAX
    mpc.set_nl_cons('energy_cap',m.aux['H'],ub=ENERGY_CAP,soft_constraint=True,penalty_term_cons=CAP_PENALTY);mpc.setup();return mpc

def rollout(kind):
    x0,target=STARTS[kind];mpc=controller(target);mpc.x0=x0;mpc.u0=np.zeros(2);mpc.set_initial_guess()
    x=x0.copy();states=[x.copy()];fine=[x.copy()];inputs=[];start=time.perf_counter()
    for k in range(int(MAX_DURATION/DT)):
        u=np.clip(np.asarray(mpc.make_step(x.reshape(-1,1)),dtype=float).reshape(2),-UMAX,UMAX)
        fine+=plant(x,u);x=fine[-1].copy();states.append(x.copy());inputs.append(u)
        if (k+1)%4==0 and b.target_residual(x,target)<=0:break
    else:raise RuntimeError(f'{kind}: target not reached in {MAX_DURATION} s; final {x}')
    states=np.array(states);inputs=np.array(inputs);fine=np.array(fine);n=len(inputs)//4;assert len(fine)==SUB*len(inputs)+1 and np.array_equal(fine[::SUB],states)
    def chain(x,U):
        for u in U:x=plant(x,u)[-1]
        return x
    anchors=states[0:4*n:4].copy();coef=inputs.reshape(n,4,2);ends=np.array([chain(anchors[i],coef[i]) for i in range(n)])
    chain=float(np.abs(ends-states[4::4]).max());E=np.array([b.energy(s) for s in states])
    np.savez_compressed(ROOT/f'{kind}_original.npz',time=np.arange(n+1)*4*DT,anchors=anchors,endpoints=ends,coefficients=coef,duration=np.full(n,4*DT),states=states,inputs=inputs,fine_states=fine)
    V=np.array([b.score(s,target) for s in states]);first=int(np.argmax(V<=0))
    return dict(kind=kind,target='up' if target else 'down',start=x0.tolist(),start_energy=float(E[0]),duration=float(n*4*DT),snippets=n,max_abs_input=float(np.abs(inputs).max()),
        share_of_inputs_at_bound=float(np.mean(np.abs(inputs)>UMAX-1e-6)),energy_min=float(E.min()),energy_max=float(E.max()),final_state=states[-1].tolist(),final_energy=float(E[-1]),
        final_target_residual=float(b.target_residual(states[-1],target)),time_of_first_band_entry=float(first*DT),chained_endpoint_error=chain,
        energy_at_snippet_starts=[round(float(e),2) for e in E[0:4*n:4]],seconds=round(time.perf_counter()-start,1))

if __name__=='__main__':
    import sys
    report=dict(nmpc=dict(control_period=DT,horizon_steps=HORIZON,Q=Q.tolist(),Q_terminal=QN.tolist(),R=R,input_bound=UMAX,energy_cap=ENERGY_CAP,cap_penalty=CAP_PENALTY,discretization='radau collocation, degree 3',plant='RK4, step 0.005 s'),demonstrations=[])
    for kind in (sys.argv[1:] or ['below','above','down']):
        r=rollout(kind);report['demonstrations'].append(r);print(json.dumps(r),flush=True)
    if len(report['demonstrations'])==3:(ROOT/'demonstrations.json').write_text(json.dumps(report,indent=1)+'\n')
