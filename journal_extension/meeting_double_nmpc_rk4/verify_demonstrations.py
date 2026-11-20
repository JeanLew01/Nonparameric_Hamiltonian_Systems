"""Replay check of the three NMPC demonstrations of the double pendulum.

Each stored input signal is replayed open loop from the stored initial state
(a) with the fixed-step RK4 of the policy runs (0.0025 s), (a') with RK4 at
one eighth of the simulation step (0.000625 s), and (b) with SciPy
DOP853 (rtol 1e-13), an independent adaptive integrator; the demonstrations
themselves were simulated with RK4 at 0.005 s.  Reported: where the
replay ends relative to the physical target (negative residual = inside),
the value of V there, and the largest path energy.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
import benchmark as b
out={}
for kind,target in [('below',1),('above',1),('down',0)]:
    o=np.load(b.ROOT/f'{kind}_original.npz');X=o['states'];U=o['inputs'];T=.025*len(U)
    z,mx=b.rollout(X[0].copy(),U,T,.0025)
    w,mw=b.rollout(X[0].copy(),U,T,.000625)   # one eighth of the simulation step, as in the single-pendulum replay
    y=X[0].copy();my=b.energy(y)
    for u in U:
        s=solve_ivp(lambda t,w:b.rhs(w,u),[0,.025],y,method='DOP853',rtol=1e-13,atol=1e-14,max_step=.005);y=s.y[:,-1];my=max(my,max(b.energy(v) for v in s.y.T))
    out[kind]=dict(target='up' if target else 'down',duration=T,control_intervals=len(U),
        rk4_final_target_residual=float(b.target_residual(z,target)),rk4_final_V=float(b.score(z,target)),rk4_final_energy=float(b.energy(z)),rk4_max_energy=float(mx),
        dop853_final_target_residual=float(b.target_residual(y,target)),dop853_final_V=float(b.score(y,target)),dop853_final_energy=float(b.energy(y)),dop853_max_energy=float(my),
        rk4_minus_stored=float(np.abs(b.difference(z.copy(),X[-1].copy())).max()),dop853_minus_stored=float(np.abs(b.difference(y.copy(),X[-1].copy())).max()),
        rk4_eighth_final_target_residual=float(b.target_residual(w,target)),rk4_eighth_final_V=float(b.score(w,target)),rk4_eighth_max_energy=float(mw),rk4_eighth_minus_stored=float(np.abs(b.difference(w.copy(),X[-1].copy())).max()),
        max_abs_input=float(np.abs(U).max()),inside_target_both=bool(b.target_residual(z,target)<0 and b.target_residual(y,target)<0),inside_target_all=bool(max(b.target_residual(z,target),b.target_residual(y,target),b.target_residual(w,target))<0))
(b.ROOT/'demonstration_replay.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out,indent=1))
