"""Replay check and push statistics of the five effort-minimal NMPC demonstrations.

Each stored input signal is replayed open loop from the stored initial state
with RK4 at one eighth of the generation step (0.625 ms instead of 5 ms).
Reported: where the replay first enters the physical target and where it is at
the stored end time (negative gap = inside), V there, the largest path energy
(the operating region is H <= 168 J), and whether V ever increases along the
stored trajectory.  For the pushes: their number, the smallest ratio |p|/pi at
a sample with full torque (the expert should push only where it is at least
eta = 0.95), and the fraction of time with nonzero input.
"""
from common import *
out={}
d=np.load(ROOT/'demonstrations.npz')
for j in range(5):
    X=d[f'states_{j}'];U=d[f'controls_{j}'][:,0];dt=float(d[f'dt_{j}'])
    y=X[0].copy();maxe=b.energy(y);first=None
    for k,u in enumerate(U):
        for _ in range(8):
            y=b.step(y,u,dt/8);maxe=max(maxe,b.energy(y))
        if first is None and b.target_gap(y)<=0:first=(k+1)*dt
    E=np.array([b.energy(x) for x in X]);V=np.array([b.score(x) for x in X]);peak=np.sqrt(2*b.I*E)
    full=np.flatnonzero((np.abs(U)>1.999)&(E[:-1]>1e-6));ratio=np.abs(X[full,1])/peak[full]
    starts=np.flatnonzero((U!=0)&(np.r_[0.,U[:-1]]==0));ends=np.flatnonzero((U!=0)&(np.r_[U[1:],0.]==0))+1
    positive=V[:-1]>0
    out[str(d['names'][j])]=dict(start=X[0].tolist(),duration=len(U)*dt,control_period=float(d[f'control_period_{j}']),max_abs_input=float(abs(U).max()),
        replay_end_target_gap=float(b.target_gap(y)),replay_end_V=float(b.score(y)),replay_end_energy=float(b.energy(y)),replay_first_target_entry=first,replay_max_energy=float(maxe),stored_max_energy=float(E.max()),
        replay_minus_stored=float(b.distance(y,X[-1])),inside_target=bool(b.target_gap(y)<0 and b.score(y)<=0),inside_operating_region=bool(maxe<=b.HX),
        pushes=int(len(starts)),active_time=float(np.count_nonzero(U)*dt),active_fraction_until_V0=float(np.count_nonzero(U[positive])/max(1,positive.sum())),
        time_V_reaches_0=float(np.argmax(V<=0)*dt),largest_V_increase_per_step=float(np.max(np.diff(V)[positive[:len(V)-1]])) if positive.any() else 0.,
        min_momentum_ratio_at_full_torque=float(np.nanmin(ratio)) if len(ratio) else None,push_duration_range=[float((ends-starts).min()*dt),float((ends-starts).max()*dt)],
        impulse=float(np.abs(U).sum()*dt),squared_effort=float((U**2).sum()*dt))
(ROOT/'demonstration_replay.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps(out,indent=1))
