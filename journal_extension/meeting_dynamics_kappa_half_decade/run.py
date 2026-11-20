"""Fixed-energy section diagnostic with half-decade gains (15 Nov 2026, evening).

Rerun of ../meeting_dynamics_kappa_log with the gains requested on 15 Nov 2026:
kappa in {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}.  Everything else is as in
that study (horizon 5000 s, 40 plotted times, H fixed, the same six initial
states and the same recorded streams); all 96 trajectories are computed anew.

Unchanged from ../meeting_dynamics_kappa10_1000: the model and force law of
diagnostic.py (byte-identical copy; u = a R v / max(1, |a| ||v||_inf / 20)),
H = 15 J, the 5 s passive prefix, the section q2 = 0 with p2 > 0, and RK4
with steps 1 ms (figure) and 2 ms (check).

Exploration follows the randomized protocol of the manuscript's definition,
as in ../meeting_double_transport_fix: at each switch the mode a is drawn
from {0, +kappa, -kappa} with probability 1/3 each and the duration uniformly
from [0.25, 0.75] s.  Steps are split exactly at the switches.  Each initial
state has one recorded stream, shared by all gains and both step sizes.

Initial states.  A scan of 89 section points (scan_passive.py) showed three
kinds of passive orbit over 5000 s: closed curves that visit 0.1-1% of the
cells, orbits confined to bounded regions that visit 1.5-7%, and orbits in
the chaotic sea that visit 21-28%.  The six states used here are of the
second kind (SECTION_POINTS), so the passive case covers a visible but
bounded part of the section.  Two earlier choices are kept for reference:
three regular plus three chaotic states (*_mixed_starts files) and six
closed curves around the period-one orbit (*_regular_starts files).
"""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,time
import numpy as np
from numba import njit
import diagnostic as d
ROOT=Path(__file__).resolve().parent
FROZEN=ROOT.parent/'meeting_dynamics_kappa10_1000'
GAINS=[0.,.001,.003,.01,.03,.1,.3,1.];HORIZON=5000.;STEPS={'fine':.001,'main':.002};PREFIX=5.;DWELL=(.25,.75)
SECTION_POINTS=[(.55,-3.75),(-.825,-2.25),(.275,-4.5),(-.825,-1.5),(0.,-4.5),(1.1,.75)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def initial_states():
    charts=np.array(SECTION_POINTS)
    return np.array([d.section_state(*c) for c in charts]),charts

def stream(seed):
    rng=np.random.default_rng(np.random.SeedSequence([20261115,5000,seed]));n=int(HORIZON/DWELL[0])+8
    return rng.choice(np.array([0.,1.,-1.]),size=n),rng.uniform(DWELL[0],DWELL[1],size=n)

@njit(cache=True)
def simulate(x0,horizon,h,kappa,signs,durations):
    """Crossings of q2 = 0 with p2 > 0: rows (t, q1, v1, p1, p2, v2, energy error)."""
    x=x0.copy();t=0.;events=np.zeros((int(horizon*6)+100,7));ne=0;pointer=0;segend=PREFIX;alpha=0.
    max_energy_error=0.;max_force=0.;switches=0
    while t<horizon-1e-12:
        if t>=segend-1e-12:
            alpha=kappa*signs[pointer];segend=t+durations[pointer];pointer+=1;switches+=1
        step=min(h,segend-t,horizon-t)
        u=d.control(x,alpha);max_force=max(max_force,abs(u[0]),abs(u[1]))
        xn=d.step(x,alpha,step)
        max_energy_error=max(max_energy_error,abs(d.energy(xn)-d.ENERGY))
        if x[1]*xn[1]<0. or (xn[1]==0. and x[1]!=0.):
            lo=0.;hi=step;left=x[1]
            for _ in range(30):
                mid=(lo+hi)/2.;ym=d.step(x,alpha,mid)
                if ym[1]*left>0.:lo=mid
                else:hi=mid
            y=d.step(x,alpha,(lo+hi)/2.);v1,v2=d.velocity(y)
            if y[3]>1e-9 and abs(v2)>1e-7:
                events[ne]=np.array((t+(lo+hi)/2.,y[0],v1,y[2],y[3],v2,d.energy(y)-d.ENERGY));ne+=1
        x=xn;t+=step
    return events[:ne],np.array((max_energy_error,max_force,float(switches),float(ne)))

def one(job):
    res,seed,j=job;file=ROOT/res/f'seed{seed}_g{j}.npz'
    if file.exists():return job
    signs,durations=stream(seed);x0=np.load(ROOT/'initial_states.npz')['states'][seed]
    events,metrics=simulate(x0,HORIZON,STEPS[res],GAINS[j],signs,durations)
    assert np.all(np.isfinite(events)) and metrics[1]<=20+1e-9
    tmp=file.with_suffix('.pending.npz');np.savez_compressed(tmp,events=events,metrics=metrics,gain=GAINS[j]);tmp.replace(file)
    return job

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    assert sha(ROOT/'diagnostic.py')==sha(FROZEN/'diagnostic.py')
    states,charts=initial_states();np.savez_compressed(ROOT/'initial_states.npz',states=states,section_points=charts)
    for r in STEPS:(ROOT/r).mkdir(exist_ok=True)
    simulate(states[0],6.,.002,.25,*stream(0))
    jobs=[(r,s,j) for r in STEPS for s in range(6) for j in range(len(GAINS))];t0=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k,job in enumerate(pool.map(one,jobs),1):print(k,'/',len(jobs),job,round(time.time()-t0),flush=True)
    (ROOT/'protocol.json').write_text(json.dumps(dict(gains=GAINS,horizon=HORIZON,steps=STEPS,passive_prefix=PREFIX,dwell=DWELL,modes='0, +kappa, -kappa with probability 1/3 each',
        stream='default_rng(SeedSequence([20261115, 5000, state]))',force='a R v / max(1, |a| ||v||_inf / 20)',section='q2=0, p2>0, |v2|>1e-7',
        initial_states='six section points (q1, v1) whose passive orbits stay in bounded regions and visit 3-7% of the cells in 5000 s (scan_passive.py)',section_points=SECTION_POINTS,sha256={f:sha(ROOT/f) for f in ['diagnostic.py','initial_states.npz','run.py']},
        note='Gains, horizon and initial states were chosen in four steps on 15 Nov 2026, each after the previous figure was seen (mixed starts; regular starts; bounded passive orbits with the gain 0.001 added and 100 dropped; then the half-decade gains 0.003, 0.03, 0.3 added and 10 dropped). The initial states were selected from the passive scan only. Not a preregistered study.'),indent=1)+'\n')
