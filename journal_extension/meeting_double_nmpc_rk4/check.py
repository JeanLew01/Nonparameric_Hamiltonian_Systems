"""Independent replay of every executed action of the NMPC double-pendulum study (500 states).

Adapted from ../meeting_double_transport_fix/check.py.  Each recorded action
start is replayed with SciPy DOP853 (no project integrator), with an event at
the band centre E_* that reproduces the stop rule.  Checked for every action
that the executor finished:
  * the start lies in the ball and in the energy window of its library row,
  * V at the replayed end equals the recorded value,
  * V decreased, by at least 0.001 times the elapsed time.
Actions cut by physical target entry have no end record and are only checked
for support membership.  One process per (target, policy).
"""
import concurrent.futures,json,os
import numpy as np
from scipy.integrate import solve_ivp
import benchmark as b
ROOT=b.ROOT
ARMS=['D','D+E','D+T8','D+E+T8','D+E+T2','D+E+T4'];CENTRE={'down':.2,'up':58.86};N=int(os.environ.get('HCP_N','500'))

def library(name,arm):
    T=int(arm[-1]) if arm[-1].isdigit() else 0
    return np.load(ROOT/'reuse'/f'{name}_library_T{T}.npz')

def replay(x,coef,tau,centre):
    side=1. if b.energy(x)<centre else -1.;dt=tau/len(coef);t=0.;y=np.array(x,float)
    def cross(_,z):return side*(centre-b.energy(z))
    cross.terminal=True;cross.direction=-1
    for u in coef:
        s=solve_ivp(lambda _,z:b.rhs(z,u),[0,dt],y,method='DOP853',rtol=1e-11,atol=1e-12,max_step=.01,events=cross)
        y=s.y[:,-1];t+=s.t[-1]
        if s.status==1:return y,t,True
    return y,t,False

def one(job):
    name,arm=job;target=0 if name=='down' else 1;lib=library(name,arm);coef=lib['coefficients'];dur=lib['durations'];elo=lib['energy_lo'];ehi=lib['energy_hi']
    total=finished=stopped=0;worst=dict(end_V_difference=0.,membership_ratio=0.,min_drop=np.inf,min_rate_margin=np.inf);bad=[]
    for i in range(N):
        with np.load(ROOT/'results'/'main'/'main'/name/arm/f'{i}.npz') as d:acts=d['actions'];rows=d['library_rows']
        for a in acts:
            if not np.isfinite(a[4]):continue
            j=rows[int(a[5])];E=float(b.energy(a[:4]));total+=1
            worst['membership_ratio']=max(worst['membership_ratio'],float(a[6]))
            if not (a[6]<=1+1e-12 and elo[j]-1e-6<=E<=ehi[j]+1e-6):bad.append(dict(target=name,arm=arm,trial=i,t=float(a[4]),reason='outside support'))
            if not np.isfinite(a[9]):continue
            y,t,cut=replay(a[:4],coef[j],float(dur[j]),CENTRE[name]);V=float(b.score(y,target));finished+=1;stopped+=cut
            diff=abs(V-a[9]);drop=a[7]-V;worst['end_V_difference']=max(worst['end_V_difference'],float(diff))
            worst['min_drop']=min(worst['min_drop'],float(drop));worst['min_rate_margin']=min(worst['min_rate_margin'],float(drop-.001*t))
            # Two separate findings: the replayed end value differs from the recorded one (integrator agreement), or V did not decrease at the required rate.
            if diff>1e-4:bad.append(dict(target=name,arm=arm,trial=i,t=float(a[4]),reason='end value differs',replayed_V=V,recorded_V=float(a[9]),start_V=float(a[7])))
            if drop<=.001*t-1e-9:bad.append(dict(target=name,arm=arm,trial=i,t=float(a[4]),reason='progress',replayed_V=V,recorded_V=float(a[9]),start_V=float(a[7])))
    return dict(key=f'{name}/{arm}',actions=total,finished=finished,stopped=int(stopped),worst=worst,violations=bad)

def main():
    jobs=[(name,arm) for name in ('up','down') for arm in ARMS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('HCP_WORKERS','12'))) as pool:res=list(pool.map(one,jobs))
    worst=dict(end_V_difference=max(r['worst']['end_V_difference'] for r in res),membership_ratio=max(r['worst']['membership_ratio'] for r in res),
               min_drop=min(r['worst']['min_drop'] for r in res),min_rate_margin=min(r['worst']['min_rate_margin'] for r in res))
    bad=[v for r in res for v in r['violations']]
    report=dict(states=N,actions=sum(r['actions'] for r in res),finished_actions_replayed=sum(r['finished'] for r in res),stopped_at_centre=sum(r['stopped'] for r in res),
                per_policy={r['key']:r['actions'] for r in res},worst=worst,violations=bad)
    (ROOT/f'independent_check_{N}.json').write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='violations'},indent=1));print('violations',len(bad),bad[:5]);print('Complete',flush=True)

if __name__=='__main__':main()
