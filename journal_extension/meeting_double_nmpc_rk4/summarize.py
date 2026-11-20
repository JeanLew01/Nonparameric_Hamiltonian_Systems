"""Summary of the double-pendulum policy comparison with the NMPC demonstrations (first HCP_N initial states, default 72)."""
from pathlib import Path
import json,math,os
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'meeting_double_transport_fix'
PRIMARY=['D','D+E','D+T8','D+E+T8'];SWEEP=['D+E','D+E+T2','D+E+T4','D+E+T8'];N=int(os.environ.get('HCP_N','72'));HORIZON=10000.;CHECKPOINTS=[100.,300.,1000.,3000.,10000.]
BAND={'up':(58.66,59.06),'down':(0.,.40)}

def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def load(name,arm):
    """Per-trial metrics, final positive residual, action waits, and the mean residual curve (the 0.1 s scores are not kept)."""
    M=[];final=[];waits=[];mean=None
    for i in range(N):
        with np.load(ROOT/'results'/'main'/'main'/name/arm/f'{i}.npz') as d:
            s=np.maximum(d['score'].astype(np.float64),0);mean=s if mean is None else mean+s;M.append(d['metrics'].copy());final.append(float(s[-1]));a=d['actions']
            if len(a)>1:waits+=list(a[1:,4]-a[:-1,8])
    return np.array(M),np.array(final),np.array(waits),mean/N

def main():
    summary={};T={};H={};curves={}
    for name in ('up','down'):
        for arm in PRIMARY+['D+E+T2','D+E+T4']:
            if not all((ROOT/'results'/'main'/'main'/name/arm/f'{i}.npz').exists() for i in range(N)):print('incomplete, skipped:',name,arm);continue
            M,final,waits,mean=load(name,arm);hit=M[:,0]>0;acted=M[:,2]>=0;T[name,arm]=M[:,1];H[name,arm]=hit;time=np.arange(len(mean))*.1
            p=np.searchsorted(np.sort(M[hit,1]),time,side='right')/N;curves[f'{name}_{arm}_V']=mean[::10];curves[f'{name}_{arm}_success']=p[::10]
            stuck=~hit&acted
            summary[f'{name}/{arm}']=dict(n=N,hits=int(hit.sum()),success_interval_95=wilson(int(hit.sum()),N),restricted_mean_time=float(M[:,1].mean()),
                final_mean_positive_residual=float(final.mean()),never_act=int((~acted).sum()),never_act_and_hit=int((~acted&hit).sum()),
                median_first_action_time=float(np.median(M[acted,2])) if acted.any() else None,
                median_wait_between_actions=float(np.median(waits)) if len(waits) else None,
                acted_without_hit=int(stuck.sum()),median_final_residual_acted_without_hit=float(np.median(final[stuck])) if stuck.any() else None,
                mean_final_residual_never_act=float(final[~acted&~hit].mean()) if (~acted&~hit).any() else None,
                actions_increasing_V=int(M[:,6].sum()),starts_outside_support=int(M[:,7].sum()),safety_exits=int(np.sum(M[:,13]<0)),max_energy_work_residual=float(np.max(np.abs(M[:,12]))),
                hits_by={str(int(c)):int(np.sum(hit&(M[:,1]<=c))) for c in CHECKPOINTS},
                residual_at={str(int(c)):float(mean[int(round(c/.1))]) for c in CHECKPOINTS})
    paired={};rng=np.random.default_rng(20261115);idx=rng.integers(0,N,size=(5000,N))
    for name in ('up','down'):
        for a in ['D','D+E','D+T8']:
            if (name,a) in T and (name,'D+E+T8') in T:
                diff=T[name,'D+E+T8']-T[name,a];boot=diff[idx].mean(1);g=int(np.sum(H[name,'D+E+T8']&~H[name,a]));l=int(np.sum(H[name,a]&~H[name,'D+E+T8']))
                paired[f'{name}: D+E+T8 vs {a}']=dict(gained=g,lost=l,restricted_mean_time_difference=float(diff.mean()),bootstrap_95=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))])
    agree={}
    init=np.load(ROOT/'initial_states.npz');energies=init['energies'][:N]
    by_energy={f'{name}/{arm}':{str(int(E)):[int(H[name,arm][energies==E].sum()),int((energies==E).sum())] for E in np.unique(energies)} for (name,arm) in H}
    (ROOT/f'summary_{N}.json').write_text(json.dumps(dict(n=N,horizon=HORIZON,summary=summary,paired=paired,hits_by_energy=by_energy),indent=1)+'\n')
    np.savez_compressed(ROOT/f'mean_curves_{N}.npz',time=np.arange(len(next(iter(curves.values()))))*1.,**curves)
    for k,v in summary.items():
        print(f"{k:14s} hits {v['hits']:3d} Tc {v['restricted_mean_time']:7.1f} V {v['final_mean_positive_residual']:6.2f} never {v['never_act']:3d} first {v['median_first_action_time']} wait {v['median_wait_between_actions']} stuck {v['acted_without_hit']} ({v['median_final_residual_acted_without_hit']}) incr {v['actions_increasing_V']} outside {v['starts_outside_support']} exits {v['safety_exits']} by {v['hits_by']} res { {a:round(x,1) for a,x in v['residual_at'].items()} }")
    for k,v in paired.items():print(k,v)
    for k,v in by_energy.items():print('by energy',k,v)

if __name__=='__main__':main()
