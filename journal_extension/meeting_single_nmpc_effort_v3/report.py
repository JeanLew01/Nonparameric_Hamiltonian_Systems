"""Time decomposition, success counts and paired bootstrap intervals of the effort-minimal NMPC study."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;GROUPS=[(1,1),(1,8),(2,1),(2,8),(3,1),(3,2),(3,4),(3,8),(4,1),(4,8),(5,1),(5,8)]
HORIZONS=(2,5,10,15,20,30,40,50,60,80,100,150,200,2000)

def group(path):
    d=np.load(path);rows=json.loads(str(d['rows']));R=d['records'];idx=R[:,0].astype(int);inter=[];term=[]
    lib_transport=None
    for r in rows:
        a=R[idx==r['index']]
        if len(a):inter.append(a[1:,2].sum());term.append(r['time']-(a[-1,1]+a[-1,3]))
    acted=[r for r in rows if r['first_action'] is not None];ok=[r for r in rows if r['success']]
    return dict(successes=len(ok),mean_time_successes=float(np.mean([r['time'] for r in ok])),median_time_successes=float(np.median([r['time'] for r in ok])),max_time_successes=float(np.max([r['time'] for r in ok])),
        active=float(np.mean([r['active'] for r in ok])),first_wait=float(np.mean([r['first_action'] for r in acted])),first_wait_states=len(acted),
        waits_between_actions=float(np.mean(inter)),terminal_wait=float(np.mean(term)),actions=float(np.mean([r['actions'] for r in rows])),total_actions=int(sum(r['actions'] for r in rows)),
        transported_actions=float(np.mean([r['transport_actions'] for r in rows])),total_transported_actions=int(sum(r['transport_actions'] for r in rows)),guard_rejections=int(sum(r['guard_rejections'] for r in rows)),
        effort=float(np.mean([r['effort'] for r in rows])),max_energy=float(max(r['max_energy'] for r in rows)),hits_by={str(T):int(sum(r['success'] and r['time']<=T for r in rows)) for T in HORIZONS},
        reasons={k:int(sum(r['reason']==k for r in rows)) for k in sorted({r['reason'] for r in rows})},
        times=np.array([r['time'] if r['success'] else np.inf for r in rows]),first=np.array([np.nan if r['first_action'] is None else r['first_action'] for r in rows]))

def main():
    out={}
    for M,K in GROUPS:
        path=ROOT/'direct_fiber'/f'main_M{M}_K{K}.npz'
        if path.exists():out[M,K]=group(path)
    rng=np.random.default_rng(20261116);idx=rng.integers(0,500,size=(5000,500));paired={}
    for M in (1,2,3,4,5):
        if (M,1) in out and (M,8) in out:
            a=out[M,1];c=out[M,8];both=np.isfinite(a['times'])&np.isfinite(c['times'])
            same=bool(np.array_equal(np.isfinite(a['times']),np.isfinite(c['times'])))
            d=np.where(both,c['times']-a['times'],0.);boot=np.array([d[i][both[i]].mean() for i in idx])
            f=np.isfinite(a['first'])&np.isfinite(c['first']);df=np.where(f,c['first']-a['first'],0.);bf=np.array([df[i][f[i]].mean() for i in idx])
            paired[M]=dict(successes=int(both.sum()),same_success_set=same,time_change=float(d[both].mean()),time_bootstrap_95=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
                first_wait_change=float(df[f].mean()),first_wait_bootstrap_95=[float(np.quantile(bf,.025)),float(np.quantile(bf,.975))],slower_with_transport=int(np.sum(d[both]>1e-9)),faster_with_transport=int(np.sum(d[both]<-1e-9)))
    for (M,K),g in out.items():
        print('M=%d K=%d | hits %3d | time mean %.2f median %.2f max %.1f = active %.2f + first wait %.3f + between %.2f + terminal %.2f | actions %.0f transported %.0f guard %d | effort %.1f | by %s'%(
            M,K,g['successes'],g['mean_time_successes'],g['median_time_successes'],g['max_time_successes'],g['active'],g['first_wait'],g['waits_between_actions'],g['terminal_wait'],g['actions'],g['transported_actions'],g['guard_rejections'],g['effort'],{k:v for k,v in g['hits_by'].items() if k in ('5','10','20','30','50','100','2000')}))
    for M,p in paired.items():print('paired K=8 minus K=1, M=%d:'%M,p)
    (ROOT/'report.json').write_text(json.dumps(dict(groups={f'M{M}/K{K}':{k:v for k,v in g.items() if k not in ('times','first')} for (M,K),g in out.items()},paired_K8_minus_K1={str(M):p for M,p in paired.items()}),indent=1)+'\n')

if __name__=='__main__':main()
