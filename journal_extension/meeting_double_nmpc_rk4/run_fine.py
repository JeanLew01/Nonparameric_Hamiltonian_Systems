"""Step halving for the NMPC double-pendulum study (15 Nov 2026).

The four primary policies on both targets are rerun with half the integration
step (0.00125 s) on the twelve predeclared states 0, 6, ..., 66, the subset of
../meeting_double_transport_fix/run.py.  Everything else equals run.py.
The comparison with the main step is written to refinement.json.
"""
import argparse,concurrent.futures,json,time
import numpy as np
import execute
from execute import ROOT
LIBRARY=str(ROOT/'reuse');PRIMARY=['D','D+E','D+T8','D+E+T8'];SUBSET=list(range(0,72,6));KAPPA=1.

def jobs():
    return [dict(study='main',library=LIBRARY,target=target,arm=arm,clock='fine',ids=SUBSET[k:k+2],horizon=10000.,protocol=1,seed=0,interrupt=True,kappa=KAPPA,guard=True)
            for arm in PRIMARY for target in (0,1) for k in range(0,len(SUBSET),2)]

def compare():
    out={};changed=0;total=0
    for name in ('up','down'):
        for arm in PRIMARY:
            fine=[np.load(ROOT/'results'/'main'/'fine'/name/arm/f'{i}.npz')['metrics'] for i in SUBSET]
            main=[np.load(ROOT/'results'/'main'/'main'/name/arm/f'{i}.npz')['metrics'] for i in SUBSET]
            hf=np.array([m[0]>0 for m in fine]);hc=np.array([m[0]>0 for m in main]);changed+=int(np.sum(hf!=hc));total+=len(SUBSET)
            out[f'{name}/{arm}']=dict(hits_main_step=int(hc.sum()),hits_half_step=int(hf.sum()),labels_changed=int(np.sum(hf!=hc)),
                restricted_mean_time_main=float(np.mean([m[1] for m in main])),restricted_mean_time_half=float(np.mean([m[1] for m in fine])),
                actions_increasing_V_half_step=int(sum(m[6] for m in fine)),starts_outside_support_half_step=int(sum(m[7] for m in fine)))
    report=dict(states=SUBSET,trials=total,labels_changed=changed,groups=out)
    (ROOT/'refinement.json').write_text(json.dumps(report,indent=1)+'\n');print(json.dumps(report,indent=1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=14);args=ap.parse_args()
    J=jobs();print(json.dumps(dict(jobs=len(J),trials=sum(len(j['ids']) for j in J))),flush=True);t0=time.monotonic();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in concurrent.futures.as_completed([pool.submit(execute.run_job,j) for j in J]):
            r=result.result();done+=len(r['ids']);print(json.dumps(dict(**r,done=done,elapsed=round(time.monotonic()-t0,1))),flush=True)
    compare();print('Complete',done,flush=True)

if __name__=='__main__':main()
