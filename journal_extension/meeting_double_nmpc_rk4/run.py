"""Double-pendulum policy comparison with the NMPC demonstrations (RK4 version, 16 Nov 2026).

Same executor, stopping rule, guard, horizon (10,000 s), exploration protocol
(kappa = 1) and per-trial streams as ../meeting_double_transport_fix.  The
libraries are those of construct.py in this folder.  The initial states are the
500 states of ../meeting_double_500 (the first 72 are the earlier 72 states);
--first and --last select a range, so the first 72 can be run before the rest.
"""
import argparse,concurrent.futures,json,time
import execute
from execute import ROOT
LIBRARY=str(ROOT/'reuse')
PRIMARY=['D','D+E','D+T8','D+E+T8'];SWEEP=['D+E+T2','D+E+T4'];KAPPA=1.

def jobs(arms,ids):
    return [dict(study='main',library=LIBRARY,target=target,arm=arm,clock='main',ids=ids[k:k+4],horizon=10000.,protocol=1,seed=0,interrupt=True,kappa=KAPPA,guard=True)
            for arm in arms for target in (0,1) for k in range(0,len(ids),4)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=14);ap.add_argument('--arms',default=','.join(PRIMARY+SWEEP));ap.add_argument('--first',type=int,default=0);ap.add_argument('--last',type=int,default=72);args=ap.parse_args()
    J=jobs(args.arms.split(','),list(range(args.first,args.last)));print(json.dumps(dict(jobs=len(J),trials=sum(len(j['ids']) for j in J),workers=args.workers)),flush=True);t0=time.monotonic();done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in concurrent.futures.as_completed([pool.submit(execute.run_job,j) for j in J]):
            r=result.result();done+=len(r['ids']);print(json.dumps(dict(**r,done=done,elapsed=round(time.monotonic()-t0,1))),flush=True)
    print('Complete',done,flush=True)

if __name__=='__main__':main()
