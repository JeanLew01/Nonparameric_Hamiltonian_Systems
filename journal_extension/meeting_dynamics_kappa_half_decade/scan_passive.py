"""Passive coverage (64x128 grid, 1 ms step) of 89 section points at 1000, 2500 and 5000 s; used to choose the initial states of run.py."""
import sys,json,concurrent.futures
import numpy as np
SRC=str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0,SRC)
import run as R, summarize as Z, diagnostic as d
pts=[]
for q in np.linspace(-1.1,1.1,9):
    for v in np.linspace(-4.5,4.5,13):
        try:d.section_state(q,v);pts.append((float(q),float(v)))
        except ValueError:pass
def one(k):
    x=d.section_state(*pts[k]);ev,m=R.simulate(x,5000.,.001,0.,np.zeros(4),np.ones(4))
    f=Z.first_visit(ev,(64,128));ok=Z.accessible((64,128)).sum()
    return k,[float((f<=T).sum()/ok) for T in (1000,2500,5000)],len(ev)
if __name__=='__main__':
    out=[]
    with concurrent.futures.ProcessPoolExecutor(14) as p:
        for k,c,n in p.map(one,range(len(pts))):out.append(dict(point=pts[k],coverage=c,crossings=n))
    json.dump(out,open(SRC+'/scan_passive.json','w'))
    for o in sorted(out,key=lambda o:o['coverage'][2]):print('q1 %+.3f v1 %+.3f  cov@1000 %5.2f @2500 %5.2f @5000 %5.2f  crossings %d'%(*o['point'],*[100*c for c in o['coverage']],o['crossings']))
