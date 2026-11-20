"""Step-halving check: the refinement runs (8 subdivisions) against the main runs (4) on the same trials."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol.json').read_text())['refinement'];trials=changed=0;worst=0.;groups=[]
for g in protocol['groups']:
    M,K=g['M'],g['K'];main={r['index']:r for r in json.loads(str(np.load(ROOT/'direct_fiber'/f'main_M{M}_K{K}.npz')['rows']))}
    fine=json.loads(str(np.load(ROOT/'direct_fiber'/f'refinement_M{M}_K{K}.npz')['rows']))
    c=sum(r['success']!=main[r['index']]['success'] for r in fine);d=max([abs(r['time']-main[r['index']]['time']) for r in fine if r['success'] and main[r['index']]['success']],default=0.)
    trials+=len(fine);changed+=c;worst=max(worst,d);groups.append(dict(M=M,K=K,trials=len(fine),labels_changed=c,max_reaching_time_difference=d))
out=dict(trials=trials,labels_changed=changed,max_reaching_time_difference=worst,groups=groups)
(ROOT/'refinement.json').write_text(json.dumps(out,indent=1)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='groups'}))
