"""Paired event-driven evaluation of one construction variant."""
from common import *
from histories import reconstruct
from fast_access import FastLibrary
from policy import rollout
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, time


def identities(variant):
    names=['protocol.json','evaluate.py','histories.py','common.py','construct.py','fast_access.py','policy.py']
    out={name:sha(ROOT/name) for name in names};out['library.npz']=sha(ROOT/variant/'library.npz');return out


def one_group(spec):
    variant,M,K,subdivisions,indices,label=spec
    protocol=check_sources();identity=identities(variant)
    path=ROOT/variant/f'{label}_M{M}_K{K}.npz'
    if path.exists():
        with np.load(path) as old:assert json.loads(str(old['metadata']))['identity']==identity
        return dict(variant=variant,M=M,K=K,label=label,reused=True)
    begin=time.perf_counter()
    payload=dict(np.load(ROOT/variant/'library.npz'));selected=library_for(payload,M,K)
    lib=FastLibrary(selected.full,M,True);baseline=FastLibrary(selected.full,M,False);del selected
    tlib=b.target_library();initials=np.load(PARENT/'initial_states.npz')['states']
    rows,records,states,values,gaps,audits=[],[],[],[],[],[]
    for index in indices:
        row,actions=rollout(initials[index],lib,baseline,tlib,protocol['horizon'],subdivisions,save_actions=True)
        row.update(M=M,K=K,index=index);rows.append(row);records.extend([[index]+a for a in actions])
        if M==3:
            state,value,gap,audit=reconstruct(initials[index],row,actions,lib.full,subdivisions,output_times())
            states.append(state);values.append(value);gaps.append(gap);audits.append(audit)
    metadata=dict(variant=variant,M=M,K=K,label=label,indices=indices,subdivisions=subdivisions,identity=identity,seconds=time.perf_counter()-begin,
        action_columns=['initial_index','start_time','preceding_wait','executed_duration','assignment_index_in_K_library','start_radius_ratio','V_start','V_end','q_start','p_start','q_end','p_end','target_hit','full_primitive'])
    save_npz(path,metadata=np.array(json.dumps(metadata)),rows=np.array(json.dumps(rows)),
        records=np.asarray(records,dtype=float).reshape(-1,14),states=np.asarray(states),signed_V=np.asarray(values),
        target_gap=np.asarray(gaps),history_audits=np.asarray(audits),times=output_times())
    return dict(variant=variant,M=M,K=K,label=label,success=sum(r['success'] for r in rows),seconds=round(metadata['seconds'],1))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('variants',nargs='+',choices=VARIANTS)
    parser.add_argument('--mode',choices=['main','refinement','m3'],default='main');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();protocol=check_sources();specs=[]
    for v in args.variants:
        if args.mode=='refinement':
            r=protocol['refinement'];specs+=[(v,g['M'],g['K'],r['subdivisions'],r['initial_indices'],'refinement') for g in r['groups']]
        else:
            groups=[g for g in protocol['evaluation_groups'] if args.mode=='main' or g['M']==3]
            specs+=[(v,g['M'],g['K'],4,list(range(500)),'main') for g in groups]
    specs.sort(key=lambda s:(s[1]!=3,s[1],s[2]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one_group,s) for s in specs]):print(json.dumps(future.result()),flush=True)

if __name__=='__main__':main()
