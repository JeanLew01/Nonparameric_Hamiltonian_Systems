"""Recompute every number quoted in Section 5 and in the numerical part of the
supplement from the stored runs.

Each entry is a sentence fragment of the manuscript with its numbers filled in
from the study folders (meeting_single_nmpc_effort_v3, meeting_double_nmpc_rk4,
meeting_dynamics_kappa_half_decade).  Without a manuscript the statements are
printed; with one, every fragment must occur verbatim in it (white space
collapsed):

    python verify_nmpc_text.py
    HCP_MAIN_TEX=/path/to/main.tex python verify_nmpc_text.py
"""
from pathlib import Path
import json,math,os,re,sys
import numpy as np
HERE=Path(__file__).resolve().parent
TEX=Path(os.environ['HCP_MAIN_TEX']) if os.environ.get('HCP_MAIN_TEX') else None
text=re.sub(r'\s+',' ',TEX.read_text()) if TEX else None   # None: no manuscript given, the statements are only recomputed and printed
checks=[]
def claim(fragment):
    fragment=re.sub(r'\s+',' ',fragment);checks.append((fragment,text is None or fragment in text))
def big(n):
    n=int(n);return f'{n:,}'.replace(',','{,}') if n>=10000 else str(n)
def r(x,n):return f'{x:.{n}f}'

# ------------------------------------------------------------------ single pendulum (effort-minimal NMPC demonstrations, 16 Nov 2026)
S=HERE/'meeting_single_nmpc_effort_v3';sys.path.insert(0,str(HERE/'reference_single'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(S/'_cache'))
import benchmark as bs
d=np.load(S/'demonstrations.npz');fp=json.loads(str(d['fingerprint']));rep=json.loads((S/'demonstration_replay.json').read_text());names=[str(x) for x in d['names']]
assert fp['u_expert']==2. and fp['eta']==.95 and fp['weight_growth']==.05 and fp['horizon']==100 and fp['control_period']==.02 and fp['sim_dt']==.005 and fp['zero_below']==1e-4 and fp['energy_eps']==1e-3 and abs(fp['E_star']-(bs.ELO+bs.EHI)/2)<1e-12
E0=[float(bs.energy(d[f'states_{j}'][0])) for j in range(5)];assert abs(E0[0]-160)<1e-9 and abs(E0[1]-160)<1e-9 and E0[2]==0 and abs(d['states_3'][0][0]-1.5)<1e-12 and abs(d['states_4'][0][0]+1.5)<1e-12 and d['states_3'][0][1]==0 and d['states_4'][0][1]==0
assert fp.get('torque_sign') is None;sgn=[np.sign(d[f'controls_{j}'][:,0]) for j in range(5)];assert set(sgn[2])<={0.,1.} and set(sgn[3])<={0.,1.} and set(sgn[4])<={0.,-1.},'one-sided torque in the libration demonstrations'
claim('The last two start at rest at $q=1.5$ and $q=-1.5$~rad, where $H=%s$~J'%r(E0[3],1));assert r(E0[3],1)==r(E0[4],1)
dur=[rep[n]['duration'] for n in names];push=[rep[n]['pushes'] for n in names]
assert r(dur[0],1)==r(dur[1],1) and r(dur[3],1)==r(dur[4],1) and push[0]==push[1] and push[3]==push[4]
claim('The two rotation demonstrations take $%s$~s with $%d$ pushes each, the swing-up from the bottom takes $%s$~s with $%d$ pushes, and the last two take $%s$~s with $%d$ pushes each'%(r(dur[0],1),push[0],r(dur[2],1),push[2],r(dur[3],1),push[3]))
lo=min(rep[n]['push_duration_range'][0] for n in names);hi=max(rep[n]['push_duration_range'][1] for n in names)
pass
for j in range(5):
    X_=d[f'states_{j}'];U_=d[f'controls_{j}'][:,0];V_=np.array([bs.score(x) for x in X_]);dV_=np.diff(V_);pos_=V_[:-1]>0
    assert dV_[(U_!=0)&pos_].max()<0 and np.abs(dV_[(U_==0)&pos_]).max()<2e-9
pass
ratio=min(v['min_momentum_ratio_at_full_torque'] for v in rep.values() if v['min_momentum_ratio_at_full_torque'] is not None)
claim('full torque occurs only where $|p|\\ge%s\\,\\pi(x)$'%r(math.floor(ratio*1000)/1000,3))
claim('$v_0=10^{-3}$, $L=%s$, and $L_H=%s$ on $X=\\{H\\le%d~\\mathrm{J}\\}$'%(r(bs.L,2),r(bs.LV,2),bs.HX));assert bs.V0==1e-3
lib=np.load(S/'original_library.npz');counts=np.bincount(lib['demo_ids'],minlength=5)
claim('covers the pushes with $%s$ short'%big(len(lib['radii'])))
claim('demonstrations give $%s$, $%s$, $%s$, $%s$, and $%s$ assignments'%tuple(big(c) for c in counts))
durations=lib['leads']+(np.diff(lib['offsets'])-1)*float(lib['dt']);claim('with durations up to $%s$~s'%r(durations.max(),3))
assert np.all(lib['control_flat'][lib['offsets'][:-1]]!=0),'every stored snippet begins with a nonzero input'
def rows(M,K):
    f=np.load(S/'direct_fiber'/f'main_M{M}_K{K}.npz');return json.loads(str(f['rows'])),f['records']
R={(M,K):rows(M,K) for M in range(1,6) for K in (1,8)};R[3,2]=rows(3,2);R[3,4]=rows(3,4)
T={k:np.array([x['time'] if x['success'] else np.inf for x in v[0]]) for k,v in R.items()}
F={k:np.array([np.nan if x['first_action'] is None else x['first_action'] for x in v[0]]) for k,v in R.items()}
hit=lambda M,K,t:int((T[M,K]<=t).sum())
assert all(np.array_equal(T[M,1]<=120,T[M,8]<=120) for M in range(1,6)) and [hit(M,1,120) for M in (3,4,5)]==[500,500,500] and hit(1,1,120)<hit(2,1,120)<500,'paired labels of D and D+T coincide at 120 s';claim('At $T=120$~s their paired trial labels coincide for every $M$, with all $500$ successes for $M\\ge3$')
assert [hit(M,8,20) for M in range(1,6)]==[hit(M,8,120) for M in range(1,6)];claim('At $T=20$~s, D+T has already reached all of them and D at most $%d$ of the $500$'%max(hit(M,1,20) for M in range(1,6)))
claim('The runs were continued to $2000$~s, and no trial enters the target after $120$~s');assert [hit(M,K,120) for M in range(1,6) for K in (1,8)]==[hit(M,K,2000) for M in range(1,6) for K in (1,8)]
init=np.load(HERE/'reference_single'/'initial_states.npz')['states'];E=np.array([bs.energy(x) for x in init])
g4=T[4,1]-T[3,1];g5=T[5,1]-T[4,1];assert g4.max()<=1e-9 and g5.max()<=1e-9,'no state becomes slower under D'
assert np.all(E[g4<-1e-9]<bs.ES) and np.all(E[g5<-1e-9]<bs.ES) and (g4<-1e-9).any() and (g5<-1e-9).any(),'demonstrations 4 and 5 help libration states under D'
pass
assert hit(3,1,2000)==hit(4,1,2000)==hit(5,1,2000)==500;claim('The last two demonstrations add no state, only pushes in libration, where D then waits less')
pass   # D+T at M = 5 against M = 4 is no longer claimed (largest difference printed by report.py)
cw=int(((E>bs.EHI)&(init[:,1]<0)).sum());ccw=int(((E>bs.EHI)&(init[:,1]>0)).sum());band=int(((E>=bs.ELO)&(E<=bs.EHI)).sum())
assert hit(1,1,2000)==cw+band and hit(2,1,2000)==cw+ccw+band and hit(3,1,2000)==500 and band==2,'the first three demonstrations add cw rotation, ccw rotation, libration'
last=[float(T[3,K].max()) for K in (1,2,4,8)];claim('the time by which every trial has reached the target falls from $%s$~s to $%s$, $%s$, and $%s$~s'%tuple(r(x,1) for x in last));assert last[0]>last[1]>last[2]>last[3]
pass
ok=np.all([np.isfinite(F[3,K]) for K in (1,2,4,8)],axis=0);first=[float(F[3,K][ok].mean()) for K in (1,2,4,8)]
claim('given here for the $%d$ trials that need an action'%int(ok.sum()));claim('mean first wait of the $%d$ trials that need an action'%int(ok.sum()))
assert all(abs(K*w/first[0]-1)<.10 for K,w in zip((2,4,8),first[1:]));claim('which follows the reference scaling $1/K$ within $10\\%$')
mean=[float(T[3,K].mean()) for K in (1,2,4,8)];claim('the mean reaching time falls from $%s$~s to $%s$, $%s$, and $%s$~s'%tuple(r(x,2) for x in mean))
def waits(M,K):
    rws,rec=R[M,K];idx=rec[:,0].astype(int);inter=[];term=[]
    for x in rws:
        a=rec[idx==x['index']]
        if len(a):inter.append(a[1:,2].sum());term.append(x['time']-(a[-1,1]+a[-1,3]))
    return float(np.mean(inter)),float(np.mean(term))
W=[waits(3,K)[0] for K in (1,2,4,8)];act=[np.mean([x['active'] for x in R[3,K][0]]) for K in (1,2,4,8)]
assert len({r(a,1) for a in act})==1
Wt=[waits(3,K)[1] for K in (1,2,4,8)];assert Wt[0]>=Wt[1]-.01 and Wt[1]>Wt[2]>Wt[3] and Wt[0]-Wt[3]<.5,'the final wait changes little'
claim('the time spent executing actions, which stays at $%s$~s; the waits between actions, which fall from $%s$~s under D to $%s$, $%s$, and $%s$~s; and the final wait for physical target entry, which changes little, from $%s$~s to $%s$~s'%(r(act[0],1),r(W[0],2),r(W[1],2),r(W[2],2),r(W[3],2),r(Wt[0],2),r(Wt[3],2)))

pass
pass
grid=np.round(np.arange(0,120.0001,.01),2);count={K:np.array([hit(3,K,t) for t in grid]) for K in (1,2,4,8)}
assert np.all(count[8]>=count[4]) and np.all(count[4]>=count[2]) and np.all(count[2]>=count[1])
pass
dt=T[3,8]-T[3,1];slow=dt>1e-9
claim('$K=8$ is faster than D from $%d$ initial states and slower from three'%int((dt<-1e-9).sum()))
assert slow.sum()==3
rot=E>=bs.ES;mr=[float(T[3,K][rot].mean()) for K in (1,2,4,8)];ml=[float(T[3,K][~rot].mean()) for K in (1,2,4,8)]
assert mr[0]>mr[1]>mr[2]>mr[3] and ml[0]>ml[1]>ml[2]>ml[3]   # reaching times by state type are no longer quoted in the text
pass
pass
eff=[float(np.mean([x['effort'] for x in R[3,K][0]])) for K in (1,2,4,8)];assert max(eff)==eff[2] and eff[3]<eff[2];claim('is $%s$, $%s$, $%s$, and $%s\\,\\mathrm{N^2\\,m^2\\,s}$ for $K=1,2,4,8$, not monotone in $K$ and largest at $K=4$'%tuple(r(e,1) for e in eff))
Lib=dict(np.load(S/'direct_fiber'/'library.npz'))
for K_ in (4,8):
    m_=Lib['introduced_K']<=K_;a_=R[3,K_][1];fr_=Lib['phase_fraction'][m_][a_[:,4].astype(int)];c_=Lib['control_flat'][Lib['offsets'][Lib['parent'][m_][a_[:,4].astype(int)]]];e_=c_**2*a_[:,3]
    assert e_[np.isclose(fr_,.25)|np.isclose(fr_,.75)].sum()>.6*e_.sum(),'quarter-period placements carry most of the effort'
claim('the quarter-period ones above all')
ms={M:[float(T[M,K][np.isfinite(T[M,K])].mean()) for K in (1,8)] for M in (1,2)}
pass
rng=np.random.default_rng(20261116);idx=rng.integers(0,500,size=(5000,500))
dw=np.where(ok,F[3,8]-F[3,1],np.nan);bw=np.array([np.nanmean(dw[i]) for i in idx]);bt=dt[idx].mean(1)
ci={}
for M in (1,2):
    both=np.isfinite(T[M,1])&np.isfinite(T[M,8]);dd=np.where(both,T[M,8]-T[M,1],0.);bb=np.array([dd[i][both[i]].mean() for i in idx]);ci[M]=np.quantile(bb,[.025,.975])
claim('the paired $95\\%%$ bootstrap interval for the change in reaching time from D to $K=8$ is $[%s,%s]$~s'%(r(np.quantile(bt,.025),2),r(np.quantile(bt,.975),2)))
pass
# appendix, single pendulum
c=json.loads((S/'construction_direct_fiber.json').read_text());sc=c['status_counts']
claim('$%s$ of the $%s$ requests are accepted'%(big(c['accepted']),big(c['attempts'])));claim('Of the requests that are not accepted, $%s$ have no such torque; the remaining seven belong to the anchor at the lower equilibrium'%big(sc['4']))
assert sc['1']==7 and c['attempts']-c['accepted']==sc['4']+sc['1'] and c['attempts']==7*c['original_assignments']
att=np.load(S/'direct_fiber'/'construction_attempts.npz')['records'];src=np.unique(att[att[:,4]==1][:,0].astype(int));assert len(src)==1 and np.all(lib['centers'][src[0]]==0),'the seven non-periodic placements are those of the equilibrium anchor'
size={(x['M'],x['K']):x['assignments'] for x in c['rows']};claim('At $M=5$, the libraries for $K=1,2,4,8$ hold $%s$, $%s$, $%s$, and $%s$ assignments'%tuple(big(size[5,K]) for K in (1,2,4,8)))
assert sum(x['guard_rejections'] for v in R.values() for x in v[0])==0;claim('this guard rejected no action')
assert max(x['max_energy'] for v in R.values() for x in v[0])<=bs.HX;claim('the path energy never exceeds $%d$~J'%bs.HX)
ref=json.loads((S/'refinement.json').read_text());assert ref['labels_changed']==0 and ref['trials']==500 and [(g['M'],g['K']) for g in ref['groups']]==[(M,K) for M in range(1,6) for K in (1,8)]
claim('Halving the step in $500$ trials ($50$ initial states for each of $M=1,\\ldots,5$ and $K=1,8$) leaves every success label unchanged and changes the reaching times by at most $3\\times10^{-10}$~s');assert ref['max_reaching_time_difference']<=3e-10
gaps=[-v['replay_end_target_gap'] for v in rep.values()]
assert all(v['inside_target'] and v['inside_operating_region'] and r(v['replay_end_V'],3)=='-0.050' and v['replay_max_energy']<=160+1e-6 for v in rep.values())
single_replay_energy=max(v['replay_max_energy'] for v in rep.values())
claim('whose energies fill $[E_-,E_+]\\approx[%s,%s]$~J'%(r(bs.ELO,3),r(bs.EHI,3)));claim('above the energy $E_{\\mathrm s}=%s$~J of the upright equilibrium'%r(bs.ES,3))
assert 'TERMINAL_SECTION=(b.ELO+b.EHI)/2' in (S/'policy.py').read_text();claim('$\\pi_*$ is its value at $E_*=(E_-+E_+)/2\\approx%s$~J, the midpoint of the target interval, at which the primitives stop'%r((bs.ELO+bs.EHI)/2,3))
claim('The expert uses at most $2\\,\\mathrm{N\\,m}$, a tenth of $u_{\\max}$');claim('it is a new constant torque of up to $20\\,\\mathrm{N\\,m}$, ten times the expert\'s bound')
for K_ in (1,2,4,8):
    f_=np.load(S/'direct_fiber'/f'main_M3_K{K_}.npz');t_=f_['times'];V_=f_['signed_V'];h_=np.array([x['time'] for x in R[3,K_][0]])
    assert all(V_[j][t_>h_[j]+1e-9].max(initial=-1.)<=1e-12 for j in range(500)),'a finished trial contributes zero residual afterwards'
chains=[];
for x in R[3,1][0]:
    a=R[3,1][1][R[3,1][1][:,0].astype(int)==x['index']]
    if len(a):chains.append(len(a)/(1+int((a[1:,2]>1e-9).sum())))
assert np.mean(chains)>2,'D executes several snippets in a row without waiting'

# ------------------------------------------------------------------ double pendulum
D=HERE/'meeting_double_nmpc_rk4';dem=json.loads((D/'demonstrations.json').read_text());n=dem['nmpc'];dd={x['kind']:x for x in dem['demonstrations']}
assert n=={'control_period':.025,'horizon_steps':80,'Q':[10.,10.,.1,.1],'Q_terminal':[100.,100.,1.,1.],'R':.001,'input_bound':20.,'energy_cap':100.,'cap_penalty':10000.,'discretization':'radau collocation, degree 3','plant':'RK4, step 0.005 s'}
assert dd['below']['start'][:2]==[.02,.02] and abs(dd['above']['start_energy']-90)<1e-9 and abs(dd['down']['start_energy']-90)<1e-9 and r(dd['down']['start'][0],2)=='4.09'
claim('They enter their targets after $%s$, $%s$, and $%s$~s'%tuple(r(dd[k]['duration'],1) for k in ('below','above','down')))
cr=json.loads((D/'certified_radius.json').read_text());assert max(v['radius_quantiles'][-1] for v in cr['demonstrations'].values())<1e-3
claim('With $L\\approx%d$ and $L_H\\approx%d$ on $X=\\{H\\le110~\\mathrm{J}\\}$, the closed-form radius of Algorithm~1 stays below $10^{-3}$'%(round(cr['L']),round(cr['L_V'])));claim('The closed-form radius of Algorithm~1 stays below $10^{-3}$ here')
con=json.loads((D/'construction_reuse.json').read_text())['targets']
for name in ('up','down'):
    for N in (0,2,4,8):
        rad=np.load(D/'reuse'/f'{name}_library_T{N}.npz')['radii'];assert np.mean(rad==.3)>.5,(name,N)
claim('most radii are $0.3$')
claim('hold $%d$, $%d$, $%d$, and $%d$ assignments for up--up and $%d$, $%d$, $%d$, and $%d$ for down--down'%tuple([con['up']['library_sizes'][k] for k in '0248']+[con['down']['library_sizes'][k] for k in '0248']))
su=json.loads((D/'summary_500.json').read_text());s=su['summary'];P=su['paired']
arms=['D','D+E','D+T8','D+E+T8']
assert [s[f'up/{a}']['hits'] for a in arms]==[344,500,395,500] and [s[f'down/{a}']['hits'] for a in arms]==[222,500,356,500]   # the counts are read from the legends of Figure 6
pass   # restricted means of D and D+T are no longer quoted in the text
pu=P['up: D+E+T8 vs D+E'];pd=P['down: D+E+T8 vs D+E']
claim('the $95\\%%$ bootstrap intervals for the change in mean reaching time under D+E+T are $[%d,%d]$~s for up--up and $[%d,%d]$~s for down--down'%(round(pu['bootstrap_95'][0]),round(pu['bootstrap_95'][1]),round(pd['bootstrap_95'][0]),round(pd['bootstrap_95'][1])))
claim('With exploration every trial starts a snippet and succeeds')
assert all(s[f'{t}/{a}']['never_act']==0 and s[f'{t}/{a}']['hits']==500 for t in ('up','down') for a in ('D+E','D+E+T8','D+E+T2','D+E+T4'))
claim('falls from $%s$~s under D+E to $%s$~s under D+E+T for up--up and from $%s$~s to $%s$~s for down--down'%(r(s['up/D+E']['median_first_action_time'],1),r(s['up/D+E+T8']['median_first_action_time'],1),r(s['down/D+E']['median_first_action_time'],1),r(s['down/D+E+T8']['median_first_action_time'],1)))
sw=lambda t:tuple(round(s[f'{t}/{a}']['restricted_mean_time']) for a in ('D+E','D+E+T2','D+E+T4','D+E+T8'))
claim('falls from $%d$~s to $%d$, $%d$, and $%d$~s for up--up and from $%d$~s to $%d$, $%d$, and $%d$~s for down--down'%(sw('up')+sw('down')))
for t in ('up','down'):
    x=[s[f'{t}/{a}']['restricted_mean_time'] for a in ('D+E','D+E+T2','D+E+T4','D+E+T8')];assert all(a>b for a,b in zip(x,x[1:])),'the mean reaching time decreases with N_T'
pass
# per-trial quantities that the summary does not hold
def trials(name,arm):
    M=[];final=[];last=[]
    for i in range(500):
        with np.load(D/'results'/'main'/'main'/name/arm/f'{i}.npz') as f:
            m=f['metrics'];M.append(m.copy());final.append(max(float(f['score'][-1]),0.));a=f['actions'];last.append(10000-a[-1,8] if len(a) else np.nan)
    return np.array(M),np.array(final),np.array(last)
X={(t,a):trials(t,a) for t in ('up','down') for a in ('D','D+E','D+T8','D+E+T8','D+E+T2','D+E+T4')}
def fails(t,a):
    M_=X[t,a][0];bad=M_[:,0]<=0;never=M_[:,2]<0;return int(bad.sum()),int((bad&never).sum()),int((bad&~never).sum())
fu=fails('up','D');fd=fails('down','D');assert fu[0]==500-s['up/D']['hits'] and fd[0]==500-s['down/D']['hits'] and fu[1]==s['up/D']['never_act'] and fd[1]==s['down/D']['never_act']
claim('$%d$ of the $%d$ unsuccessful up--up trials under D never activated a primitive. For down--down, $%d$ of the $%d$ unsuccessful trials never activated a primitive, while the remaining $%d$ executed at least one primitive before the horizon expired'%(fu[1],fu[0],fd[1],fd[0],fd[2]))
assert sum(int(X[t,a][0][:,17].sum()) for t in ('up','down') for a in ('D+E+T8','D+E+T2','D+E+T4','D+E','D'))==0;claim('D+E+T does not use this guard')
src=(D/'execute.py').read_text();assert 'Estar=.2 if target==0 else 58.86' in src and 'GUARD_HORIZON=5.;GUARD_RETRY=.25' in src and 'stall>=5.-1e-11' in src;claim('primitives stop at the midpoints $E_*=58.86$~J and $E_*=0.20$~J');claim('or if $5$~s of passive motion from its predicted endpoint reach the target or an original support; otherwise transported assignments are skipped for $0.25$~s');claim('exploration starts after $5$~s of uninterrupted waiting')
stall={}
for a in ('D','D+T8'):
    M,final,last=X['down',a];st=(M[:,0]<=0)&(M[:,2]>=0);stall[a]=(int(st.sum()),float(np.median(final[st])),float(np.median(last[st])))
pass;assert min(stall['D'][2],stall['D+T8'][2])>9600
M,final,last=X['up','D+T8'];st=(M[:,0]<=0)&(M[:,2]>=0);assert np.all(final[st]==0) and np.nanmin(last[st])>9000
pass
rf=json.loads((D/'refinement.json').read_text());g=rf['groups']
assert all(g[f'{t}/{a}']['labels_changed']==0 for t in ('up','down') for a in ('D+E','D+E+T8'))
W=['zero','one','two','three','four','five','six','seven','eight','nine','ten'];lc=[g[k]['labels_changed'] for k in ('up/D','up/D+T8','down/D','down/D+T8')];assert lc[0]==0 and sum(lc)==rf['labels_changed']
claim('changes $%d$ of $%d$ hit labels: %s for D+T on up--up, and %s for D and %s for D+T on down--down'%(rf['labels_changed'],rf['trials'],W[lc[1]],W[lc[2]],W[lc[3]]))
assert lc[2]==6;claim('changes $%d$ of $%d$ success labels, all under D or D+T, six of them under D for down--down'%(rf['labels_changed'],rf['trials']))
claim('Double-pendulum policy comparison on $500$ paired initial states per target with $N_T = 8$')
# appendix, double pendulum
ou=np.load(D/'up_originals.npz');od=np.load(D/'down_originals.npz')
claim('Of $%d$, $%d$, and $%d$ anchors tried on the three demonstrations, $%d$, $%d$, and $%d$ are accepted, which gives $%d$ original assignments for up--up and $%d$ for down--down'%(ou['tried'][0],ou['tried'][1],od['tried'][0],ou['accepted'][0],ou['accepted'][1],od['accepted'][0],len(ou['A']),len(od['A'])))
pad=sum(int(s+20>o['grid_points'][d_]) for o in (ou,od) for s,d_ in zip(o['snip'],o['demo']));claim('for $%d$ of the $%d$ original snippets, which start less than $0.1$~s before the end of a demonstration, it is completed with zero input'%(pad,len(ou['A'])+len(od['A'])))
up,dn=con['up'],con['down'];assert up['requests']==9*len(ou['A']) and up['accepted']==up['requests'] and not up['rejected_by_reason']
claim('For up--up, all $%d$ requested additions are accepted. For down--down, $%d$ of $%d$ are accepted; in $%d$ cases no stored snippet moves the energy toward the band, and in $%d$ the ball contains no admissible start energy'%(up['requests']-len(ou['A']),dn['accepted']-len(od['A']),dn['requests']-len(od['A']),dn['rejected_by_reason']['no_signal'],dn['rejected_by_reason']['no_activation_state']))
assert all(v['rows_without_admissible_energy']==0 and v['rows_cut']==0 and v['rows_split']==0 for t in (up,dn) for v in t['landing'].values());pass
ist=np.load(D/'initial_states.npz')['energies'][:500];cnt={int(e):int((ist==e).sum()) for e in np.unique(ist)};assert cnt=={5:84,15:84,30:83,45:83,60:83,90:83}
claim('The $500$ initial states comprise $84$ states at each of the energies $5$ and $15$~J and $83$ at each of $30$, $45$, $60$, and $90$~J')
gc=sum(int(X[t,'D+T8'][0][:,17].sum()) for t in ('up','down'));gr=sum(int(X[t,'D+T8'][0][:,18].sum()) for t in ('up','down'))
claim('which rejected $%s$ of the $%s$ candidates it examined in the $1000$ D+T trials'%(big(gr),big(gc)))
assert sum(int(v[0][:,6].sum())+int(v[0][:,7].sum()) for v in X.values())==0;emax=max(float(v[0][:,8].max()) for v in X.values())
claim('In all $%d$ trials (the four policies and the two further transport counts, on both targets), no action increases $V$, none starts outside its support, and the path energy stays below $%s$~J'%(500*len(X),r(math.ceil(emax*10)/10,1)))
ic=json.loads((D/'independent_check_500.json').read_text());assert not ic['violations'] and ic['worst']['end_V_difference']<4e-5 and ic['worst']['min_drop']>=.015 and ic['worst']['membership_ratio']<=1+1e-12
fc=json.loads((D/'rk4_fine_check_500.json').read_text());assert not fc['violations'] and fc['worst']['end_V_difference']<4e-5 and fc['worst']['membership_ratio']<=1+1e-12 and fc['integrator']=='RK4, step 0.0003125 s' and fc['finished_actions_replayed']==ic['finished_actions_replayed'];claim('A replay of all $%s$ finished actions at one eighth of the step reproduces the recorded value of $V$ at the end of each action within $4\\times10^{-5}$~J'%big(fc['finished_actions_replayed']))
dr=json.loads((D/'demonstration_replay.json').read_text());assert all(v['inside_target_all'] for v in dr.values())
vs=[v[k] for v in dr.values() for k in ('rk4_final_V','rk4_eighth_final_V','dop853_final_V')];em=max(v[k] for v in dr.values() for k in ('rk4_max_energy','rk4_eighth_max_energy','dop853_max_energy'))
assert max(v['rk4_eighth_minus_stored'] for v in dr.values())<1e-4
assert single_replay_energy<=160+1e-6 and max(vs)<0;claim('The path energy is at most $160$~J for the single pendulum and $%s$~J for the double pendulum'%r(math.ceil(em*100)/100,2))

# ------------------------------------------------------------------ section visitation (Figure 5)
ks=json.loads((HERE/'meeting_dynamics_kappa_half_decade'/'summary.json').read_text())['rows']
def cov(gain,step):
    return [x for x in ks if x['grid']=='64x128' and abs(x['gain']-gain)<1e-12 and abs(x['step']-step)<1e-12][0]
claim('the fraction of the $%d$ energetically allowed cells of a $64\\times128$ grid'%cov(0,.001)['cells'])
claim('at $5000$~s it rises from $%s\\%%$ for passive motion to $%s\\%%$ at $\\kappa=1$'%(r(100*cov(0,.001)['mean_coverage'],1),r(100*cov(1,.001)['mean_coverage'],1)))
nc=cov(1,.001)['mean_crossings'];assert abs(nc-2500)<100 and r(100*(1-math.exp(-2500/cov(1,.001)['cells'])),0)=='35';claim('the about $2500$ crossings of a trajectory would cover $35\\%$ of the cells')
claim('at $\\kappa=0.03$ it is $%s\\%%$, against $%s\\%%$ with $2$~ms'%(r(100*cov(.03,.001)['mean_coverage'],1),r(100*cov(.03,.002)['mean_coverage'],1)))
assert len({x['gain'] for x in ks})==8 and min(x['gain'] for x in ks)==0 and max(x['gain'] for x in ks)==1;claim('with eight gains $\\kappa\\in[0,1]$')

if text is None:
    for f,_ in checks:print(f)
    print(f'{len(checks)} statements recomputed from the stored runs; set HCP_MAIN_TEX to compare them with a manuscript');sys.exit(0)
bad=[f for f,ok_ in checks if not ok_]
for f in bad:print('NOT FOUND:',f)
print(f'{len(checks)-len(bad)} of {len(checks)} statements found in {TEX}')
sys.exit(1 if bad else 0)
