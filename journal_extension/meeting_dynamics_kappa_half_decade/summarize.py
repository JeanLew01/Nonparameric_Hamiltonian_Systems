"""Section coverage for eight gains: 40 checkpoints through 5000 s, and the figure."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from run import ROOT,GAINS,HORIZON,STEPS
PHI_MAX=np.arccos(1-15/(2*9.81));V_MAX=np.sqrt(30);TIMES=np.linspace(0,HORIZON,40);MAPS=[0,3,7]

def accessible(bins):
    qe=np.linspace(-PHI_MAX,PHI_MAX,bins[0]+1);ve=np.linspace(-V_MAX,V_MAX,bins[1]+1)
    closest=np.where((qe[:-1]<=0)&(qe[1:]>=0),0.,np.minimum(abs(qe[:-1]),abs(qe[1:])))
    c=np.cos(closest);bound=np.sqrt(np.maximum(0,2*(15-2*9.81*(1-c))/(2-c*c)))
    return (ve[:-1][None,:]<=bound[:,None])&(ve[1:][None,:]>=-bound[:,None])

def first_visit(events,bins):
    """Time of the first crossing in each cell (inf if never)."""
    qi=np.clip(((events[:,1]+PHI_MAX)/(2*PHI_MAX)*bins[0]).astype(int),0,bins[0]-1)
    vi=np.clip(((events[:,2]+V_MAX)/(2*V_MAX)*bins[1]).astype(int),0,bins[1]-1)
    first=np.full(bins[0]*bins[1],np.inf);np.minimum.at(first,bins[1]*qi+vi,events[:,0])
    return first.reshape(bins)

def main():
    data={(r,s,j):dict(np.load(ROOT/r/f'seed{s}_g{j}.npz')) for r in STEPS for s in range(6) for j in range(len(GAINS))}
    cover={};rows=[]
    for bins in [(64,128),(32,64),(16,32)]:
        ok=accessible(bins)
        for r in STEPS:
            c=np.zeros((len(GAINS),6,len(TIMES)))
            for j in range(len(GAINS)):
                for s in range(6):
                    first=first_visit(data[r,s,j]['events'],bins);assert np.all(np.isinf(first[~ok]))
                    c[j,s]=[(first<=t).sum()/ok.sum() for t in TIMES]
            assert np.all(np.diff(c,axis=-1)>=0) and np.all(c[...,0]==0);cover[bins,r]=c
            for j,g in enumerate(GAINS):
                M=np.array([data[r,s,j]['metrics'] for s in range(6)])
                rows.append(dict(grid=f'{bins[0]}x{bins[1]}',cells=int(ok.sum()),step=STEPS[r],gain=g,mean_coverage=float(c[j,:,-1].mean()),min_coverage=float(c[j,:,-1].min()),max_coverage=float(c[j,:,-1].max()),
                    mean_crossings=float(M[:,3].mean()),max_energy_error=float(M[:,0].max()),max_force=float(M[:,1].max())))
    primary=cover[(64,128),'fine'];means=primary.mean(1)
    with (ROOT/'coverage_checkpoints.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['time_s']+[f'kappa_{g:g}_mean_percent' for g in GAINS])
        for i,t in enumerate(TIMES):w.writerow([t]+(100*means[:,i]).tolist())
    (ROOT/'summary.json').write_text(json.dumps(dict(gains=GAINS,horizon=HORIZON,times=TIMES.tolist(),rows=rows,
        raw_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for r in STEPS for f in sorted((ROOT/r).glob('*.npz'))},
        scope='Six fixed initial states with bounded passive orbits, one recorded stream each; finite-time cell occupancy of one section, not an ergodicity certificate.'),indent=1)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    import sys;sys.path.insert(0,str(ROOT.parent));import figure_style;figure_style.use_times()   # Times New Roman, shared by all Section 5 figures
    fig=plt.figure(figsize=(10.5,6.3),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,.85])
    axes=[fig.add_subplot(grid[0,j]) for j in range(3)];cov=fig.add_subplot(grid[1,:])
    phi=np.linspace(-PHI_MAX,PHI_MAX,600);c=np.cos(phi);bound=np.sqrt(np.maximum(0,2*(15-2*9.81*(1-c))/(2-c*c)))
    cmap=ListedColormap(['#ffffff','#d8e9f5','#aacfe6','#73afd2','#408cbb','#17649d','#083b6c']);norm=BoundaryNorm(np.arange(-.5,7),cmap.N);ok=accessible((64,128))
    for col,j in enumerate(MAPS):
        visits=np.sum([np.isfinite(first_visit(data['fine',s,j]['events'],(64,128))) for s in range(6)],axis=0)
        mesh=axes[col].pcolormesh(np.linspace(-PHI_MAX,PHI_MAX,65),np.linspace(-V_MAX,V_MAX,129),np.ma.masked_where(~ok,visits).T,cmap=cmap,norm=norm,shading='flat')
        for sign in (-1,1):axes[col].plot(phi,sign*bound,'--',color='#35866b',lw=.8)
        axes[col].set(xlim=(-1.38,1.38),ylim=(-5.65,5.65),xlabel=r'$q_1$ (rad)',ylabel=r'$\dot q_1$ (rad/s)')
        axes[col].set_title(f'({chr(97+col)}) '+(r'$\kappa=0$' if GAINS[j]==0 else rf'$\kappa={GAINS[j]:g}$'),fontsize=10)
        axes[col].text(.02,.02,f'Mean visited cells: {100*means[j,-1]:.1f}%',transform=axes[col].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=2))
    bar=fig.colorbar(mesh,ax=axes,orientation='horizontal',fraction=.05,pad=.02,ticks=range(7),shrink=.65);bar.set_label('Trajectories visiting each cell by 5,000 s (of six)',fontsize=8)
    colors=['#333333']+[plt.get_cmap('viridis')(v) for v in np.linspace(.05,.9,len(GAINS)-1)];markers=['o','s','^','D','v','P','X','*'];styles=['-','--','-.',':','-','--','-.',':']
    for j,g in enumerate(GAINS):
        cov.plot(TIMES,100*means[j],color=colors[j],ls=styles[j],marker=markers[j],ms=3,lw=1.1,mfc='white',label=rf'$\kappa={g:g}$')
    cov.set(xlim=(0,HORIZON),ylim=(0,None),xlabel='Time (s)',ylabel='Mean visited cells per trajectory (%)');cov.set_title('(d) Section coverage');cov.grid(alpha=.2)
    cov.legend(ncol=2,frameon=False,fontsize=8,loc='upper left')
    for suffix in ('pdf','png'):fig.savefig(ROOT/f'poincare_diagnostic.{suffix}',dpi=200)
    plt.close(fig)
    for x in rows:print('%-7s step %.3f gain %-6g coverage %5.2f%% [%5.2f,%5.2f] crossings %7.1f energy err %.1e force %.2f'%(x['grid'],x['step'],x['gain'],100*x['mean_coverage'],100*x['min_coverage'],100*x['max_coverage'],x['mean_crossings'],x['max_energy_error'],x['max_force']))

if __name__=='__main__':main()
