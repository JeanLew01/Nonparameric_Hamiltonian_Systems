"""Draw the four Section 5 figures in their original layouts, in Times New Roman.

The layouts, shapes and annotations are those of the study scripts
(analyze.py / summarize.py); only the typeface changes.  With a number as
first argument every font size of a figure is multiplied by one factor, chosen
so that ticks and axis labels print at that many points when the figure is
included at 0.9 of the 504 pt text width; with `original` (the default) the
font sizes of the study scripts are kept.  All data are read from the study
folders; nothing is recomputed.

    python paper_figures.py [original|effective_pt] [output_folder]

The output folder defaults to the manuscript's figures/.  PDFs and PNGs are
written side by side when an output folder is given; otherwise the PNG
previews go to paper_figures_preview/.
"""
from pathlib import Path
import json,os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.ticker import PercentFormatter
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import figure_style
PAPER=Path(os.environ.get('HCP_PAPER',Path.home()/'overleaf_project'/'Hamiltonian_Control_via_Chain_Policies'))
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else PAPER/'figures'
PREVIEW=OUT if len(sys.argv)>2 else HERE/'paper_figures_preview'
INCLUDED=.9*504/72.27          # printed width in inches
EFFECTIVE=None if len(sys.argv)<2 or sys.argv[1]=='original' else float(sys.argv[1])
COLORS=['#485f89','#5d9978','#d19a35','#b44747']
POLICY=['#0072B2','#E69F00','#009E73','#CC79A7']

def start(width,base):
    """Original rcParams of the study scripts in Times New Roman; returns the font factor for a figure `width` inches wide."""
    k=1. if EFFECTIVE is None else EFFECTIVE*72/72.27/(base*INCLUDED/width)
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({'font.size':base*k,'axes.spines.top':False,'axes.spines.right':False})
    figure_style.use_times()
    return k

def save(fig,name):
    PREVIEW.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.03);fig.savefig(PREVIEW/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

SINGLE=os.environ.get('HCP_SINGLE_STUDY','')   # e.g. meeting_single_nmpc_fiber (NMPC demonstrations); empty: the hand-written expert studies
SHORT=float(os.environ.get('HCP_SINGLE_SHORT','4' if SINGLE else '20'));XMAX=float(os.environ.get('HCP_SINGLE_XMAX','10' if SINGLE else '40'))
LONG=float(os.environ.get('HCP_SINGLE_LONG','2000'))   # second horizon of the budget figure

def single_rows(M,K):
    d=np.load(HERE/SINGLE/'direct_fiber'/f'main_M{M}_K{K}.npz');return json.loads(str(d['rows'])),d

def single_budget():
    k=start(8.5,10)
    if SINGLE:
        S={}
        for M in range(1,6):
            for K in (1,8):
                rows,_=single_rows(M,K);S[M,K]={'short':sum(r['success'] and r['time']<=SHORT for r in rows),'successes_at_2000':sum(r['success'] and r['time']<=LONG for r in rows)}
    else:
        S={(g['M'],g['K']):dict(g,short=g['successes_at_20']) for g in json.loads((HERE/'meeting_single_budget_fix'/'summary.json').read_text())['groups']}
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2));x=np.arange(1,6);width=.34
    for ax,(horizon,key) in zip(axes,[(int(SHORT) if SHORT==int(SHORT) else SHORT,'short'),(int(LONG) if SINGLE else 2000,'successes_at_2000')]):
        for K,offset,color,label in [(1,-width/2,COLORS[0],'D'),(8,width/2,COLORS[3],'D+T ($K=8$)')]:
            rates=np.array([S[M,K][key]/5 for M in x]);bars=ax.bar(x+offset,rates,width,color=color,label=label)
            ax.bar_label(bars,labels=[f'{v:g}' for v in rates],padding=2,fontsize=6.5*k)
        ax.set(xlabel='Demonstrations $M$',ylabel='Success rate (%)',xticks=x,ylim=(0,112),title=f'$T={horizon:,}$ s'.replace(',','{,}'))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=9*k,bbox_to_anchor=(.5,1.07));fig.tight_layout()
    save(fig,'single_success_by_demonstrations')

def single_time():
    k=start(11,10);folder=HERE/(SINGLE or 'meeting_single_transport_fix')
    if SINGLE:
        F=np.array([[np.nan if r['first_action'] is None else r['first_action'] for r in single_rows(3,K)[0]] for K in (1,2,4,8)]);common=np.all(np.isfinite(F),axis=0);first=[float(F[j,common].mean()) for j in range(4)]
    else:
        first=json.loads((folder/'audit.json').read_text())['M3']['mean_first_wait'];first=[first[str(K)] for K in (1,2,4,8)]
    fig,axes=plt.subplots(1,3,figsize=(11,3.1))
    for K,color in zip([1,2,4,8],COLORS):
        d=np.load(folder/'direct_fiber'/f'main_M3_K{K}.npz');rows=json.loads(str(d['rows']));t=d['times']
        axes[0].plot(t,np.maximum(d['signed_V'],0).mean(0),label=f'$K={K}$',color=color,lw=1.6)
        axes[1].step(t,np.mean([np.asarray(r['success'])&(t>=r['time']) for r in rows],axis=0),where='post',color=color,lw=1.6)
    for ax in axes[:2]:ax.grid(alpha=.2);ax.set_xlabel('Time (s)');ax.set_xlim(0,XMAX)
    axes[0].set_ylabel(r'$\overline{V}_+$ (J)');axes[0].legend(frameon=False,fontsize=8*k)
    axes[1].set_ylabel(r'$\widehat{p}$');axes[1].set_ylim(-.025,1.025)
    bars=axes[2].bar(range(4),first,color=COLORS,width=.62)
    if SINGLE:   # the 1/K reference crosses the bar centres here, so the values sit at the foot of the bars
        for x,v in enumerate(first):axes[2].text(x,.035*first[0],f'{v:.3f}',ha='center',va='bottom',color='white',fontsize=8*min(k,1.3))
    else:axes[2].bar_label(bars,labels=[f'{v:.3f}' for v in first],label_type='center',color='white',fontsize=8*min(k,1.3))
    axes[2].plot(range(4),[first[0]/q for q in (1,2,4,8)],'k--',lw=1,marker='o',ms=3,label=r'$1/K$ reference');axes[2].legend(frameon=False,fontsize=8*k)
    axes[2].set(xticks=range(4),xticklabels=['1','2','4','8'],xlabel='Placements $K$',ylabel='Mean first wait (s)',ylim=(0,1.22*first[0]))
    axes[2].grid(axis='y',alpha=.2);axes[2].set_axisbelow(True)
    for ax,t in zip(axes,['(a) Graph progress','(b) Target reaching','(c) First wait']):ax.set_title(t,fontsize=10*k)
    fig.tight_layout();save(fig,'single_reeb_time')

def coverage():
    k=start(10.5,9);folder=HERE/os.environ.get('HCP_KAPPA_STUDY','meeting_dynamics_kappa_half_decade');sys.path.insert(0,str(folder))   # eight gains; meeting_dynamics_kappa_log holds the earlier six
    import summarize as Z
    from run import GAINS,HORIZON
    first={(s,j):Z.first_visit(np.load(folder/'fine'/f'seed{s}_g{j}.npz')['events'],(64,128)) for s in range(6) for j in range(len(GAINS))}
    ok=Z.accessible((64,128));times=Z.TIMES;means=np.array([[[(first[s,j]<=t).sum()/ok.sum() for t in times] for s in range(6)] for j in range(len(GAINS))]).mean(1)
    fig=plt.figure(figsize=(10.5,6.3),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,.85])
    axes=[fig.add_subplot(grid[0,j]) for j in range(3)];cov=fig.add_subplot(grid[1,:])
    phi=np.linspace(-Z.PHI_MAX,Z.PHI_MAX,600);c=np.cos(phi);bound=np.sqrt(np.maximum(0,2*(15-2*9.81*(1-c))/(2-c*c)))
    cmap=ListedColormap(['#ffffff','#d8e9f5','#aacfe6','#73afd2','#408cbb','#17649d','#083b6c']);norm=BoundaryNorm(np.arange(-.5,7),cmap.N)
    for col,j in enumerate(Z.MAPS):
        visits=np.sum([np.isfinite(first[s,j]) for s in range(6)],axis=0)
        mesh=axes[col].pcolormesh(np.linspace(-Z.PHI_MAX,Z.PHI_MAX,65),np.linspace(-Z.V_MAX,Z.V_MAX,129),np.ma.masked_where(~ok,visits).T,cmap=cmap,norm=norm,shading='flat',rasterized=True)
        for sign in (-1,1):axes[col].plot(phi,sign*bound,'--',color='#35866b',lw=.8)
        axes[col].set(xlim=(-1.38,1.38),ylim=(-5.65,5.65),xlabel=r'$q_1$ (rad)',ylabel=r'$\dot q_1$ (rad/s)')
        if k>1.2:axes[col].set(ylim=(-7.0,5.65),yticks=[-4,-2,0,2,4],xticks=[-1,0,1])
        axes[col].set_title(f'({chr(97+col)}) '+(r'$\kappa=0$' if GAINS[j]==0 else rf'$\kappa={GAINS[j]:g}$'),fontsize=10*k)
        if k>1.2:axes[col].text(.5,.025,f'Mean visited cells: {100*means[j,-1]:.1f}%',transform=axes[col].transAxes,fontsize=8*k,ha='center')
        else:axes[col].text(.02,.02,f'Mean visited cells: {100*means[j,-1]:.1f}%',transform=axes[col].transAxes,fontsize=8*k,bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=2))
    bar=fig.colorbar(mesh,ax=axes,orientation='horizontal',fraction=.05,pad=.02,ticks=range(7),shrink=.65);bar.set_label('Trajectories visiting each cell by 5,000 s (of six)',fontsize=8*k)
    colors=['#333333']+[plt.get_cmap('viridis')(v) for v in np.linspace(.05,.9,len(GAINS)-1)];markers=['o','s','^','D','v','P','X','*'];styles=['-','--','-.',':','-','--','-.',':']
    for j,g in enumerate(GAINS):cov.plot(times,100*means[j],color=colors[j],ls=styles[j],marker=markers[j],ms=3,lw=1.1,mfc='white',label=rf'$\kappa={g:g}$')
    cov.set(xlim=(0,HORIZON),ylim=(0,None),xlabel='Time (s)',ylabel='Mean visited cells\nper trajectory (%)' if k>1.2 else 'Mean visited cells per trajectory (%)');cov.set_title('(d) Section coverage');cov.grid(alpha=.2)
    cov.legend(ncol=2,frameon=False,fontsize=8*k,loc='upper left')
    save(fig,'poincare_diagnostic')

def double_curves(folder,n):
    """Per policy and target: mean and standard deviation over the trials of the positive residual (0.1 s grid), and the sorted hit times (cached)."""
    cache=folder/f'figure_curves_{n}.npz'
    if cache.exists():return dict(np.load(cache))
    out={}
    for name in ['up','down']:
        for arm in ['D','D+E','D+T8','D+E+T8']:
            S=Q=None;hits=[]
            for i in range(n):
                with np.load(folder/'results'/'main'/'main'/name/arm/f'{i}.npz') as d:
                    s=np.maximum(d['score'].astype(np.float64),0);S=s if S is None else S+s;Q=s*s if Q is None else Q+s*s;m=d['metrics']
                    if m[0]>0:hits.append(m[1])
            mean=S/n;out[f'{name}/{arm}/mean']=mean;out[f'{name}/{arm}/sd']=np.sqrt(np.maximum(Q/n-mean*mean,0)*n/(n-1));out[f'{name}/{arm}/hits']=np.sort(hits)
    np.savez_compressed(cache,**out);return out

BAND=os.environ.get('HCP_DOUBLE_BAND','')   # '': curves only; 'ci': pointwise 95% confidence band of each curve; 'se': +-1 standard error; 'sd': +-1 standard deviation over the trials

def double_policy():
    k=start(10.8,10);folder=HERE/os.environ.get('HCP_DOUBLE_STUDY','meeting_double_transport_fix')   # meeting_double_nmpc: NMPC demonstrations; meeting_double_500: the hand-made ones on 500 states
    n=int(os.environ.get('HCP_DOUBLE_N',len(list((folder/'results'/'main'/'main'/'up'/'D').glob('[0-9]*.npz')))))   # trials 0..n-1
    C=double_curves(folder,n);z=1.959963984540054
    fig,axs=plt.subplots(2,2,figsize=(10.8,6.5),sharex='col',layout='constrained')
    for col,name in enumerate(['up','down']):
        for j,(arm,label) in enumerate(zip(['D','D+E','D+T8','D+E+T8'],['D','D+E','D+T','D+E+T'])):
            mean=C[f'{name}/{arm}/mean'];sd=C[f'{name}/{arm}/sd'];hits=C[f'{name}/{arm}/hits']
            t=np.arange(len(mean))*.1;p=np.searchsorted(hits,t,side='right')/n
            if BAND:
                if BAND=='sd':lo,hi=mean-sd,mean+sd;q=np.sqrt(p*(1-p)*n/(n-1));plo,phi=p-q,p+q
                elif BAND=='se':lo,hi=mean-sd/np.sqrt(n),mean+sd/np.sqrt(n);q=np.sqrt(p*(1-p)/(n-1));plo,phi=p-q,p+q
                else:   # normal interval for the mean residual, Wilson interval for the success rate
                    lo,hi=mean-z*sd/np.sqrt(n),mean+z*sd/np.sqrt(n);c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);plo,phi=c-h,c+h
                # The bands are drawn on a thinned grid (every 0.1 s up to 10 s, then 1,500 log-spaced times); 100,001 points per band give a 14 MB PDF.
                g=np.unique(np.r_[np.arange(101),np.round(np.logspace(2,np.log10(len(t)-1),1500)).astype(int)])
                axs[0,col].fill_between(t[g],np.maximum(lo,0)[g],hi[g],color=POLICY[j],alpha=.18,lw=0);axs[1,col].fill_between(t[g],100*np.clip(plo,0,1)[g],100*np.clip(phi,0,1)[g],step='post',color=POLICY[j],alpha=.18,lw=0)
            axs[0,col].plot(t,mean,color=POLICY[j],lw=1.7);axs[1,col].step(t,100*p,where='post',color=POLICY[j],lw=1.7,label=f'{label} ({len(hits)}/{n})')
        axs[0,col].set_title('Up–up, $M=2$' if name=='up' else 'Down–down, $M=1$');axs[0,col].set_ylim(0,None);axs[1,col].set_ylim(-2,102)
        axs[1,col].yaxis.set_major_formatter(PercentFormatter(100));axs[1,col].set_xlabel('Time (s)');axs[1,col].legend(loc='upper left',fontsize=8*k,frameon=False)
        for row in range(2):
            ax=axs[row,col];ax.set_xscale('symlog',linthresh=10);ax.set_xlim(0,10000);ax.set_xticks([0,10,100,1000,10000],['0','10','100','1000','10,000']);ax.grid(alpha=.2)
            for tick in ax.get_xticklabels():
                if tick.get_text()=='1000':tick.set_horizontalalignment('right')
                if tick.get_text()=='10,000':tick.set_horizontalalignment('left')
    axs[0,0].set_ylabel(r'$\overline{V}_+$ (J)');axs[1,0].set_ylabel(r'$\widehat{p}$')
    save(fig,'double_policy_time'+os.environ.get('HCP_DOUBLE_SUFFIX',''))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    single_budget();single_time();coverage();double_policy()
    print('font sizes','original' if EFFECTIVE is None else f'{EFFECTIVE:g} pt printed','; written to',OUT)
