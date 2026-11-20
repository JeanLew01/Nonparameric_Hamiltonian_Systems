"""Fixed E=15 passive/exploration diagnostic; numerical evidence only.

The observed section is q2=0, p2>0, as in Cabrera et al. (2023), Eq. (5).
Both crossing directions are retained.  Dynamics are derived from H,
not transcribed from the misprinted equations in the source article.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import root

ROOT = Path(__file__).resolve().parent
G = 9.81
ENERGY = 15.0


@njit(cache=True)
def velocity(x):
    c = np.cos(x[0]-x[1]); d = 2-c*c
    return (x[2]-c*x[3])/d, (2*x[3]-c*x[2])/d


@njit(cache=True)
def energy(x):
    v1,v2=velocity(x)
    return .5*(x[2]*v1+x[3]*v2)+G*(3-2*np.cos(x[0])-np.cos(x[1]))


@njit(cache=True)
def control(x,alpha):
    v1,v2=velocity(x)
    a=alpha/max(1.,abs(alpha)*max(abs(v1),abs(v2))/20.)
    return -a*v2,a*v1


@njit(cache=True)
def rhs(x,alpha):
    v1,v2=velocity(x);u1,u2=control(x,alpha)
    s=np.sin(x[0]-x[1])*v1*v2
    return np.array((v1,v2,-s-2*G*np.sin(x[0])+u1,s-G*np.sin(x[1])+u2))


@njit(cache=True)
def step(x,alpha,h):
    k1=rhs(x,alpha);k2=rhs(x+h*k1/2,alpha)
    k3=rhs(x+h*k2/2,alpha);k4=rhs(x+h*k3,alpha)
    return x+h*(k1+2*k2+2*k3+k4)/6


def section_state(phi,v1):
    c=np.cos(phi)
    disc=2*(ENERGY-2*G*(1-c))-(2-c*c)*v1*v1
    if disc<=0:raise ValueError('outside the positive-p2 section')
    p2=np.sqrt(disc);v2=-c*v1+p2
    return np.array((phi,0.,2*v1+c*v2,p2))


def return_map(z,rtol=2e-12,atol=2e-13):
    x0=section_state(*z)
    def crossing(t,x):return x[1]
    crossing.direction=0
    sol=solve_ivp(lambda t,x:rhs(x,0.),(0,8),x0,method='DOP853',
                  rtol=rtol,atol=atol,max_step=.03,events=crossing)
    for t,x in zip(sol.t_events[0],sol.y_events[0]):
        v1,v2=velocity(x)
        if t>1e-7 and x[3]>1e-9 and abs(v2)>1e-7:
            return np.array((x[0],v1)),float(t),x
    raise RuntimeError('No transverse positive-p2 return in eight seconds')


def prepare():
    # The locator uses only the uncontrolled dynamics, before any comparison.
    sol=root(lambda z:return_map(z)[0]-z,np.array((0.,1.165)),tol=1e-10)
    z=sol.x;ret,period,xret=return_map(z,rtol=3e-13,atol=3e-14)
    residual=float(np.linalg.norm(ret-z))
    if not sol.success or residual>1e-8:raise RuntimeError((sol.message,residual))
    # Fixed seeds: period-one center, two island offsets, and three broad seeds.
    charts=[z.tolist(),[0.,float(z[1]+.03)],[0.,float(z[1]+.20)],
            [0.,0.],[0.,2.5],[.65,0.]]
    names=['R0','R1','R2','B0','B1','B2']
    states=np.array([section_state(*a) for a in charts])
    np.savez_compressed(ROOT/'seeds.npz',states=states,section=np.array(charts))
    protocol=dict(energy=ENERGY,g=G,masses=[1,1],lengths=[1,1],
      seed_names=names,section_charts=charts,canonical_initial_states=states.tolist(),
      periodic_locator=dict(initial_guess=[0,1.165],root=z.tolist(),period=period,
                            high_accuracy_return_residual=residual),
      passive_horizon=10000.,exploration_horizon=2000.,main_step=.002,fine_step=.001,
      strengths=[0.,.8,5.],passive_prefix=5.,sign_dwell=.5,random_seed=20261106,
      force='alpha R v / max(1, |alpha| ||v||inf / 20)',
      sampling='same predeclared six states, signs paired across strengths and resolutions',
      section='q2=0, p2>0, both crossing directions, |v2|>1e-7',
      separation=dict(perturbation=1e-8,reset_interval=1.,
                      direction=[.31,.7,.2,.6],meaning='finite-time separation rate, no chaos proof'),
      patch=dict(center=z.tolist(),half_widths=[.05,.05],
                 meaning='fixed section diagnostic patch, not a policy assignment or target'),
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol['periodic_locator'],indent=2),flush=True)


@njit(cache=True)
def simulate(x0,horizon,h,strength,signs,center):
    n=int(round(horizon/h)); dwell=int(round(.5/h)); prefix=int(round(5./h))
    renorm=int(round(1./h));sample_every=int(round(1./h))
    x=x0.copy();direction=np.array((.31,.7,.2,.6))
    direction/=np.sqrt(np.sum(direction*direction));pert=x+1e-8*direction
    total_log=0.;effort=0.;abs_work=0.;max_power=0.;max_energy_error=0.
    events=np.zeros((int(horizon*5)+100,7));ne=0
    traces=np.zeros((int(horizon)+1,7));nt=1
    traces[0]=np.array((0.,0.,0.,0.,0.,0.,0.))
    first_patch=np.inf;patch_hits=0;max_force=0.
    for k in range(n):
        alpha=0.
        if strength>0. and k>=prefix:
            alpha=strength*signs[k//dwell]
        un=control(x,alpha);v=velocity(x)
        power=abs(un[0]*v[0]+un[1]*v[1])
        max_power=max(max_power,power);abs_work+=power*h
        max_force=max(max_force,abs(un[0]),abs(un[1]))
        xn=step(x,alpha,h);pert=step(pert,alpha,h)
        un1=control(xn,alpha)
        effort+=.5*h*(un[0]**2+un[1]**2+un1[0]**2+un1[1]**2)
        max_energy_error=max(max_energy_error,abs(energy(xn)-ENERGY))
        if x[1]*xn[1]<0. or (xn[1]==0. and x[1]!=0.):
            lo=0.;hi=h;left=x[1]
            for _ in range(26):
                mid=(lo+hi)/2.;ym=step(x,alpha,mid)
                if ym[1]*left>0.:lo=mid
                else:hi=mid
            t=k*h+(lo+hi)/2.;y=step(x,alpha,(lo+hi)/2.)
            v1,v2=velocity(y)
            if y[3]>1e-9 and abs(v2)>1e-7:
                if ne>=len(events):raise ValueError('event capacity exceeded')
                events[ne]=np.array((t,y[0],v1,y[2],y[3],v2,energy(y)-ENERGY));ne+=1
                if abs(y[0]-center[0])<.05 and abs(v1-center[1])<.05:
                    patch_hits+=1;first_patch=min(first_patch,t)
        x=xn
        if (k+1)%renorm==0:
            d=pert-x;norm=np.sqrt(np.sum(d*d))
            total_log+=np.log(norm/1e-8);pert=x+1e-8*d/norm
        if (k+1)%sample_every==0:
            t=(k+1)*h
            traces[nt]=np.array((t,total_log/t,effort,energy(x)-ENERGY,
                                max_energy_error,abs_work,max_power));nt+=1
    metrics=np.array((max_energy_error,effort,abs_work,max_power,max_force,
                      first_patch,patch_hits,ne,total_log/horizon))
    return events[:ne],traces[:nt],metrics,x


def run(label):
    p=json.loads((ROOT/'protocol.json').read_text())
    states=np.load(ROOT/'seeds.npz')['states'];h=p[label+'_step']
    center=np.array(p['periodic_locator']['root']);out=ROOT/label;out.mkdir(exist_ok=True)
    for k,x in enumerate(states):
        signs=np.random.default_rng(p['random_seed']+k).choice(np.array((-1.,1.)),
                      size=int(p['passive_horizon']/.5)+1)
        for strength in p['strengths']:
            file=out/f'seed{k}_k{strength:g}.npz'
            if file.exists():continue
            horizon=p['passive_horizon'] if strength==0 else p['exploration_horizon']
            tic=time.time();events,traces,metrics,xf=simulate(x,horizon,h,strength,signs,center)
            np.savez_compressed(file,events=events,traces=traces,metrics=metrics,
                                final_state=xf,source_sha256=np.array(p['source_sha256']))
            print(label,k,strength,len(events),'energy',metrics[0],
                  'separation',metrics[-1],'seconds',round(time.time()-tic,2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','main','fine']);args=a.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode)
