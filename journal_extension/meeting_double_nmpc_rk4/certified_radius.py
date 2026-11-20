"""What the closed-form certified radius of the conference paper gives for the double pendulum.

r_i(t) = (V(x_i) - V(phi(t, x_i, u)) - v0 t) / (L_V (1 + e^{L t})), maximized over the duration t,
with the global constants L = sup ||df/dx||_2 and L_V = sup ||grad H|| on the operating region
X = {H <= 110 J} (estimated by sampling, so they are lower bounds and the radii upper bounds),
evaluated along the three NMPC demonstrations on the 5 ms grid.
"""
import json
import numpy as np
import benchmark as b
import construct as c
rng=np.random.default_rng(0);L=0.;LV=0.;n=0
def jac(x):
    J=np.zeros((4,4));e=1e-6
    for i in range(4):
        d=np.zeros(4);d[i]=e;J[:,i]=(b.rhs(x+d,np.zeros(2))-b.rhs(x-d,np.zeros(2)))/(2*e)
    return J
while n<40000:
    q=rng.uniform(-np.pi,np.pi,2);U=b.G*(3-2*np.cos(q[0])-np.cos(q[1]))
    if U>=110:continue
    th=rng.uniform(0,2*np.pi);d=np.array([np.cos(th),np.sin(th)]);x=np.r_[q,d];K=(110-U)*rng.uniform()**.5
    x[2:]*=np.sqrt(K/(.5*d@b.velocity(x)));n+=1
    L=max(L,np.linalg.norm(jac(x),2));LV=max(LV,np.linalg.norm(c.grad_energy(x)))
out=dict(L=float(L),L_V=float(LV),demonstrations={})
for kind,target in [('below',1),('above',1),('down',0)]:
    X,U=c.fine_demo(kind);V=np.array([b.score(x,target) for x in X]);t=np.arange(len(X))*c.STEP;best=[]
    for s in range(len(X)-1):
        if V[s]<=0:continue
        tt=t[s+1:]-t[s];r=(V[s]-V[s+1:]-1e-3*tt)/(LV*(1+np.exp(np.minimum(L*tt,700.))))
        if r.max()>0:best.append((float(r.max()),float(tt[int(np.argmax(r))])))
    R=np.array([x[0] for x in best]);out['demonstrations'][kind]=dict(anchors_with_positive_radius=len(best),radius_quantiles=[float(q) for q in np.quantile(R,[.1,.5,.9,1.])],median_duration=float(np.median([x[1] for x in best])))
print(json.dumps(out,indent=1));open('certified_radius.json','w').write(json.dumps(out,indent=1)+'\n')
