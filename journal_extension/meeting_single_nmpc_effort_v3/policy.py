"""Common physical-target executor and prospective continuation safeguard."""
from common import *
from scipy.optimize import brentq

TERMINAL_SECTION=(b.ELO+b.EHI)/2


def terminal_crossing(x,control,lead,dt,subdivisions,remaining):
    states,hs,us=b.primitive_trace(x,control,lead,dt,subdivisions)
    time=0.
    for j,h in enumerate(hs):
        h=min(float(h),remaining-time)
        if h<=0:break
        start=states[j];end=b.step(start,us[j],h)
        cuts=[0.,h]
        # Hdot=u*p/4. Split at a momentum turning point to catch a
        # within-step energy crossing even when endpoint signs coincide.
        if start[1]*end[1]<0:
            cuts.insert(1,brentq(lambda t:b.step(start,us[j],t)[1],0,h,xtol=1e-14))
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            f=lambda t:float(b.energy(b.step(start,us[j],t))-TERMINAL_SECTION)
            a,c=f(lo),f(hi)
            if a==0:return time+lo
            if a*c<=0:
                root=brentq(f,lo,hi,xtol=1e-14)
                return time+root
        time+=h
        if time>=remaining:break
    return np.inf


def execute(x,control,lead,dt,subdivisions,remaining):
    crossing=terminal_crossing(x,control,lead,dt,subdivisions,remaining)
    answer=b.execute(x,control,lead,dt,subdivisions,min(remaining,crossing))
    if np.isfinite(crossing) and not answer[3]:
        assert b.ELO < b.energy(answer[0]) < b.EHI
        assert abs(b.energy(answer[0])-TERMINAL_SECTION)<1e-9
    return answer


def rollout(initial,lib,baseline,tlib,horizon=2000.,subdivisions=4,save_actions=False):
    x=initial.copy();t=active=effort=0.;actions=[];waits=[];number=nt=0
    maxe=float(b.energy(x));reason='horizon';hit=False;queries=eventcalls=0;first_action=None
    guard_rejections=terminal_stops=0;max_preimage_ratio=0.
    if b.target_gap(x)<=0:
        return dict(success=True,time=0.,active=0.,idle=0.,effort=0.,actions=0,transport_actions=0,
            max_energy=maxe,reason='initial_target',final=x.tolist(),first_action=None,
            guard_rejections=0,terminal_stops=0,max_preimage_ratio=0.),actions
    while t<horizon:
        terminal=b.score(x)<=0
        chosen=tlib if terminal else lib
        waiting,y,idx,nc=chosen.first_access(x);queries+=nc
        if not np.isfinite(waiting):
            reason='no_target_orbit_intersection' if terminal else 'no_support_orbit_intersection';break
        if t+waiting>horizon:reason='horizon_during_wait';break
        if terminal or b.target_gap(y)<=0:
            t+=waiting;waits.append(waiting);x=y
            assert b.target_gap(x)<=0
            hit=True;reason='passive_target';break
        source=int(chosen.ids[idx]);p=chosen.full
        parent=int(p['parent'][source]);u=p['control_flat'][p['offsets'][parent]:p['offsets'][parent+1]]
        prediction=None
        if p['is_transport'][source]:
            # All original assignments are the unchanged prefix, and retain
            # their original indices in the frozen reference payload.
            origin=int(p['original_source'][source])
            preimage=b.passive(y,-float(p['shift'][source]))
            ratio=float(b.distance(preimage,p['centers'][origin])/p['radii'][origin])
            max_preimage_ratio=max(max_preimage_ratio,ratio)
            # Recorded only: direct supports are certified without inverse-flow inclusion.
            prediction=execute(y,u,float(p['leads'][source]),float(p['dt']),subdivisions,horizon-t-waiting)
            end,elapsed,_,physical,_,_=prediction
            eligible=physical or b.score(end)<=0
            if not eligible:
                next_wait,_,next_index,nc=baseline.first_access(end);queries+=nc
                eligible=np.isfinite(next_wait) and next_index>=0
            if not eligible:
                # Reject before changing the physical state. Resume the
                # original policy from the actual current decision state.
                guard_rejections+=1;prediction=None;chosen=baseline
                waiting,y,idx,nc=baseline.first_access(x);queries+=nc
                if not np.isfinite(waiting):reason='no_original_continuation';break
                if t+waiting>horizon:reason='horizon_during_wait';break
                source=int(chosen.ids[idx]);p=chosen.full;parent=int(p['parent'][source])
                u=p['control_flat'][p['offsets'][parent]:p['offsets'][parent+1]]
        t+=waiting;waits.append(waiting);x=y
        if b.target_gap(x)<=0:hit=True;reason='passive_target';break
        assert b.distance(x,chosen.centers[idx])<=chosen.radii[idx]
        if first_action is None:first_action=float(t)
        before=x.copy();vbefore=float(b.score(x))
        rho=float(b.distance(x,chosen.centers[idx])/chosen.radii[idx])
        duration=float(p['leads'][source]+(len(u)-1)*float(p['dt']))
        result=prediction if prediction is not None else execute(x,u,float(p['leads'][source]),float(p['dt']),subdivisions,horizon-t)
        x,elapsed,cost,hit,peak,calls=result
        t+=elapsed;active+=elapsed;effort+=cost;number+=1;nt+=int(p['is_transport'][source]);maxe=max(maxe,peak);eventcalls+=calls
        if not hit and elapsed<duration-1e-10 and b.score(x)<=0:terminal_stops+=1
        if save_actions:actions.append([t-elapsed,waiting,elapsed,source,rho,vbefore,float(b.score(x)),before[0],before[1],x[0],x[1],int(hit),int(elapsed>=duration-1e-10)])
        if hit:reason='active_target';break
        if elapsed>=duration-1e-10 and b.score(x)>vbefore-b.V0*duration+b.ENDPOINT_BUDGET:
            raise RuntimeError('Completed primitive violated progress')
        assert maxe<=b.HX
        if number>100000:raise RuntimeError('Unexpected excessive action count')
    final_time=t if hit else horizon
    if not hit and t<horizon:
        try:x=b.passive(x,horizon-t)
        except ValueError:pass
    return dict(success=bool(hit),time=float(final_time),active=float(active),idle=float(final_time-active),
        effort=float(effort),actions=number,transport_actions=nt,max_energy=maxe,reason=reason,final=x.tolist(),
        max_completed_wait=float(max(waits,default=0)),open_wait=float(max(0,horizon-t)) if not hit else 0.,
        first_action=first_action,candidate_queries=queries,event_flow_calls=eventcalls,
        guard_rejections=guard_rejections,terminal_stops=terminal_stops,max_preimage_ratio=max_preimage_ratio),actions
