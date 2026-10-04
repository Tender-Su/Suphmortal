"""Paired fixed-input evaluation for a single bounded calibration endpoint."""
import hashlib
import json
import os
from pathlib import Path
import time
from mortal.research.frozen_probe_cache import ProbeBudget, raw_metrics, reference_summary


class PairedBudget(ProbeBudget):
    def __init__(self, deadline_unix, *, stop_file=None):
        if not 0 < deadline_unix-time.time() <= 1200:
            raise ValueError('paired calibration requires its existing <=1200s experiment deadline')
        self.deadline=deadline_unix
        self.stop_unix=deadline_unix-120
        self.stop_monotonic=time.monotonic()+self.stop_unix-time.time()
        self.stop_file=Path(stop_file) if stop_file else None

    def manifest(self):
        return {**super().manifest(),'cleanup_reserve_seconds':120,'total_cap_seconds':1200,
                'scope':'same experiment clock as calibration; no reset for evaluation'}


def input_fingerprint(trajectory, target):
    import numpy as np
    values={**{k:trajectory[k] for k in ('obs','invisible_obs','actions','masks','at_kyoku','decision_indices')},
            'four_head_MC':target}
    result={}
    for name,value in values.items():
        value=np.ascontiguousarray(value)
        digest=hashlib.sha256(json.dumps([str(value.dtype),list(value.shape)]).encode())
        digest.update(value.tobytes())
        result[name]=digest.hexdigest()
    return result


def paired_block(games, targets, references, candidates, masks, *, replicates, seed):
    import numpy as np
    if not (len(games)==len(targets)==len(references)==len(candidates)==len(masks)) or len(games)%4:
        raise ValueError('paired block requires matched complete groups')
    group_sums=[];counts=[]
    kept_y=[];kept_a=[];kept_b=[]
    columns=['p0_bias','p0_mae','p0_mse','all_players_mse','p1_mse','p2_mse','p3_mse']
    for g in range(len(games)//4):
        sums=np.zeros(len(columns));count=0
        block=games[g*4:g*4+4]
        if {x['challenger_seat'] for x in block}!={0,1,2,3} or len({(x['seed'],x['seed_key']) for x in block})!=1:
            raise ValueError('paired seed-group identity differs')
        for i in range(g*4,g*4+4):
            y,a,b=(np.asarray(x[i],dtype=np.float64) for x in (targets,references,candidates))
            if y.shape!=a.shape or a.shape!=b.shape or y.ndim!=2 or y.shape[1]!=4:
                raise ValueError('two predictions must match identical four-head labels')
            keep=np.asarray(masks[i],dtype=bool)
            y,a,b=y[keep],a[keep],b[keep]
            if not len(y):continue
            if not all(np.isfinite(x).all() for x in (y,a,b)):raise ValueError('nonfinite paired inputs')
            ea,eb=a-y,b-y; delta=eb**2-ea**2
            sums+=np.array([(eb[:,0]-ea[:,0]).sum(),
                            (abs(eb[:,0])-abs(ea[:,0])).sum(),delta[:,0].sum(),
                            delta.mean(1).sum(),*delta[:,1:].sum(0)])
            count+=len(y);kept_y.append(y);kept_a.append(a);kept_b.append(b)
        group_sums.append(sums);counts.append(count)
    if not sum(counts):
        return {'states':0,'complete_seed_groups':len(group_sums),'paired_delta':None}
    ys,aa,bb=map(np.concatenate,(kept_y,kept_a,kept_b))
    sums,counts=np.array(group_sums),np.array(counts)
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,len(counts),size=(replicates,len(counts)))
    denominator=counts[selected].sum(1);valid=denominator>0
    draws=sums[selected[valid]].sum(1)/denominator[valid,None]
    return {'states':int(counts.sum()),'complete_seed_groups':len(counts),
            'groups_contributing_states':int((counts>0).sum()),
            'C0':raw_metrics(ys,aa),'C1':raw_metrics(ys,bb),'constant_zero':raw_metrics(ys,np.zeros_like(ys)),
            'paired_delta':{'direction':'C1-C0','columns':columns,
                'estimate':(sums.sum(0)/counts.sum()).tolist(),
                'ci95_low':np.quantile(draws,.025,axis=0).tolist(),
                'ci95_high':np.quantile(draws,.975,axis=0).tolist(),
                'replicates':replicates,'valid_replicates':int(valid.sum()),'seed':seed}}


def paired_summary(args,games,targets,references,candidates,contexts,advantages,groups):
    import numpy as np
    result=reference_summary(args,games,targets,references,contexts,advantages,groups)
    result['mode']='paired_calibration256'
    result['primary']='p0 MSE C1-C0; p0 is the controlled S70, not absolute seat0'
    result['no_automatic_extension']=True
    if result['status']!='complete':
        result['interpretation']='incomplete games or paired predictions; no planned estimate, no top-up'
        return result
    if len(candidates)!=len(games):raise ValueError('missing candidate predictions')
    def block(masks):
        return paired_block(games,targets,references,candidates,masks,
                            replicates=args.bootstrap_replicates,seed=args.bootstrap_seed)
    result['paired']=block([np.ones(len(y),dtype=bool) for y in targets])
    result['by_controlled_absolute_seat']={str(seat):block([
        np.full(len(y),game['challenger_seat']==seat,dtype=bool)
        for game,y in zip(games,targets)]) for seat in range(4)}
    result['paired_input_strata']={f'current_rank={rank},all_last={last}':block([
        (c[:,4]==rank)&(c[:,3]==last) for c in contexts]) for rank in range(4) for last in (0,1)}
    result['interpretation']='one fixed endpoint; all_players gains from opponents do not establish controlled-S70 improvement; no actor promotion'
    return result
