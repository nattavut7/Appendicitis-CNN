"""Paired patient-level permutation tests and Holm correction.

Never compare models tested on different patients with this paired protocol.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import classification_metrics


def matched_models(a,b):
    for x in (a,b):
        required={'patient_id','label','probability'}
        if not required<=set(x):raise ValueError('Each CSV must include patient_id,label,probability')
        if x.patient_id.duplicated().any():raise ValueError('Use patient_predictions.csv, not slice predictions')
    merged=a[['patient_id','label','probability']].merge(b[['patient_id','label','probability']],
             on='patient_id',suffixes=('_a','_b'),validate='one_to_one',how='inner')
    if len(merged)!=len(a) or len(merged)!=len(b):
        raise ValueError('Paired comparisons require identical patient IDs in both CSVs')
    if not np.array_equal(merged.label_a,merged.label_b):
        raise ValueError('Paired model labels disagree')
    return merged


def paired_permutation(y,a,b,metric='accuracy',iterations=2000,seed=2025):
    y=np.asarray(y);a=np.asarray(a);b=np.asarray(b)
    if len(set(y))<2 and metric=='auc':raise ValueError('AUC requires two classes')
    def score(p):
        result=classification_metrics(y,p)
        val=result[metric]
        if val is None:raise ValueError('Metric unavailable')
        return val
    obs=score(a)-score(b)
    rng=np.random.default_rng(seed)
    extreme=0
    for _ in range(iterations):
        mask=rng.random(len(y))<.5
        a2=np.where(mask,b,a);b2=np.where(mask,a,b)
        if abs(score(a2)-score(b2))>=abs(obs)-1e-12:
            extreme+=1
    return {'delta':float(obs),'p_value':float((extreme+1)/(iterations+1)),
            'iterations':iterations,'test':'paired_label_preserving_model_swap_permutation'}


def holm_correction(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p)
    out=np.zeros_like(p);n=len(p);prev=0
    for index,idx in enumerate(order):
        prev=max(prev,min(1.0,(n-index)*float(p[idx])))
        out[idx]=prev
    return out.tolist()


def compare_csv(a,b,out,metrics=('accuracy','auc','f1'),iterations=2000,seed=2025):
    joined=matched_models(pd.read_csv(a),pd.read_csv(b))
    rows=[]
    for metric in metrics:
        result=paired_permutation(joined.label_a,joined.probability_a,joined.probability_b,
                                  metric=metric,iterations=iterations,seed=seed)
        rows.append({'metric':metric,**result})
    adj=holm_correction([x['p_value'] for x in rows])
    for x,p in zip(rows,adj):x['holm_adjusted_p_value']=p
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps({'patients':len(joined),'comparisons':rows},indent=2),encoding='utf-8')
    return rows
