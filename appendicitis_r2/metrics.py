"""Classification, calibration, uncertainty and patient-bootstrap metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score, brier_score_loss,
                             roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def classification_metrics(y, p, cutoff=0.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if not len(y):
        raise ValueError('No predictions')
    yp=(p>=cutoff).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    result={'n':int(len(y)), 'accuracy':float(accuracy_score(y,yp)),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else None,
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'precision':float(precision_score(y,yp,zero_division=0)),
            'f1':float(f1_score(y,yp,zero_division=0)),
            'dice_positive':float((2*tp)/(2*tp+fp+fn)) if 2*tp+fp+fn else None,
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'brier':float(brier_score_loss(y,p)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    return result


def expected_calibration_error(y,p,bins=10):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    confidence=np.maximum(p,1-p)
    predicted=(p>=0.5).astype(int)
    edges=np.linspace(0,1,bins+1)
    out=[];ece=0.0
    for i in range(bins):
        idx=(confidence>=edges[i]) & ((confidence<=edges[i+1]) if i==bins-1 else (confidence<edges[i+1]))
        if not idx.any():
            continue
        acc=float(np.mean(predicted[idx]==y[idx]));conf=float(np.mean(confidence[idx]))
        ece+=idx.mean()*abs(acc-conf)
        out.append({'n':int(idx.sum()),'confidence':conf,'accuracy':acc})
    return float(ece),out


def plot_calibration(y,p,path,bins=10):
    ece,items=expected_calibration_error(y,p,bins)
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1],'--',label='Ideal calibration')
    if items:
        ax.plot([x['confidence'] for x in items],[x['accuracy'] for x in items],marker='o',label='Observed')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Empirical accuracy',title=f'Calibration (ECE={ece:.3f})')
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_roc(y,p,path):
    if len(set(y))<2: return
    fpr,tpr,_=roc_curve(y,p)
    fig,ax=plt.subplots(figsize=(5,5));ax.plot(fpr,tpr,label=f'AUC={roc_auc_score(y,p):.3f}')
    ax.plot([0,1],[0,1],'--');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Patient-level ROC')
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_confusion(y,p,path):
    mat=confusion_matrix(y,np.asarray(p)>=0.5,labels=[0,1])
    fig,ax=plt.subplots(figsize=(4,4));ax.imshow(mat,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(mat[i,j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title='Patient-level confusion matrix')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def aggregate_patients(frame):
    if frame.groupby('patient_id')['label'].nunique().max()>1:
        raise ValueError('Inconsistent patient labels')
    agg={'label':'first','probability':'mean'}
    if 'uncertainty' in frame: agg['uncertainty']='mean'
    return frame.groupby('patient_id',as_index=False).agg(agg)


def bootstrap_ci_patients(frame,iterations=1000,seed=2025):
    if iterations<50: raise ValueError('Bootstrap requires at least 50 iterations')
    patient=aggregate_patients(frame)
    rng=np.random.default_rng(seed)
    idx=np.arange(len(patient)); vals={x:[] for x in ['accuracy','auc','sensitivity','specificity','f1','dice_positive','brier']}
    for _ in range(iterations):
        pick=patient.iloc[rng.choice(idx,len(idx),replace=True)]
        m=classification_metrics(pick.label,pick.probability)
        for key in vals:
            if m[key] is not None: vals[key].append(m[key])
    return {key:[float(np.percentile(value,2.5)),float(np.percentile(value,97.5))]
            if value else None for key,value in vals.items()}


def risk_coverage(y,p,uncertainty,path):
    if uncertainty is None: return None
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);u=np.asarray(uncertainty,dtype=float)
    order=np.argsort(u,kind='stable');ok=(p>=.5).astype(int)==y
    keep=np.arange(1,len(y)+1)
    coverage=keep/len(y);risk=np.cumsum(~ok[order])/keep
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(coverage,risk)
    ax.set(xlabel='Retained coverage (lowest uncertainty first)',ylabel='Selective error rate',title='Risk–coverage')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    return {'aurc':float(np.trapezoid(risk,coverage) if hasattr(np,'trapezoid') else np.trapz(risk,coverage))}


def subgroup_metrics(frame,columns=('severity','age_group','image_quality')):
    out={}
    for col in columns:
        if col not in frame: continue
        out[col]={}
        for val, group in frame.dropna(subset=[col]).groupby(col):
            # Mark estimates with insufficient cases/descriptive only.
            out[col][str(val)]=classification_metrics(group['label'],group['probability'])
    return out


def write_report(frame,output_dir,bootstrap=1000,seed=2025,threshold=None):
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    frame.to_csv(root/'slice_predictions.csv',index=False)
    patient=aggregate_patients(frame)
    patient.to_csv(root/'patient_predictions.csv',index=False)
    m=classification_metrics(patient.label,patient.probability)
    m['ece'],m['calibration_bins']=expected_calibration_error(patient.label,patient.probability)
    m['bootstrap_95ci_patient_level']=bootstrap_ci_patients(frame,iterations=bootstrap,seed=seed)
    plot_calibration(patient.label,patient.probability,root/'calibration.png')
    plot_confusion(patient.label,patient.probability,root/'confusion.png')
    plot_roc(patient.label,patient.probability,root/'roc.png')
    if 'uncertainty' in patient:
        m['risk_coverage']=risk_coverage(patient.label,patient.probability,patient.uncertainty,root/'risk_coverage.png')
        if threshold is not None:
            referred=(patient.uncertainty>=threshold)
            m['validation_selected_referral_threshold']=float(threshold)
            m['test_referral_fraction']=float(referred.mean())
            m['retained_accuracy']=float(np.mean((patient.probability[~referred]>=.5)==patient.label[~referred])) if (~referred).any() else None
    merged=patient.merge(frame.drop_duplicates('patient_id')[[c for c in ('patient_id','severity','age_group','image_quality') if c in frame]],on='patient_id',how='left')
    m['subgroup_descriptive']=subgroup_metrics(merged)
    (root/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False),encoding='utf-8')
    return m
