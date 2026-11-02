"""Training/evaluation; all results recomputed from patient-safe manifests."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .augmentation import CTTransform,batch_mix,components_for_profile
from .config import provenance,save_config,set_seed
from .data import CTDataset,make_loader,validate_manifest
from .metrics import (aggregate_patients,classification_metrics,write_report)
from .model import create_model,count_parameters


def device_auto(explicit=None):
    if explicit and explicit!='auto':
        return torch.device(explicit)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def curriculum_allowed(df,epoch,total,cfg):
    if not cfg['train'].get('curriculum',False):
        return df
    if 'difficulty' not in df:
        raise ValueError('Curriculum requires a difficulty column: easy/moderate/hard')
    col=df['difficulty'].astype(str).str.lower()
    if not col.isin(['easy','moderate','hard']).all():
        raise ValueError('Every training case must have a verified easy/moderate/hard label')
    # Maintain monotone introduction by training epoch fractions.
    fraction=(epoch+1)/total
    if fraction<=float(cfg['train']['curriculum_easy_until']):
        return df[col=='easy']
    if fraction<=float(cfg['train']['curriculum_moderate_until']):
        return df[col.isin(['easy','moderate'])]
    return df


def eval_pass(model,loader,device,mc_passes=1):
    model.eval()
    rows=[]
    for images,labels,patient_ids,paths in loader:
        images=images.to(device)
        with torch.no_grad():
            if mc_passes>1:
                # Keep only dropout enabled; batchnorm continues to use eval statistics.
                for module in model.modules():
                    if isinstance(module,nn.Dropout): module.train()
                probabilities=torch.stack([model(images).softmax(-1)[:,1] for _ in range(mc_passes)],dim=0)
                probs=probabilities.mean(0).cpu().numpy()
                variances=probabilities.var(0,unbiased=False).cpu().numpy()
                model.eval()
            else:
                probs=model(images).softmax(-1)[:,1].cpu().numpy()
                variances=np.full_like(probs,np.nan)
        for j in range(len(probs)):
            rows.append({'image_path':str(paths[j]),'patient_id':str(patient_ids[j]),
                         'label':int(labels[j]),'probability':float(probs[j]),
                         'uncertainty':float(variances[j])})
    return pd.DataFrame(rows)


def _criterion(c,training_df,device):
    if c['train'].get('class_weighting'):
        counts=training_df['label'].value_counts().reindex([0,1],fill_value=0).to_numpy()
        if (counts==0).any(): raise ValueError('Both training classes required')
        w=counts.sum()/(2*counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    return nn.CrossEntropyLoss()


def fit(config,output_dir=None,device=None):
    c=copy.deepcopy(config)
    out=Path(output_dir or c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    set_seed(int(c['seed']))
    frame=validate_manifest(pd.read_csv(c['manifest'],dtype={'patient_id':str}))
    dev=device_auto(device)
    model=create_model(c).to(dev)
    profile=c['augmentation']['profile']
    components_for_profile(profile)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(c['train']['lr']),weight_decay=float(c['train']['weight_decay']))
    batch_size=int(c['train']['batch_size'])
    nworkers=int(c['data']['num_workers'])
    val_ds=CTDataset(frame,'val',c)
    val_loader=make_loader(val_ds,batch_size,num_workers=nworkers)
    train_df=frame[frame.split=='train'].copy()
    loss_fn=_criterion(c,train_df,dev)
    history=[];best_val=float('inf');best_epoch=0
    patience=int(c['train']['early_stopping_patience'])
    epochs=int(c['train']['epochs'])
    if epochs<1: raise ValueError('epochs must be >=1')
    best_path=out/'checkpoint_best.pt'
    save_config(c,out/'config_resolved.yaml')
    (out/'environment.json').write_text(json.dumps(provenance(),indent=2),encoding='utf-8')
    (out/'manifest_summary.json').write_text(json.dumps(frame.groupby(['split','label']).size().rename('images').reset_index().to_dict('records'),indent=2),encoding='utf-8')
    for epoch in range(epochs):
        epoch_df=curriculum_allowed(train_df,epoch,epochs,c)
        if epoch_df.empty:
            raise ValueError(f'No curriculum cases available at epoch {epoch+1}')
        train_ds=CTDataset(epoch_df,'train',c,augment=CTTransform(c['augmentation'],profile))
        loader=make_loader(train_ds,batch_size,shuffle=True,num_workers=nworkers)
        model.train();loss_sum=0;count=0
        for images,labels,_,_ in loader:
            images=images.to(dev,non_blocking=True)
            labels=labels.to(dev,non_blocking=True)
            images,labels_a,labels_b,lam=batch_mix(images,labels,c['augmentation'],profile)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=lam*loss_fn(logits,labels_a)+(1-lam)*loss_fn(logits,labels_b)
            loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(images);count+=len(images)
        model.eval();valid_loss=0;val_count=0
        with torch.no_grad():
            for images,labels,_,_ in val_loader:
                images=images.to(dev);labels=labels.to(dev)
                l=loss_fn(model(images),labels)
                valid_loss+=l.item()*len(images);val_count+=len(images)
        val_loss=valid_loss/val_count
        row={'epoch':epoch+1,'train_loss':loss_sum/count,'val_loss':val_loss,
             'train_images':len(epoch_df),'train_patients':int(epoch_df.patient_id.nunique()),
             'curriculum_stage':('easy' if len(epoch_df)<len(train_df) and
                 (epoch+1)/epochs<=float(c['train']['curriculum_easy_until']) else
                 'easy_moderate' if len(epoch_df)<len(train_df) else 'all')}
        history.append(row)
        print(f"Epoch {epoch+1:03d}: train {row['train_loss']:.4f} val {val_loss:.4f} stage={row['curriculum_stage']}",flush=True)
        if val_loss<best_val-1e-8:
            best_val=val_loss;best_epoch=epoch+1
            torch.save({'model_state':model.state_dict(),'config':c,'epoch':best_epoch,
                        'parameter_count':count_parameters(model),'validation_loss':best_val},best_path)
        if epoch+1-best_epoch>=patience:
            print('Early stopping triggered')
            break
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    # Reload BEST checkpoint; never select best model using test set.
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=False)['model_state'])
    set_seed(int(c['seed'])+100000)  # fixed MC-dropout validation sequence
    valid_pred=eval_pass(model,val_loader,dev,mc_passes=int(c['uncertainty']['mc_passes']))
    valid_pred.to_csv(out/'validation_predictions.csv',index=False)
    patient_val=aggregate_patients(valid_pred)
    quantile=float(c['uncertainty']['validation_referral_quantile'])
    if not 0<quantile<1: raise ValueError('validation_referral_quantile must be in (0,1)')
    referral=float(patient_val.uncertainty.quantile(quantile))
    (out/'referral_threshold_from_validation.json').write_text(json.dumps({'threshold':referral,'quantile':quantile,'status':'research_threshold_not_clinically_validated'},indent=2),encoding='utf-8')
    test_loader=make_loader(CTDataset(frame,'test',c),batch_size,num_workers=nworkers)
    set_seed(int(c['seed'])+100001)  # fixed MC-dropout held-out test sequence
    pred=eval_pass(model,test_loader,dev,mc_passes=int(c['uncertainty']['mc_passes']))
    # Add subgroup metadata only if recorded at patient level.
    cols=['image_path','severity','age_group','image_quality']
    cols=[s for s in cols if s in frame]
    if len(cols)>1:
        pred=pred.merge(frame[cols],on='image_path',how='left',validate='one_to_one')
    results=write_report(pred,out/'test_evaluation',bootstrap=int(c['statistical']['bootstrap_iterations']),seed=int(c['seed']),threshold=referral)
    print('Best epoch:',best_epoch,'Trainable parameters:',count_parameters(model))
    print('TEST results from REAL manifest:',json.dumps({k:results[k] for k in ('accuracy','auc','f1','ece','n')},indent=2))
    return results


def evaluate_checkpoint(checkpoint,manifest=None,output_dir=None,device=None):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    c=saved['config'];set_seed(int(c['seed']))
    if manifest:c['manifest']=manifest
    frame=validate_manifest(pd.read_csv(c['manifest'],dtype={'patient_id':str}))
    dev=device_auto(device);model=create_model(c).to(dev)
    model.load_state_dict(saved['model_state'])
    out=Path(output_dir or Path(checkpoint).parent/'reevaluation')
    test_loader=make_loader(CTDataset(frame,'test',c),int(c['train']['batch_size']))
    set_seed(int(c['seed'])+100001)  # fixed MC-dropout held-out test sequence
    pred=eval_pass(model,test_loader,dev,mc_passes=int(c['uncertainty']['mc_passes']))
    if {'image_path','severity'}<=set(frame):
        cols=['image_path']+[col for col in ('severity','age_group','image_quality') if col in frame]
        pred=pred.merge(frame[cols],on='image_path',how='left',validate='one_to_one')
    thresholdfile=Path(checkpoint).parent/'referral_threshold_from_validation.json'
    thr=json.loads(thresholdfile.read_text())['threshold'] if thresholdfile.exists() else None
    return write_report(pred,out,bootstrap=int(c['statistical']['bootstrap_iterations']),seed=int(c['seed']),threshold=thr)
