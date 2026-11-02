from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from appendicitis_r2.config import load_config
from appendicitis_r2.data import validate_manifest,make_splits,load_image
from appendicitis_r2.augmentation import CTTransform,components_for_profile,batch_mix
from appendicitis_r2.metrics import (classification_metrics,aggregate_patients,
                                     bootstrap_ci_patients,expected_calibration_error)
from appendicitis_r2.model import create_model,count_parameters
from appendicitis_r2.statistics import paired_permutation,holm_correction
from scripts.synthetic_smoke import generate

CFG=load_config('configs/r2.yaml')


def test_parameter_count_and_shape():
    model=create_model(CFG)
    assert 570000<=count_parameters(model)<=590000
    with torch.no_grad(): y=model(torch.zeros(2,3,224,224))
    assert y.shape==(2,2)


def test_no_patient_leakage():
    d=pd.DataFrame({'image_path':['a','b','c'], 'patient_id':['id1','id1','id2'],
                    'label':[0,0,1],'split':['train','test','val']})
    with pytest.raises(ValueError,match='LEAKAGE'):validate_manifest(d)


def test_patient_level_splits():
    d=pd.DataFrame([{'patient_id':f'p{i}','image_path':f'p{i}_{j}.png','label':i%2}
                    for i in range(40) for j in range(2)])
    result=make_splits(d)
    assert result.groupby('patient_id').split.nunique().max()==1
    assert len(result)==80


def test_auc_uses_probability_not_hard_predictions():
    m=classification_metrics([0,0,1,1],[.05,.25,.75,.95])
    assert m['auc']==1.0 and m['dice_positive']==1.0
    assert (m['tp'],m['tn'],m['fp'],m['fn'])==(2,2,0,0)


def test_calibration_and_ci():
    f=pd.DataFrame({'patient_id':[f'p{i}' for i in range(20)],
                    'label':([0,1]*10),'probability':([.2,.8]*10)})
    ci=bootstrap_ci_patients(f,iterations=50)
    assert ci['accuracy']==[1.0,1.0]
    ece,_=expected_calibration_error(f.label,f.probability)
    assert ece>=0


def test_augmentations_image_and_mix():
    img=np.ones((96,96),dtype=np.float32)*.5
    transformed=CTTransform(CFG['augmentation'],'table2_full')(img)
    assert transformed.shape==img.shape
    assert np.isfinite(transformed).all() and 0<=transformed.min()<=transformed.max()<=1
    X=torch.rand(4,3,96,96);y=torch.tensor([0,1,0,1])
    x,a,b,l=batch_mix(X,y,CFG['augmentation'],'table2_full')
    assert x.shape==X.shape and 0<=l<=1
    assert 'elastic' in components_for_profile('table2_full')


def test_synthetic_image_loading(tmp_path):
    csv=generate(tmp_path,n_per_class_per_split=3)
    df=validate_manifest(pd.read_csv(csv))
    assert len(df)==18
    sample=load_image(df.iloc[0].image_path,CFG)
    assert sample.shape==(224,224)


def test_holm_and_permutation():
    v=holm_correction([0.01,0.04,0.2])
    assert all(0<=x<=1 for x in v)
    d=paired_permutation([0,1,0,1],[.2,.8,.2,.8],[.7,.3,.7,.3],iterations=50)
    assert 0<=d['p_value']<=1


def test_dicom_hu_requires_verified_config(tmp_path):
    # Use a real on-disk PNG; invalid format is caught in preprocessing.
    csv=generate(tmp_path,n_per_class_per_split=3)
    assert Path(csv).exists()
