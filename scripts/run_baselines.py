"""Same config, splits, augmentation profile and curriculum for all baselines.

Use identical seeds / compute budget for an auditable, fair comparison. All
weights are trained from scratch unless methodology is explicitly changed.
"""
import argparse
import copy
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appendicitis_r2.config import load_config
from appendicitis_r2.engine import fit

MODELS=['r2_cnn','mobilenet_v2','efficientnet_b0','efficientnet_b2','inception_v3','vgg19']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/r2.yaml')
    p.add_argument('--models',nargs='+',default=MODELS,choices=MODELS)
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--output',default='results/baselines')
    args=p.parse_args();base=load_config(args.config)
    rows=[]
    for model in args.models:
        for run in range(args.runs):
            c=copy.deepcopy(base);c['model']['architecture']=model;c['seed']=int(base['seed'])+run
            out=Path(args.output)/model/f'run_{run+1}'
            metrics=fit(c,output_dir=out)
            rows.append({'model':model,'seed':c['seed'],**{k:metrics.get(k) for k in ('accuracy','auc','dice_positive','f1','ece','n')}})
            pd.DataFrame(rows).to_csv(Path(args.output)/'baseline_results.csv',index=False)
    pd.DataFrame(rows).groupby('model').agg({k:['mean','std'] for k in ('accuracy','auc','dice_positive')}).to_csv(Path(args.output)/'baseline_summary.csv')

if __name__=='__main__':main()
