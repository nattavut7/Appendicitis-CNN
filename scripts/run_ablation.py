"""Actual ablation runner; generates NEW empirical results only when real data exists.

Examples:
  python scripts/run_ablation.py --config configs/r2.yaml --mode selected --runs 5
  python scripts/run_ablation.py --config configs/r2.yaml --mode full --runs 5
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appendicitis_r2.config import load_config
from appendicitis_r2.engine import fit
from appendicitis_r2.statistics import compare_csv

SELECTED=['none','rotation','crop','noise','rotation_crop','rotation_crop_noise']
EXTENDED=['table2_full']+[f'full_without_{x}' for x in ['rotation','flip','affine','elastic','brightness','contrast','noise','cutmix','mixup']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/r2.yaml')
    p.add_argument('--mode',choices=['selected','full','both'],default='selected')
    p.add_argument('--runs',type=int,default=5);p.add_argument('--output',default='results/ablation')
    args=p.parse_args();c=load_config(args.config)
    profiles=(SELECTED if args.mode in ('selected','both') else [])+(EXTENDED if args.mode in ('full','both') else [])
    if args.runs<1:raise ValueError('runs >=1 required')
    all_rows=[];references={}
    for profile in profiles:
        for run in range(args.runs):
            cfg=copy.deepcopy(c);cfg['seed']=int(c['seed'])+run
            cfg['augmentation']['profile']=profile
            out=Path(args.output)/profile/f'run_{run+1}'
            print(f'RUN profile={profile} run={run+1}/{args.runs}',flush=True)
            metrics=fit(cfg,output_dir=out)
            all_rows.append({'profile':profile,'run':run+1,'seed':cfg['seed'],
                **{k:metrics.get(k) for k in ('accuracy','auc','dice_positive','sensitivity','specificity','f1','ece','n')}})
            references[(profile,run)]=out/'test_evaluation'/'patient_predictions.csv'
            pd.DataFrame(all_rows).to_csv(Path(args.output)/'ablation_results.csv',index=False)
    # Compare each profile against the corresponding reference profile for
    # each seed using the SAME held-out patient IDs, then Holm adjustment.
    stats=[]
    for profile in profiles:
        ref='rotation_crop_noise' if profile in SELECTED else 'table2_full'
        if profile==ref or ref not in profiles:continue
        for run in range(args.runs):
            dest=Path(args.output)/'comparisons'/f'{ref}_vs_{profile}_run{run+1}.json'
            for item in compare_csv(references[(ref,run)],references[(profile,run)],dest,
                                    iterations=int(c['statistical']['permutation_iterations']),seed=int(c['seed'])+run):
                stats.append({'reference':ref,'variant':profile,'run':run+1,**item})
    Path(args.output).mkdir(parents=True,exist_ok=True)
    if stats:pd.DataFrame(stats).to_csv(Path(args.output)/'paired_tests.csv',index=False)
    pd.DataFrame(all_rows).groupby('profile').agg({k:['mean','std'] for k in ['accuracy','auc','dice_positive']}).to_csv(Path(args.output)/'summary_mean_std.csv')
    print('Empirical ablations complete:',Path(args.output).resolve())

if __name__=='__main__':main()
