"""SYNTHETIC image data for running CI and smoke tests ONLY.

Not real CT, not a clinical dataset and NEVER to be used for paper metrics.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def generate(root,seed=2025,n_per_class_per_split=3):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);rows=[]
    for split in ('train','val','test'):
        for label in (0,1):
            for j in range(n_per_class_per_split):
                pid=f'synthetic_{split}_{label}_{j:03d}'
                arr=np.uint8(rng.uniform(0,225,size=(80,80)))
                arr[20:45,20:45]=170 if label else 40
                path=root/f'{pid}.png';Image.fromarray(arr).save(path)
                rows.append({'image_path':str(path.resolve()),'patient_id':pid,'label':label,
                             'split':split,'difficulty':('easy','moderate','hard')[j%3],
                             'severity':'SYNTHETIC','age_group':'SYNTHETIC','image_quality':'SYNTHETIC'})
    csv=root/'manifest_synthetic.csv'
    pd.DataFrame(rows).to_csv(csv,index=False)
    (root/'DO_NOT_USE_FOR_RESEARCH.txt').write_text('SYNTHETIC TEST FIXTURE. Not medical images or evidence for the manuscript.\n')
    return csv

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/synthetic_smoke')
    args=p.parse_args();print(generate(args.output))
