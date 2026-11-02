"""Local-only dataset audit; does not upload CT scans, paths or patient identifiers."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appendicitis_r2.data import validate_manifest,manifest_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--image-root',default='.')
    p.add_argument('--hash-files',action='store_true',help='Slow scan for duplicate image content across patient/splits')
    p.add_argument('--output',required=True,help='Private local JSON destination (do not push to GitHub)')
    a=p.parse_args()
    frame=validate_manifest(pd.read_csv(a.manifest,dtype={'patient_id':str}))
    out={'patient_safe_splits':True,'counts':manifest_summary(frame),
         'unique_patients':int(frame.patient_id.nunique()),
         'image_rows':len(frame),'content_duplicates_across_splits':None,
         'note':'No patient identifiers or paths have been included in this summary.'}
    if a.hash_files:
        seen=defaultdict(set);missing=0
        for _,row in frame.iterrows():
            path=Path(str(row.image_path))
            if not path.is_absolute():path=Path(a.image_root)/path
            if not path.is_file():missing+=1;continue
            sha=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1048576),b''):sha.update(chunk)
            seen[sha.hexdigest()].add((str(row.patient_id),str(row.split)))
        cross=sum(1 for x in seen.values() if len({v[1] for v in x})>1)
        out['content_duplicates_across_splits']=cross
        out['missing_images']=missing
        if cross:out['warning']='Potential duplicate-image leakage across splits; investigate locally before training.'
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
