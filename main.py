"""Reproducible R2 command line entry point."""
import argparse
import json
from pathlib import Path
import pandas as pd
from appendicitis_r2.config import load_config
from appendicitis_r2.data import make_splits,validate_manifest,manifest_summary
from appendicitis_r2.engine import fit,evaluate_checkpoint
from appendicitis_r2.model import create_model,count_parameters
from appendicitis_r2.statistics import compare_csv
from appendicitis_r2.benchmark import analyze
from appendicitis_r2.interpretability import gradcam
from appendicitis_r2.data import load_image
from appendicitis_r2.config import set_seed
from appendicitis_r2.engine import device_auto
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description='Appendicitis-CNN R2 reproducibility toolkit')
    sub=parser.add_subparsers(dest='command',required=True)
    count=sub.add_parser('inspect-model');count.add_argument('--config',default='configs/r2.yaml')
    split=sub.add_parser('split');split.add_argument('--input',required=True);split.add_argument('--output',required=True);split.add_argument('--seed',type=int,default=2025)
    audit=sub.add_parser('audit');audit.add_argument('--manifest',required=True)
    train=sub.add_parser('train');train.add_argument('--config',default='configs/r2.yaml');train.add_argument('--output');train.add_argument('--device',default='auto')
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--checkpoint',required=True);evaluate.add_argument('--manifest');evaluate.add_argument('--output');evaluate.add_argument('--device',default='auto')
    compare=sub.add_parser('compare');compare.add_argument('--a',required=True);compare.add_argument('--b',required=True);compare.add_argument('--output',required=True);compare.add_argument('--iterations',type=int,default=2000)
    bench=sub.add_parser('benchmark');bench.add_argument('--checkpoint',required=True);bench.add_argument('--manifest');bench.add_argument('--output',default='results/benchmark');bench.add_argument('--device',default='cpu')
    infer=sub.add_parser('infer');infer.add_argument('--checkpoint',required=True);infer.add_argument('--image',required=True);infer.add_argument('--device',default='auto');infer.add_argument('--roi')
    cam=sub.add_parser('gradcam');cam.add_argument('--checkpoint',required=True);cam.add_argument('--image',required=True);cam.add_argument('--output',required=True);cam.add_argument('--roi')
    opts=parser.parse_args()
    if opts.command=='inspect-model':
        c=load_config(opts.config);model=create_model(c)
        print(json.dumps({'architecture':c['model']['architecture'],'parameters':count_parameters(model),'model':str(model)},indent=2))
    elif opts.command=='split':
        frame=make_splits(pd.read_csv(opts.input,dtype={'patient_id':str}),seed=opts.seed)
        Path(opts.output).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(opts.output,index=False)
        print(json.dumps(manifest_summary(frame),indent=2))
    elif opts.command=='audit':
        frame=validate_manifest(pd.read_csv(opts.manifest,dtype={'patient_id':str}))
        print(json.dumps(manifest_summary(frame),indent=2))
    elif opts.command=='train':
        fit(load_config(opts.config),output_dir=opts.output,device=opts.device)
    elif opts.command=='evaluate':
        print(json.dumps(evaluate_checkpoint(opts.checkpoint,opts.manifest,opts.output,opts.device),indent=2))
    elif opts.command=='compare':
        print(json.dumps(compare_csv(opts.a,opts.b,opts.output,iterations=opts.iterations),indent=2))
    elif opts.command=='benchmark':
        print(json.dumps(analyze(opts.checkpoint,opts.manifest,opts.output,opts.device),indent=2))
    elif opts.command=='infer':
        saved=torch.load(opts.checkpoint,map_location='cpu',weights_only=False)
        c=saved['config'];set_seed(int(c['seed'])+100001)
        device=device_auto(opts.device);model=create_model(c).to(device)
        model.load_state_dict(saved['model_state']);model.eval()
        arr=load_image(opts.image,c,roi=opts.roi)
        x=torch.tensor(np.repeat(arr[None,None],int(c['data']['channels']),axis=1),dtype=torch.float32,device=device)
        ps=[]
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,torch.nn.Dropout):module.train()
            for _ in range(int(c['uncertainty']['mc_passes'])):
                ps.append(model(x).softmax(-1)[0,1].item())
        print(json.dumps({'research_only':True,'class_1_probability':float(np.mean(ps)),
             'mc_predictive_variance':float(np.var(ps)),
             'label':int(np.mean(ps)>=0.5),'mc_passes':len(ps)},indent=2))
    elif opts.command=='gradcam':
        print(json.dumps(gradcam(opts.checkpoint,opts.image,opts.output,opts.roi),indent=2))

if __name__=='__main__':main()
