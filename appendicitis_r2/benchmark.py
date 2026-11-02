"""Real-device latency and dynamic quantization; NO fabricated numbers.

Masked pruning does not change dense parameter shapes, hence is not labeled
structural compression. Baseline/QAT claims must be backed by actual experiments.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .model import create_model,count_parameters
from .engine import device_auto,eval_pass
from .config import set_seed
from .data import CTDataset,make_loader,validate_manifest
from .metrics import aggregate_patients,classification_metrics
import pandas as pd


def latency(model,device,input_shape,warmup=8,runs=30):
    model.eval();x=torch.rand(*input_shape,device=device)
    with torch.inference_mode():
        for _ in range(warmup):model(x)
        if device.type=='cuda':torch.cuda.synchronize()
        times=[]
        for _ in range(runs):
            start=time.perf_counter()
            model(x)
            if device.type=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    return {'latency_ms_median':float(np.median(times)), 'latency_ms_p95':float(np.percentile(times,95)),
            'batch_size':input_shape[0], 'warmup':warmup,'runs':runs}


def analyze(checkpoint,manifest=None,out='results/benchmark',device='cpu'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    c=saved['config'];dev=device_auto(device)
    if manifest:c['manifest']=manifest
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    original=create_model(c).to(dev)
    original.load_state_dict(saved['model_state'])
    size=int(c['data']['image_size']);chs=int(c['data']['channels'])
    table=[]
    base={'variant':'float32','trainable_parameters':count_parameters(original),
          'device':str(dev),**latency(original,dev,(1,chs,size,size))}
    if dev.type=='cuda':
        base['peak_memory_bytes'] = None
        torch.cuda.reset_peak_memory_stats()
        latency(original,dev,(1,chs,size,size),warmup=2,runs=5)
        base['peak_memory_bytes']=int(torch.cuda.max_memory_allocated())
    table.append(base)
    # Dynamic quantization only affects Linear layers, not convolutions.
    # It is a documented, functioning compression baseline, NOT QAT.
    if dev.type=='cpu':
        try:
            q=torch.ao.quantization.quantize_dynamic(copy.deepcopy(original).cpu(),{nn.Linear},dtype=torch.qint8)
            table.append({'variant':'dynamic_int8_linear_only',
                'trainable_parameters':None,'device':'cpu',
                **latency(q,torch.device('cpu'),(1,chs,size,size))})
        except (RuntimeError,NotImplementedError,AttributeError) as err:
            table.append({'variant':'dynamic_int8_linear_only','error':str(err)})
    frame=validate_manifest(pd.read_csv(c['manifest'],dtype={'patient_id':str}))
    loader=make_loader(CTDataset(frame,'test',c),int(c['train']['batch_size']))
    set_seed(int(c['seed'])+100001)
    original_pred=aggregate_patients(eval_pass(original,loader,dev,mc_passes=int(c['uncertainty']['mc_passes'])))
    base['patient_metrics']=classification_metrics(original_pred.label,original_pred.probability)
    # Quantized inference on the same test patients. MC dropout variance on
    # dynamic quantization is reported only if execution is supported.
    if dev.type=='cpu' and any(row['variant']=='dynamic_int8_linear_only' and 'error' not in row for row in table):
        try:
            set_seed(int(c['seed'])+100001)
            qpred=aggregate_patients(eval_pass(q,loader,torch.device('cpu'),mc_passes=int(c['uncertainty']['mc_passes'])))
            table[1]['patient_metrics']=classification_metrics(qpred.label,qpred.probability)
            table[1]['mean_predictive_variance']=float(qpred.uncertainty.mean())
        except (RuntimeError,NotImplementedError) as err:
            table[1]['evaluation_error']=str(err)
    (root/'benchmarks.json').write_text(json.dumps({'caveat':
        'Latency depends on this machine; CPU quantization is dynamic Linear-only, NOT quantization-aware training or proof of edge deployment.',
        'results':table},indent=2),encoding='utf-8')
    return table
