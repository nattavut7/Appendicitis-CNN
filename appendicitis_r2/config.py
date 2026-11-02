"""Configuration, deterministic seeding, and experiment provenance."""
from __future__ import annotations
import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding='utf-8') as handle:
        c = yaml.safe_load(handle)
    if not isinstance(c, dict):
        raise ValueError('YAML config must contain a mapping')
    return c


def save_config(config, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False


def provenance():
    def git_head():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'git_commit': git_head()}
