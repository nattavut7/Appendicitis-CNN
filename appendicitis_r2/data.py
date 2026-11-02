"""Patient-safe CT manifest data access; no patient data bundled in GitHub."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader

REQUIRED = ('image_path', 'patient_id', 'label', 'split')
ALLOWED_SPLITS = {'train', 'val', 'test'}


def validate_manifest(frame):
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f'Manifest missing required columns: {sorted(missing)}')
    if len(frame) == 0:
        raise ValueError('Manifest is empty')
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError('Missing image path, patient ID, label or split')
    df = frame.copy()
    df['patient_id'] = df['patient_id'].astype(str)
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype(int)
    df['split'] = df['split'].astype(str).str.lower()
    if not set(df['label'].unique()).issubset({0, 1}):
        raise ValueError('Labels must be binary, 0=negative, 1=appendicitis')
    if not set(df['split'].unique()).issubset(ALLOWED_SPLITS):
        raise ValueError('split must be train/val/test')
    if df.duplicated(subset=['image_path']).any():
        raise ValueError('Duplicate image_path entries')
    if df.groupby('patient_id')['split'].nunique().max() > 1:
        raise ValueError('LEAKAGE: a patient_id appears in more than one split')
    if df.groupby('patient_id')['label'].nunique().max() > 1:
        raise ValueError('Contradictory labels for one patient')
    if {'train', 'val', 'test'} - set(df['split']):
        raise ValueError('Train, validation, and test partitions are all required')
    return df


def manifest_summary(df):
    return df.groupby(['split', 'label']).agg(images=('image_path', 'size'),
        patients=('patient_id', 'nunique')).reset_index().to_dict(orient='records')


def make_splits(frame, seed=2025, train_fraction=2/3, val_fraction=2/15):
    """Patient-level stratified partitions; never split individual slices."""
    from sklearn.model_selection import train_test_split
    if not {'image_path', 'patient_id', 'label'} <= set(frame.columns):
        raise ValueError('Need image_path, patient_id, label')
    df = frame.copy()
    patient = df.groupby('patient_id', as_index=False)['label'].agg('first')
    if (df.groupby('patient_id')['label'].nunique() > 1).any():
        raise ValueError('Patient has discordant labels')
    ids = patient['patient_id'].to_numpy()
    y = patient['label'].to_numpy()
    try:
        train, rest = train_test_split(ids, train_size=train_fraction, random_state=seed, stratify=y)
        rest_patient = patient.set_index('patient_id').loc[rest]
        val, test = train_test_split(rest, train_size=val_fraction / (1-train_fraction),
                                     random_state=seed, stratify=rest_patient['label'])
    except ValueError as exc:
        raise ValueError('Insufficient patients per class for stratified splits') from exc
    assignment = {str(pid): group for group, members in [('train', train), ('val', val), ('test', test)]
                  for pid in members}
    df['split'] = df['patient_id'].astype(str).map(assignment)
    return validate_manifest(df)


def load_image(path, config, roi=None):
    """Return a float image [H,W] in [0,1]. DICOM HU range must be explicit."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    d = config['data']
    if path.suffix.lower() in {'.dcm', '.dicom'}:
        import pydicom
        ds = pydicom.dcmread(str(path))
        if ds.pixel_array.ndim != 2:
            raise ValueError('Only individual 2-D CT slices are supported: '+str(path))
        if not hasattr(ds, 'RescaleSlope') or not hasattr(ds, 'RescaleIntercept'):
            raise ValueError('DICOM missing RescaleSlope/RescaleIntercept (cannot establish HU)')
        arr = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        hu = d.get('hu_window')
        if hu is None or len(hu) != 2 or float(hu[0]) >= float(hu[1]):
            raise ValueError('Set data.hu_window to the ACTUAL study HU clipping limits')
        arr = np.clip(arr, float(hu[0]), float(hu[1]))
        arr = (arr-float(hu[0])) / (float(hu[1])-float(hu[0]))
        target_spacing = d.get('target_spacing_mm')
        if target_spacing is not None:
            if not hasattr(ds, 'PixelSpacing'):
                raise ValueError('DICOM missing PixelSpacing')
            old_spacing = np.array([float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])])
            new_spacing = np.asarray(target_spacing, dtype=float)
            if new_spacing.shape != (2,) or np.any(new_spacing <= 0):
                raise ValueError('target_spacing_mm should be [row_mm, col_mm] > 0')
            arr = zoom(arr, old_spacing/new_spacing, order=1)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            # Pillow supports non-ASCII Windows paths.
            from PIL import Image
            arr = np.asarray(Image.open(path).convert('L'))
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr[:,:,:3], cv2.COLOR_BGR2GRAY)
        arr = arr.astype(np.float32)
        if d.get('non_dicom_normalization') != 'per_image_minmax':
            raise ValueError('Unsupported non-DICOM pixel scale')
        lo, hi = float(np.min(arr)), float(np.max(arr))
        arr = (arr-lo) / (hi-lo) if hi > lo else np.zeros_like(arr)
    if roi is not None and str(roi).strip() and str(roi).lower() != 'nan':
        coords = [int(t.strip()) for t in str(roi).split(',')]
        if len(coords) != 4:
            raise ValueError('roi_xyxy must be x0,y0,x1,y1')
        x0,y0,x1,y1 = coords
        if not (0<=x0<x1<=arr.shape[1] and 0<=y0<y1<=arr.shape[0]):
            raise ValueError('roi_xyxy outside image')
        arr = arr[y0:y1,x0:x1]
    elif d.get('require_roi', False):
        raise ValueError('ROI bounding box required but missing')
    mode = str(d.get('histogram_equalization', 'none')).lower()
    if mode != 'none':
        arr8 = np.round(np.clip(arr,0,1)*255).astype(np.uint8)
        if mode == 'global':
            arr = cv2.equalizeHist(arr8).astype(np.float32)/255
        elif mode == 'clahe':
            arr = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(arr8).astype(np.float32)/255
        else:
            raise ValueError(f'Unsupported histogram_equalization mode: {mode}')
    sz = int(d['image_size'])
    arr = cv2.resize(arr, (sz,sz), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr,0,1).astype(np.float32)


class CTDataset(Dataset):
    def __init__(self, manifest, split, config, augment=None):
        self.frame = manifest[manifest['split']==split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f'No rows for {split}')
        self.config = config
        self.augment = augment
        self.root = Path(config.get('image_root', '.')).resolve()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        record = self.frame.iloc[index]
        p = Path(str(record['image_path']))
        if not p.is_absolute():
            p = self.root / p
        arr = load_image(p, self.config, record.get('roi_xyxy'))
        if self.augment is not None:
            arr = self.augment(arr)
        channels = int(self.config['data']['channels'])
        if channels not in (1,3):
            raise ValueError('Supported channels are 1 or 3')
        img = torch.from_numpy(np.repeat(arr[None],channels,axis=0).copy()).float()
        return img, int(record['label']), str(record['patient_id']), str(record['image_path'])


def make_loader(dataset, batch_size, shuffle=False, num_workers=0, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                      sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
