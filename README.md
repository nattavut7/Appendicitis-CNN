# Appendicitis Detection using Lightweight CNN

This repository provides the implementation of an ultra-lightweight CNN framework for detecting acute appendicitis from abdominal CT images.

## Model Performance
- Accuracy: 99.7%
- AUC: 99.2%
- Dice: 99.1%
- Parameters: ~0.58M

## Installation
```bash
pip install -r requirements.txt
```

## Run Inference
```bash
python main.py --mode infer --image data/sample_ct_images/example.png
```

## Dataset
Data used in this study is available via institutional collaboration. Sample CT slices are included for demonstration.

## License
MIT
