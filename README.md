# Discriminative Self-Supervised Pre-Training for Esophagitis Detection in Upper GI Endoscopy Images

[ :scroll: [`Paper`](https://proceedings.mlr.press/v315/friedetzki26a.html)] [ :book: [`BibTeX`](#citation)]

This repository contains the code for esophagitis detection in upper gastrointestinal endoscopy images using discriminative self-supervised learning. We demonstrate that domain-specific pre-training using DINOv3 on unlabeled upper GI endoscopy images significantly improves esophagitis detection performance compared to supervised ImageNet pre-training.

<br/>

<div align="center">
  <img src="https://github.com/user-attachments/assets/04f7f99f-b7ab-42ac-94a8-c1dbea4e2849" height=450 alt="pretraining_finetuning_overview">
  
  <i>Overview of the training pipeline employed in this study</i>
</div>

<br/>

## DINOv3 Self-Supervised Pre-Training

### Training

For self-supervised pre-training, please use [the official DINOv3 code](https://github.com/facebookresearch/dinov3). The specific training configurations and modifications to the data augmentations are available in the `dinov3` directory of this repository.

### Pre-trained Models

ViT models pre-trained on upper gastrointestinal endoscopy images (UpperGI-400K). These models are vision backbones that can be plugged to other models for endoscopy downstream tasks.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16</td>
      <td align="right">21M</td>
      <td align="center">UpperGI-400K</td>
      <td align="center"><a href="https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vits16-pretrain-upperGI400k.pth">link</a></td>
    </tr>
    <tr>
      <td>ViT-B/16</td>
      <td align="right">86M</td>
      <td align="center">UpperGI-400K</td>
      <td align="center"><a href="https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vitb16-pretrain-upperGI400k.pth">link</a></td>
    </tr>
    <tr>
      <td>ViT-L/16</td>
      <td align="right">300M</td>
      <td align="center">UpperGI-400K</td>
      <td align="center"><a href="https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vitl16-pretrain-upperGI400k.pth">link</a></td>
    </tr>
  </tbody>
</table>

<br/>

If you want to use one of the pre-trained models for feature extraction, take a look at the `UpperGIFeatureExtractor` class in `feature_extractor.py`.

```python
import torch

from feature_extractor import UpperGIFeatureExtractor

model = UpperGIFeatureExtractor(size="base")

img = torch.randn(1, 3, 224, 224)
features = model(img)
# features is a (1, num_features) shaped tensor
```

## Supervised Fine-tuning for Esophagitis Detection

Fine-tune pre-trained models on the labeled esophagitis dataset.

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

This work uses a binary classification task (esophagitis vs. non-esophagitis) from upper GI endoscopy images.


#### 1. Download the public source datasets

The class aggregation and train/val/test splits can be reproduced using the `train_val.csv` and `test.csv` files located in the `esodetector/data` directory.

- HyperKvasir ([link](https://datasets.simula.no/hyper-kvasir/))
- ERS ([link](https://cvlab.eti.pg.gda.pl/en/publications/endoscopy-dataset))
- GastroVision ([link](https://osf.io/84e7f/))

#### 2. Organize your data in the following structure

```
data/
├── train/
│   ├── esophagitis/
│   └── normal/
│   └── barretts/
│   └── ...
├── val/
│   ├── esophagitis/
│   └── normal/
│   └── barretts/
└── test/
    ├── esophagitis/
    └── normal/
    └── barretts/
```

#### 3. Configure paths using environment variables

Create a `.env` file in the root directory:

```bash
DATA_DIR=/path/to/your/data
CLASS_MAP=/path/to/class_mapping.json
```

#### 4. Class Mapping

The class mapping file ([esodetector/data/class_mapping.json](esodetector/data/class_mapping.json)) defines the binary classification:
- Esophagitis: class 1
- All other conditions (normal, Barrett's, cancer, polyp, ulcer, varices): class 0

### Training

Place pre-trained model checkpoints in `dinov3/pretrained/`

```bash
# With domain-specific self-supervised pre-training
./distributed_train.sh 2 --config esodetector/configs/train/vit_large_dinov3_upperGI400k.yaml

# With ImageNet pre-training (baseline)
./distributed_train.sh 2 --config esodetector/configs/train/vit_large_imagnet1k.yaml
```


### Evaluation

Evaluate a trained model on the test set:

```bash
python validate.py --config esodetector/configs/val/vit_large_dinov3_upperGI400k.yaml
```

### Results

| Model | Pre-training | AUPRC |  AUROC | F1 | Download |
|-------|--------------|-----------|-----------|--------------| -----|
| ViT-S/16 | Supervised | 83.23±1.26 | 94.46±1.46 | 75.57±2.40 | - |
| ViT-B/16 | Supervised | 82.40±4.15 | 94.51±2.48 | 75.64±2.92 | - |
| ViT-L/16 | Supervised | 84.31±2.06 | 94.98±1.06 | 76.74±1.34 | - |
| ViT-S/16 | DINOv3 | 86.16±1.53 | 96.64±0.34 | 77.47±1.40 | [link](https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vits16-esophagitis-detector.pth) |
| ViT-B/16 | DINOv3 | 88.88±0.92 | 97.04±0.57 | 80.15±0.93 | [link](https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vitb16-esophagitis-detector.pth) |
| ViT-L/16 | DINOv3 | **89.82**±0.68 | **97.17**±0.34 | **81.44**±0.74 | [link](https://huggingface.co/tofriede/dinov3-upperGI/blob/main/dinov3-vitl16-esophagitis-detector.pth) |


## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{friedetzki2026discriminative,
  title={Discriminative Self-Supervised Pre-Training for Esophagitis Detection in Upper GI Endoscopy Images},
  author={Friedetzki, Tobias and Chandraiah, Naveen and Svoboda, Emil and Pecina, Pavel and Puppe, Frank and Krenzer, Adrian},
  booktitle={Proceedings of The 9th International Conference on Medical Imaging with Deep Learning},
  pages={1137--1152},
  year={2026},
  volume={315},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v315/friedetzki26a.html},
}

```
