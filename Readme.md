# [Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes]

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.12982)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/AbyssLumine/GAPL/tree/main)
## 💡 Motivation

![Motivation Framework](assets/overview.jpg)
<p align="center"><em>Figure 1: Overview of our proposed Generator-Aware Prototype Learning (GAPL) framework.</em></p>

In scaling up AIGI detections, generated images show heterogenity and cause previous AIGI detector fails to scale.

We learn a small set of forgery concepts as Generator-Aware Prototypes. And convert diverse generators into some certain prototypes.

## 🛠️ Preparation
### 1. Environment Setup
We provide the minimum requirement package in `requirements.txt`, you could check whether your environment satisfy it or create an enviroment with the following command.

```bash
pip install -r requirements.txt
```

## 🚀 Quick inference
To evaluate performance of the proposed GAPL, You need to download the checkpoint from 
* 🔗 **Pretrained**: [Huggingface](https://huggingface.co/AbyssLumine/GAPL) 

### 1. Evaluate the proposed model in benchmarks.
To reproduce the results reported in our paper across various benchmarks:

1.  Modify the dataset paths in `benchmarks.py` to point to your local data.
2.  Run the evaluation script:

<!-- end list -->

```bash
bash scripts/val_bench.sh
```
 

### 2\. Single Image Inference

You can also run inference on a single image to detect whether it is **Real** or **Fake**.

```bash
python inference.py \
  --model_path pretrained/checkpoint.pt \
  --image_path assets/test_image.jpg \
  --device cuda
```

**Output Example:**

```text
[INFO] Loading model from pretrained/checkpoint.pt...
[RESULT] Image: assets/test_image.jpg
  -> Prediction: Fake (AI-Generated)
  -> Confidence: 99.8%
```

## 🏋️ Training a GAPL model

### 📦 Prerequisites

Before starting, please ensure you have prepared the required datasets:

* **Stage 1 Data**: 
    * [GenImage](https://github.com/GenImage-Dataset/GenImage) or [CNNSpot](https://github.com/PeterWang512/CNNDetection) Training Set.
    * CLIP Pre-trained Model (will be automatically downloaded via HuggingFace).
* **Stage 2 Data**: 
    * **Community Forensics (Small Training Set)**: Please download it from Hugging Face.
    * 🔗 **Download Link**: [OwensLab/CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) 

---

### Stage 1: Backbone Training & Prototype Extraction

In this stage, we train the backbone and learn the initial generator-aware prototypes.

**Step 1: Configure Paths**
Please open `prototype_dataset.py` and modify the dataset paths to match your local environment.

**Step 2: Train Backbone**
Run the following script to start training:
```bash
bash scripts/stage1.sh
```
**Step 3: Extract Prototypes**
 After the backbone training converges, run the extraction script to generate the prototype vectors:

```
prototype/dream_prototype.py
```
⚡ Fast Track: We provide the pre-trained Stage 1 checkpoint and pre-extracted prototype vectors. You can skip this stage by downloading them from [pretrained](https://huggingface.co/AbyssLumine/GAPL).

### Stage 2: Fine-tuning
In the second stage, we fine-tune the model using the Community Forensics dataset to enhance robustness against diverse generators.

Run Training:

```
scripts/stage2.sh
```

## 


## 🖊️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{qin2025Scaling,
  title={Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes},
  author={Qin, Ziheng and Ji, Yuheng and Tao, Renshuai and Tian, Yuxuan and Liu, Yuyang and Wang, Yipu and Zheng, Xiaolong},
  journal={arXiv preprint arXiv:2512.12982},
  year={2025}
}
```

## 🙏 Acknowledgements

Our code is developed based on the following excellent open-source repositories. We appreciate their excellent work and contributions to the community:

**[CNNDetection](https://github.com/peterwang512/CNNDetection)**

**[Community Forensics](https://github.com/JeongsooP/Community-Forensics)** We leverage the dataset and borrow some code from this codebase.