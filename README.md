# ICS-RL: Intelligent Chaotic Steganography with Reinforcement Learning

## Description

ICS-RL is a hybrid spatial-domain image steganography framework that combines **logistic chaotic map-based candidate generation** with **deep reinforcement learning (RL)** for secure and content-aware pixel selection.

The system integrates:
- A **Logistic Chaotic Map** to generate a pseudo-random candidate pixel set (N = 15,000 per image), providing cryptographic unpredictability.
- A **Texture Feature Extractor** computing local variance, Sobel gradient, and Laplacian response over 7×7 neighbourhoods.
- A **PPO Policy Network** (Proximal Policy Optimisation) that selects the optimal embedding subset within the chaotic candidate space.
- A **Frozen SRNet Reward Environment** providing a steganalysis-based detection penalty, SSIM distortion penalty, and texture bonus.

This repository accompanies the paper:

> Çalçoban MN, Kasapbaşı MC. *Intelligent chaotic steganography: reinforcement learning-based optimal pixel selection for secure spatial-domain image steganography.* PeerJ Computer Science (under review).

---

## Repository Structure

```
ics_rl_project/
├── src/
│   ├── envs/                     # SRNet reward environment
│   │   ├── steganography_env.py
│   │   └── stego_env.py
│   ├── models/                   # SRNet model definition
│   │   └── srnet.py
│   ├── modules/                  # Core modules
│   │   ├── chaotic_generator.py
│   │   ├── policy_network.py
│   │   └── texture_extractor.py
│   ├── training/                 # PPO training loop
│   │   └── ppo_trainer.py
│   └── __init__.py
├── utils/
│   ├── bossbase_loader.py        # BOSSBase dataset loader
│   ├── message_codec.py          # Payload encoding/decoding
│   └── __init__.py
├── baselines.py                  # Baseline implementations (Chaotic LSB, HILL, S-UNIWARD)
├── evaluate.py                   # Main evaluation script
├── evaluate_seeded.py            # Seeded evaluation on n=150 images
├── evaluate_seeded_n20.py        # Seeded evaluation on n=20 subset
├── evaluate_paper_figures.py     # Figure generation for paper
├── train.py                      # Main ICS-RL training script
├── train_srnet.py                # SRNet steganalyser training script
└── README.md
```

> **Note:** `data/`, `checkpoints/`, and `results/` directories are excluded from this repository via `.gitignore`. See Dataset and Usage sections below.

---

## Dataset Information

All experiments use the **BOSSBase 1.01** benchmark dataset.

- **Source:** [https://dde.binghamton.edu/download/](https://dde.binghamton.edu/download/)
- **Contents:** 10,000 greyscale images at 512×512 pixels
- **Usage in this work:** Images are centre-cropped and downsampled to 256×256 pixels prior to use.
- **Licence:** BOSSBase is freely available for academic research. Please refer to the licence terms at the download page above.

> ⚠️ The dataset is not included in this repository. Please download directly from the official Binghamton University link above.

---

## Requirements

Python 3.9+ is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
opencv-python>=4.8.0
Pillow>=9.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

CUDA 11.8 or later is recommended for GPU-accelerated training.

---

## Usage Instructions

### 1. Prepare the Dataset

Download BOSSBase 1.01 from the official DDE Lab, Binghamton University:

```
http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
```

Extract to `data/bossbase/`.

### 2. Train the SRNet Steganalyser

```bash
python train_srnet.py \
  --data_dir data/bossbase/ \
  --max_images 1000 \
  --epochs 6 \
  --output_dir checkpoints/srnet/
```

This uses cover/stego index splitting to avoid identity memorisation (see Section 4 of the paper). Training stops automatically when validation accuracy reaches ≥ 0.72.

### 3. Train the ICS-RL Policy

```bash
python train.py \
  --data_dir data/bossbase/ \
  --srnet_checkpoint checkpoints/srnet/best.pth \
  --num_train_images 50 \
  --total_steps 30000 \
  --chaotic_key 0.123456 \
  --payload_bpp 0.4 \
  --output_dir checkpoints/icsrl/
```

### 4. Evaluate (n=150, seed=42)

```bash
python evaluate_seeded.py \
  --data_dir data/bossbase/ \
  --policy_checkpoint checkpoints/icsrl/best.pth \
  --srnet_checkpoint checkpoints/srnet/best.pth \
  --n_eval 150 \
  --seed 42 \
  --payload_bpp 0.04 \
  --texture_percentile 60 \
  --output results/
```

### 5. Evaluate Baselines

```bash
python baselines.py \
  --data_dir data/bossbase/ \
  --n_eval 150 \
  --seed 42 \
  --payload_bpp 0.04 \
  --output results/
```

### 6. Reproduce Paper Figures

```bash
python evaluate_paper_figures.py \
  --data_dir data/bossbase/ \
  --policy_checkpoint checkpoints/icsrl/best.pth \
  --srnet_checkpoint checkpoints/srnet/best.pth \
  --output results/figures/
```

---

## Methodology

The ICS-RL pipeline (during inference) proceeds as follows:

1. **Chaotic candidate generation:** Given secret key x₀, iterate the logistic map (`x_{n+1} = r · x_n · (1 − x_n)`, r = 3.99) over all 65,536 pixel positions; select the top N = 15,000 as candidate set C.
2. **Texture feature extraction:** Compute local variance, Sobel gradient magnitude, and Laplacian response over 7×7 patches; normalise and average to produce texture saliency map T.
3. **Percentile filtering (inference only):** Retain only candidates in C whose texture score exceeds the 60th percentile of the image saliency distribution.
4. **Policy selection:** Feed the three-channel tensor [cover image, chaotic mask, texture map] to the PPO policy network; obtain binary embedding mask S ⊆ C.
5. **LSB embedding:** Apply 1-bit LSB substitution at all positions in S (raster-scan order) to produce stego image I′.
6. **Extraction:** Regenerate C from x₀, recompute T, reapply filtering, rerun the policy to recover S, and extract LSBs in raster-scan order.

---

## Reward Function

The PPO agent is trained with a three-component reward plus a capacity penalty:

**R = w₁ · r_detect + w₂ · r_distort + w₃ · r_texture − λ_cap · r_capacity**

| Component | Weight | Formula |
|---|---|---|
| Detection penalty | w₁ = 0.50 | `1 − 2·\|p_detect − 0.5\|` |
| Distortion penalty | w₂ = 0.35 | `SSIM − 1` |
| Texture bonus | w₃ = 0.15 | `tanh(2 · τ_mean)` |
| Capacity penalty | λ_cap = 1.0 | Active when selected pixels < payload requirement |

An exponential moving average (EMA, α = 0.70) is applied to p_detect to smooth reward variance.

---

## Trained Model Checkpoints

Trained model checkpoints (SRNet steganalyser and ICS-RL policy) are not included in this repository. They are available from the corresponding author upon reasonable request.

**Contact:** mustafacemkasapbasi@beykoz.edu.tr

---

## Citations

If you use this code, please cite:

```bibtex
@article{calcobankasapbasi2025icsrl,
  author    = {Çalçoban, Merve Nur and Kasapbaşı, Mustafa Cem},
  title     = {Intelligent chaotic steganography: reinforcement learning-based optimal pixel selection for secure spatial-domain image steganography},
  journal   = {PeerJ Computer Science},
  year      = {2025},
  note      = {Under review}
}
```

For the BOSSBase dataset:

```bibtex
@inproceedings{bas2011boss,
  author    = {Bas, Patrick and Filler, Tomáš and Pevný, Tomáš},
  title     = {"Break Our Steganographic System": The Ins and Outs of Organizing BOSS},
  booktitle = {Information Hiding},
  year      = {2011},
  doi       = {10.1007/978-3-642-24178-9_5}
}
```

For the SPAR-RL framework that conceptually informed this work:

```bibtex
@article{tang2021sparrl,
  author    = {Tang, Weixuan and Li, Bin and Barni, Mauro and Li, Jin and Huang, Jiwu},
  title     = {An Automatic Cost Learning Framework for Image Steganography Using Deep Reinforcement Learning},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2021},
  doi       = {10.1109/TIFS.2020.3025438}
}
```

---

## Code Notice

ICS-RL is an independently implemented framework. The policy network architecture and PPO training design are conceptually informed by SPAR-RL (Tang et al., 2021, IEEE T-IFS), but the ICS-RL codebase is a from-scratch implementation and does not include or redistribute any SPAR-RL source code.

---

## Licence

This code is released for academic and research purposes only. Commercial use is not permitted without explicit written consent from the authors.

**Dual-use notice:** Steganographic tools have dual-use potential. This repository is provided solely for academic research. Users are responsible for ensuring compliance with applicable laws and institutional ethical guidelines.

Contributions (bug reports, improvements, extensions) are welcome via pull requests. Please open an issue first to discuss major changes.