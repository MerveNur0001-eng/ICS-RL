# ICS-RL: Intelligent Chaotic Steganography using Reinforcement Learning

This project implements **Intelligent Chaotic Steganography (ICS-RL)**, a hybrid framework that integrates deep reinforcement learning (RL) with logistic chaotic map theory for optimal, content-aware pixel selection in spatial-domain image steganography.

The system is based on the **SPAR-RL** framework (Tang et al., 2021) and extends it with chaotic candidate generation and a texture-aware reward function.

## Project Structure

```text
ics_rl_project/
├── data/
│   └── bossbase/          # BOSSBase 1.01 dataset (10,000 images)
├── models/                # Saved model checkpoints (SRNet, Policy Network)
├── src/
│   ├── modules/
│   │   ├── chaotic_generator.py  # Logistic Map (x_{n+1} = r * x_n * (1 - x_n))
│   │   ├── texture_extractor.py  # Local Variance, Sobel, Shannon Entropy
│   │   ├── policy_network.py     # 6-layer CNN Policy Network (PPO)
│   │   └── steganography_env.py  # Frozen SRNet & Reward Function (w1, w2, w3)
│   └── __init__.py
├── utils/
│   └── __init__.py
├── results/               # Evaluation metrics (PSNR, SSIM, P_o)
├── main.py                # Sample embedding/extraction pipeline
└── README.md              # Project documentation
```

## Key Features

1.  **Chaotic Candidate Generator**: Uses a logistic map with a secret key $x_0$ to generate $N=30,000$ candidate pixels, providing strong unpredictability.
2.  **Texture Feature Extractor**: Computes a saliency map based on local variance, Sobel gradients, and Shannon entropy to identify texture-rich regions.
3.  **Deep RL Policy Network**: A PPO-based agent that selects the final embedding positions from the chaotic candidates, guided by image content.
4.  **Texture-Aware Reward Function**: Combines detection penalty (SRNet), distortion penalty (MSE), and texture bonus to optimize for both security and imperceptibility.

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.x
- OpenCV, NumPy, SciPy

### Installation
```bash
pip install torch torchvision opencv-python numpy scipy
```

### Running the Sample Pipeline
To verify the embedding and extraction synchronization:
```bash
python main.py
```

## References
- [1] Tang, W., Li, B., Barni, M., Li, J., & Huang, J. (2021). An automatic cost learning framework for image steganography using deep reinforcement learning. IEEE TIFS.
- [2] Bas, P., Filler, T., & Pevny, T. (2011). Break Our Steganographic System: The BOSSbase 1.01.
- [3] Boroumand, M., Chen, M., & Fridrich, J. (2019). Deep residual network for steganalysis of digital images. IEEE TIFS.
