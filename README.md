# Image Denoising with ResNet

Grayscale image denoising on the [LFW (Labelled Faces in the Wild)](https://www.tensorflow.org/datasets/catalog/lfw) dataset using residual convolutional networks in TensorFlow/Keras. Two notebooks are provided — a baseline and an enhanced version with channel attention, blind denoising, and a perceptual loss function.

[![Open Baseline In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdebanjiAdelowo/Image_denoising_using_ResNet/blob/main/final_image_denoising.ipynb)
&nbsp;
[![Open Enhanced In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdebanjiAdelowo/Image_denoising_using_ResNet/blob/main/resnet_enhanced_denoiser.ipynb)

---

## Notebooks

### `final_image_denoising.ipynb` — Baseline ResNet
Straightforward residual network trained end-to-end on full 250×250 images with fixed Gaussian noise (σ = 0.09).

### `resnet_enhanced_denoiser.ipynb` — Enhanced ResNet with Channel Attention
Significant architectural and training improvements designed for higher PSNR/SSIM.

---

## Architecture Comparison

| Feature | Baseline | Enhanced |
|---|---|---|
| Residual blocks | 5 plain res blocks | 8 RCAB blocks |
| Channel attention | ✗ | ✓ Squeeze-and-Excitation |
| Training data | 3 000 full images | 80 000 random 64×64 patches |
| Augmentation | None | Random horizontal + vertical flip |
| Noise during training | Fixed σ = 0.09 | Random σ ∈ [0.05, 0.15] (blind) |
| Loss function | MSE | 0.8 × (1−SSIM) + 0.2 × L1 |
| LR schedule | Fixed 1e-3 | Cosine decay 1e-3 → 1e-6 |
| Batch size | 32 | 64 |

### Residual Channel Attention Block (RCAB)

```
x ──► Conv(3×3)─BN─ReLU─Conv(3×3)─BN ──► Squeeze-and-Excitation ──► Add(x) ──► ReLU
                                                      │
                                          GlobalAvgPool → FC(C/8)→ReLU→FC(C)→Sigmoid
                                          (channel-wise attention weights)
```

The **Squeeze-and-Excitation** sub-block learns to re-weight each feature channel by its global importance, allowing the network to focus on the most informative features for denoising.

---

## Key Design Decisions

**Patch-based training.** Extracting 16 random 64×64 patches per image gives 80 000 training samples versus 3 000 full images, at a fraction of the memory cost. The model is fully convolutional and infers on any image size at test time.

**Blind denoising (random σ).** Training with σ drawn uniformly from [0.05, 0.15] instead of a fixed value makes the model robust across noise levels without retraining.

**Combined SSIM + L1 loss.** MSE optimises pixel-level fidelity but can produce blurry results. SSIM captures luminance, contrast, and structural similarity, giving sharper, more visually pleasing outputs.

**Cosine LR decay.** The learning rate follows a cosine curve from 1e-3 to 1e-6, providing large steps early in training and fine-grained updates near convergence.

---

## Results

Evaluated on 1 000 held-out LFW test images (250×250, σ = 0.09 Gaussian noise).

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Noisy input | ~20.9 | ~0.72 |
| Baseline ResNet (5 blocks, MSE) | ~24–25 | ~0.86 |
| Enhanced ResNet (8 RCAB, SSIM+L1) | ~26–28 | ~0.91 |

*Exact numbers depend on random seed and early stopping epoch; run the notebooks to reproduce.*

---

## Setup

**Requirements** (installed automatically on Colab with GPU runtime):

```
tensorflow >= 2.10
tensorflow-datasets
numpy
scikit-image
matplotlib
```

**Local install:**

```bash
pip install tensorflow tensorflow-datasets scikit-image matplotlib
```

---

## How to Run

### Colab (recommended — free GPU)

1. Click the **Open in Colab** badge above for the notebook you want.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (`Runtime → Run all`).

### Local

```bash
git clone https://github.com/AdebanjiAdelowo/Image_denoising_using_ResNet
cd Image_denoising_using_ResNet
pip install tensorflow tensorflow-datasets scikit-image matplotlib
jupyter notebook
```

Open either notebook and run all cells. GPU is strongly recommended; CPU training on 80 000 patches will be slow.

---

## Repository Layout

```
Image_denoising_using_ResNet/
├── final_image_denoising.ipynb        # Baseline ResNet (5 blocks, MSE loss)
├── resnet_enhanced_denoiser.ipynb     # Enhanced ResNet (8 RCAB, SSIM+L1 loss)
└── README.md
```

Saved models (`best_denoiser.keras`, `enhanced_denoiser_resnet.keras`) and output figures are written to the Colab working directory during training and are not committed to the repository.

---

## References

- He et al. (2016) — *Deep Residual Learning for Image Recognition*, CVPR
- Zhang et al. (2017) — *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*, TIP
- Hu et al. (2018) — *Squeeze-and-Excitation Networks*, CVPR
- Zhang et al. (2018) — *Image Super-Resolution Using Very Deep Residual Channel Attention Networks (RCAN)*, ECCV
- LFW Dataset — [tensorflow.org/datasets/catalog/lfw](https://www.tensorflow.org/datasets/catalog/lfw)

---

## License

MIT
