# Generative AI for Synthetic Multimodal Sensor Data in Situation Awareness Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Paper: ICMI 2026](https://img.shields.io/badge/Paper-ICMI'26-blue.svg)](https://icmi.acm.org/)
[![Dataset: UCI HAR](https://img.shields.io/badge/Dataset-UCI%20HAR-green.svg)](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

---

## 📋 Overview

A persistent barrier to developing robust Situation Awareness (SA) systems is the scarcity of labelled, high-quality multimodal sensor data. Real-world data is expensive to collect, privacy-sensitive, and critically under-represents rare events — precisely the events SA systems must handle robustly.

This repository presents a **systematic evaluation of four generative architectures** for synthetic multimodal sensor data generation using the UCI Human Activity Recognition (HAR) benchmark — a multimodal dataset combining accelerometer and gyroscope IMU signals from wrist-worn wearables.

### Key Contributions

1. Systematic comparison of **GAN, VAE, CGAN, and CVA-GAN** for multimodal SA data synthesis
2. Empirical evaluation using **PCA, t-SNE, MMD**, and downstream classification metrics
3. **Adversarial debiasing** strategy for fairness in synthetic data generation
4. Demonstration that **augmenting real data with synthetic data improves SA classifier performance** while preserving participant privacy

---

## 📊 Key Results

### Distributional Alignment

| Model | MMD Score ↓ | Notes |
|---|---|---|
| **GAN** | **0.0020** | Best statistical alignment |
| VAE | 0.6430 | Untrained prior sampling |

### Downstream Classification (Random Forest, 100 estimators)

| Training Data | Test Accuracy | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| Real only | 91.92% | 0.9269 | 0.9257 | 0.9260 | 0.9949 |
| **Augmented (Real + GAN)** | **92.15%** | **0.9330** | **0.9315** | **0.9322** | **0.9957** |
| VAE synthetic only | 64.54% | — | — | — | — |
| CGAN synthetic only | 38.79% | — | — | — | — |
| CVA-GAN synthetic only | 17.85% | 0.1813 | 0.1785 | 0.0938 | — |
| GAN synthetic only | 20.43% | 0.1342 | 0.1252 | 0.1047 | — |

### Bias Mitigation

| Metric | Before Training | After 400 Epochs |
|---|---|---|
| Fairness Loss | 0.6779 | **0.0074** |
| Bias Score | — | **0.027** |

---

## 🏗️ Framework Overview

```
UCI HAR Dataset (10,299 samples × 561 features)
Accelerometer + Gyroscope IMU signals
        │
        ▼
  Z-score Normalisation
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   GAN / CGAN / VAE / CVA-GAN         Isolation Forest
   Synthetic Data Generation          Outlier retention
        │                             (736 samples kept)
        ▼
  Evaluation
  ├── Distributional: PCA, t-SNE, MMD
  ├── Downstream: Random Forest classification
  ├── Rare event synthesis (CGAN, class 4)
  └── Fairness: Adversarial debiasing (bias score)
```

---

## 🗂️ Dataset

**[UCI Human Activity Recognition (HAR)](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)**

- 10,299 samples across 6 activities: walking, walking upstairs, walking downstairs, sitting, standing, laying
- 30 participants, waist-mounted smartphone
- 561-dimensional feature vectors from accelerometer and gyroscope signals
- Publicly available, ethically sourced

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Farjana-Yesmin/Generative-AI-for-Synthetic-Multimodal-Sensor-Data-in-Situation-Awareness-Training.git
cd Generative-AI-for-Synthetic-Multimodal-Sensor-Data-in-Situation-Awareness-Training

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Run the Notebook

```bash
jupyter notebook "Generative_AI_for_Synthetic_Multimodal_Sensor_Data_in_Situation_Awareness_Training.ipynb"
```

Download the UCI HAR dataset from the [official UCI repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) and place the extracted folder as `UCI HAR Dataset/` in the project directory.

---

## ⚙️ Model Architectures

### GAN
- Generator: Dense(256 → 512 → 1024 → 561), LeakyReLU + BatchNorm, tanh output
- Discriminator: 1,231,873 parameters
- Optimiser: Adam (lr=0.0002, β₁=0.5), batch size 16, 500 epochs
- Memory: 16.8% GPU utilisation on Google Colab

### VAE
- Encoder: 561-dim → latent space (dim=20) via mean + log-variance
- Reparameterisation trick for backpropagation
- Loss: reconstruction + KL divergence, 50 epochs (final loss: 24,594.96)

### Conditional GAN (CGAN)
- Conditions generator and discriminator on activity labels via embedding layers
- Enables class-specific synthesis and rare event augmentation
- Discriminator loss stabilises at 0.30–0.40 after 500 epochs

### CVA-GAN (Hybrid)
- Combines VAE encoder-decoder structure with GAN adversarial training
- Tracks VAE loss, discriminator loss (4.37), and generator loss (0.008) simultaneously

---

## 🎯 Rare Event Synthesis

The CGAN enables targeted generation of underrepresented SA-critical events:

```python
rare_labels = np.full((1000,), 4)  # SA-relevant rare class (sitting/transition)
noise = np.random.normal(0, 1, (1000, latent_dim))
synth_rare = c_generator.predict([noise, rare_labels])
# 1000 synthetic rare-event samples generated
```

---

## ⚖️ Bias Mitigation via Adversarial Debiasing

A fairness-aware discriminator uses subject IDs as a proxy for protected attributes, with a dual-output head (validity + fairness). Fairness loss decreased from **0.6779 → 0.0074** over 400 epochs, confirming progressive demographic bias reduction in generated samples.

---

## 🔍 Why GAN-Only Training Underperforms

The 71-percentage-point gap between GAN-only and real-data training has three causes:

1. **Mode collapse** — discriminator accuracy collapses to below 2% by epoch 100; the generator finds a narrow output set that consistently fools the discriminator
2. **Unconditional blending** — without class conditioning, the GAN produces a single blended distribution across all 6 activity classes
3. **High dimensionality** — the 561-dimensional HAR feature space is large relative to the training set, making stable GAN convergence harder

Future work: WGAN-GP, conditional architectures across all comparisons, and feature-matching losses.

---

## 📁 Repository Structure

```
├── Generative_AI_for_Synthetic_Multimodal_Sensor_Data_in_Situation_Awareness_Training.ipynb
├── t-SNE of Real vs Synthetic Data.png
├── PCA of Real vs Synthetic Data.png
├── Plot Loss.png
├── Feature Distribution Comparison (Real vs. Synthetic).png
├── Multi-Class ROC Curve.png
├── Confusion Matrix.png
└── Real Data Distribution.png
```

---

## 🔮 Future Work

- Replace vanilla GAN with WGAN-GP to address mode collapse
- Extend conditional generation to all four architectures
- Apply diffusion model approaches for structured sensor data
- Extend to truly multimodal data (combined IMU, audio, and video)
- Transfer learning between synthetic and real multimodal SA domains
- Domain-validated SA evaluation pipelines using Endsley's measurement framework

---

## 🛡️ Responsible Innovation Statement

Synthetic data supplements rather than replaces real human-derived sensor data, preserving privacy by enabling model training without raw data sharing. The fairness-aware discriminator explicitly mitigates demographic bias. Limitations of synthetic-only training (20.43% accuracy) are transparently reported. All data used (UCI HAR) is publicly available and ethically sourced.

---

## 📝 Citation

```bibtex
@inproceedings{anonymous2026generativesa,
  title     = {Generative AI for Synthetic Multimodal Sensor Data in Situation Awareness Training},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the ACM International Conference on Multimodal Interaction (ICMI '26)},
  year      = {2026},
  address   = {Napoli, Italy},
  publisher = {ACM}
}
```

---

## 👥 Contributors

- **Farjana Yesmin** — farjanayesmin76@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
