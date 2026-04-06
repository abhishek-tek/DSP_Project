# EchoFind
### Self-Supervised Audio Representation & Retrieval Engine
*IEEE Impulse 2026 — Signal Processing & Deep Learning Track*

---

## Overview

EchoFind is a self-supervised learning (SSL) pipeline for music audio. It learns compact, semantically rich vector representations of audio signals **without using any genre labels during pre-training**, leveraging only the intrinsic structure of the audio waveform itself.

The system is built in five phases:

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Input Pipeline — preprocessing, augmentation, feature visualisation | ✅ Complete |
| 2 | Representation Learning — SSL encoder training (SimCLR / BYOL / Barlow Twins) | 🔄 In Progress |
| 3 | Robust Retrieval — vector search engine ("Shazam Test") | ⏳ Planned |
| 4 | Semantic Utility Verification — linear probe + t-SNE clustering | ⏳ Planned |
| 5 | Grandmaster Extensions — latent interpolation, OOD detection, variable-length inference | ⏳ Planned |

This repository contains the **Phase 1 deliverable**: a desktop GUI application and Jupyter notebook implementing the full audio preprocessing and augmentation pipeline on the [FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset.

---

## Architecture

```
Raw Audio (.mp3 / .wav / .flac)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Phase 1 — Input Pipeline                         │
│                                                   │
│  librosa.load() → resample to 22,050 Hz (mono)   │
│        │                                          │
│        ▼                                          │
│  STFT (n_fft=2048, hop=512, Hann window)          │
│        │                                          │
│        ▼                                          │
│  Mel Filterbank (128 bands, 0–8000 Hz)            │
│  m = 2595 · log₁₀(1 + f / 700)                   │
│        │                                          │
│        ▼                                          │
│  Log-Amplitude (power_to_dB)                      │
│        │                                          │
│        ▼                                          │
│  Normalise → [0, 1]  →  Tensor [1, 128, T]        │
│        │                                          │
│        ▼                                          │
│  AugmentationPipeline                             │
│  ├── Time Stretch   (rate ∈ [0.85, 1.15])         │
│  ├── Pitch Shift    (n   ∈ [−3, +3] semitones)    │
│  ├── Gaussian Noise (σ   ∈ [0.005, 0.02])         │
│  ├── Frequency Mask (f   ∈ [0, 30] Mel bands)     │
│  └── Time Mask      (t   ∈ [0, 80] frames)        │
│        │                                          │
│        ▼                                          │
│  (view_i, view_j) — SimCLR positive pair          │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 2 — SSL Encoder      │   (ResNet-18 / ViT)
│  Loss: NT-Xent / Barlow Twins│
│  Output: h ∈ ℝ⁵¹²           │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Phase 3 — FAISS Index      │   Cosine similarity retrieval
│  query() → Track ID         │
└─────────────────────────────┘
```

---

## Repository Structure

```
echofind-impulse2026/
├── echofind_gui.py              # Phase 1: Desktop GUI application (tkinter + matplotlib)
├── requirements.txt             # Python dependencies
├── README.md
│
├── notebooks/
│   └── phase1_echofind.ipynb   # Interactive pipeline walkthrough + eval checks
│
├── submission.py                # Phase 2+: AudioEncoder, get_embedding(), predict_track()
│
└── weights/
    └── encoder.pth              # Phase 2 trained weights (added post-training)
```

---

## Phase 1: Desktop Application

### Installation

```bash
git clone https://github.com/your-username/echofind-impulse2026.git
cd echofind-impulse2026
pip install -r requirements.txt
```

### Running

```bash
python echofind_gui.py
```

No server, no browser. A native desktop window opens immediately.

### Dependencies

```
librosa>=0.10.0      # Audio I/O, spectral features, beat tracking, HPSS
matplotlib>=3.7.0    # Rendering all visualisations
numpy>=1.24.0        # Array operations
soundfile>=0.12.0    # Audio file backend for librosa
tkinter              # GUI framework (stdlib — no install required)
```

---

## Feature Visualisations (9 Tabs)

| Tab | Feature | Key Function |
|-----|---------|-------------|
| ① | Log-Mel Spectrogram | `librosa.feature.melspectrogram` → `power_to_db` |
| ② | STFT Log-Power | `librosa.stft` → `amplitude_to_db` |
| ③ | Waveform | `librosa.display.waveshow` |
| ④ | Chromagram | `librosa.feature.chroma_cqt` |
| ⑤ | MFCC | `librosa.feature.mfcc` (n=20) |
| ⑥ | Spectral Features | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff` |
| ⑦ | Onset Strength + Beats | `librosa.onset.onset_strength`, `librosa.beat.beat_track` |
| ⑧ | HPSS | `librosa.effects.hpss` (Fitzgerald 2010) |
| ⑨ | SSL Augmented Views | `AugmentationPipeline` → `(x̃ᵢ, x̃ⱼ)` |

Stats computed on load: **Duration · Tempo (BPM) · Spectral Centroid · Bandwidth · Rolloff · ZCR · RMS**

---

## Augmentation Pipeline

```python
from echofind_gui import AugmentationPipeline
import librosa

aug = AugmentationPipeline(sr=22050)
y, sr = librosa.load("track.mp3", sr=22050)

view_i, ops_i = aug(y)   # Tensor [1, 128, T], list of applied transforms
view_j, ops_j = aug(y)   # Independent stochastic augmentation

# ops example: ['Stretch(x0.93)', 'Pitch(+1.8st)', 'FreqMask(14bands)', 'TimeMask(67fr)']
```

**Invariances encoded:**

| Augmentation | Invariance Learned |
|-------------|-------------------|
| Time Stretching | Tempo variation (±15%) |
| Pitch Shifting | Musical key (±3 semitones) |
| Gaussian Noise | Background/recording noise |
| Frequency Masking | Spectral dropout |
| Time Masking | Temporal occlusion |

---

## SSL Dataset Class

```python
from echofind_gui import FMASSLDataset
from torch.utils.data import DataLoader
import glob

paths = glob.glob("fma_small/**/*.mp3", recursive=True)
dataset = FMASSLDataset(audio_paths=paths, target_sr=22050, duration=5.0)
loader  = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

for view_i, view_j, idx in loader:
    # view_i, view_j: [B, 1, 128, T]  — no labels used
    loss = contrastive_loss(encoder(view_i), encoder(view_j))
```

Genre labels are **never** passed to the model during pre-training.

---

## Organiser Evaluation Checks

```python
import numpy as np, torch
from echofind_gui import AugmentationPipeline

aug   = AugmentationPipeline(sr=22050)
dummy = np.random.randn(22050 * 5).astype(np.float32) * 0.1
v1, _ = aug(dummy)
v2, _ = aug(dummy)

v1t, v2t = torch.tensor(v1), torch.tensor(v2)
assert not torch.allclose(v1t, v2t),        "FAIL: identical views"
assert 0.05 < v1t.std() < 0.5,             "FAIL: looks like noise"
assert v1.shape[0] == 1,                   "FAIL: wrong channel dim"
print("All Phase 1 evaluation checks passed")
```

---

## Dataset

**FMA-Small** — 8,000 tracks x 30 seconds, 8 balanced genres.

```bash
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip && unzip fma_small.zip
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip && unzip fma_metadata.zip
```

> **Constraint**: `tracks.csv` genre labels must NOT be used during Phase 2 pre-training.

---

## Mathematical Reference

**Mel Scale:** `m = 2595 · log₁₀(1 + f / 700)`

**NT-Xent Loss:** `ℓ(i,j) = −log [ exp(sim(zᵢ,zⱼ)/τ) / Σₖ≠ᵢ exp(sim(zᵢ,zₖ)/τ) ]`

**Retrieval Score:** `score(q,v) = (q·v) / (‖q‖·‖v‖)`

---

## References

1. McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015.
2. Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020.
3. Grill et al. (2020). *Bootstrap Your Own Latent.* NeurIPS 2020.
4. Zbontar et al. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* ICML 2021.
5. Park et al. (2019). *SpecAugment: A Simple Data Augmentation Method for ASR.* Interspeech 2019.
6. Defferrard et al. (2017). *FMA: A Dataset for Music Analysis.* ISMIR 2017.
7. Fitzgerald (2010). *Harmonic/Percussive Separation Using Median Filtering.* DAFx-10.
8. IEEE Impulse 2026 Problem Statement — EchoFind.

---

## License

MIT License.
