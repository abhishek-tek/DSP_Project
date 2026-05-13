# Audio Analyzer

> A local desktop application for interactive audio signal analysis and self-supervised learning preprocessing.
> Built on **librosa**, **Tkinter**, and **Matplotlib** — no browser, no server, no internet required.

---

## What Is This?

Audio Analyzer is a single-file Python desktop application (`Audio_Analyzer.py`) that transforms any audio file into a suite of nine interactive signal-processing visualisations. It runs entirely on your local machine, embedding all computation and rendering inside a native window.

The project serves a dual purpose:

1. **Educational tool** — an interactive, code-free environment for exploring MIR (Music Information Retrieval) concepts grounded in McFee et al. (2015).
2. **Phase 1 Input Pipeline** — preprocessing and augmentation you can reuse for contrastive or other self-supervised setups.

The core premise: before any deep learning model can learn audio representations, the raw waveform must be converted into a perceptually meaningful, augmentation-ready format. This application implements and visualises every step of that transformation.

---

## Architecture Overview

```
Raw Audio File (.mp3 / .wav / .ogg / .flac / .m4a)
         │
         ▼
 librosa.load()  ──  resample to 22,050 Hz, downmix to mono
         │
         ▼
 STFT  (librosa defaults: n_fft=2048, hop_length=512, Hann window)
         │
         ├──── Mel Filterbank (128 bands)
         │         │
         │         ▼
         │     power_to_dB  ──  normalise [0,1]  ──  NumPy [128, T]  (batch as [1,128,T] in your trainer)
         │         │
         │         ▼
         │     AugmentationPipeline
         │     ├── Time Stretch    rate  ~  U[0.85, 1.15]
         │     ├── Pitch Shift     n     ~  U[−3, +3]  semitones
         │     ├── Gaussian Noise  σ     ~  U[0.005, 0.02]
         │     ├── Freq Masking    f     ~  U[0, 30]   Mel bands
         │     └── Time Masking    t     ~  U[0, 80]   frames
         │         │
         │         ▼
         │     (view_i, view_j)  ←  SimCLR-style positive pair
         │
         ├──── Chromagram  (CQT, 12 pitch classes)
         ├──── MFCC        (20 coefficients)
         ├──── Spectral Features  (centroid, bandwidth, rolloff)
         ├──── Onset Strength  +  Beat Tracking
         └──── HPSS  (harmonic / percussive separation)
```

---

## Quick Start

### Requirements

- Python 3.8+
- `tkinter` — ships with the Python standard library (no pip install needed)

### Install

```bash
git clone https://github.com/your-username/audio-analyzer.git
cd audio-analyzer
pip install -r requirements.txt
```

### Run

```bash
python Audio_Analyzer.py
```

Click **Open Audio File** in the header bar, select any supported audio file — the first tab renders immediately; select each remaining tab once to build its plot (see below).

---

## Dependencies

```
librosa>=0.10.0       # Audio I/O, STFT, Mel filterbanks, HPSS, beat tracking
matplotlib>=3.7.0     # All figure rendering via FigureCanvasTkAgg
numpy>=1.24.0         # Array operations throughout
soundfile>=0.12.0     # Audio file backend for librosa (WAV/FLAC/OGG; MP3 often uses audioread/ffmpeg)
```

> **tkinter** is part of the Python standard library. If it is missing on your Linux system, install it with:
> `sudo apt-get install python3-tk`

---

## The Nine Visualisation Tabs

| Tab | Feature | Core Function | What It Shows |
|-----|---------|--------------|---------------|
| ① | Log-Mel Spectrogram | `librosa.feature.melspectrogram` + `power_to_db` | Perceptual frequency energy over time (128 Mel bands) |
| ② | STFT Log-Power | `librosa.stft` + `amplitude_to_db` | Linear-frequency power spectrum |
| ③ | Waveform | `librosa.display.waveshow` | Raw PCM amplitude (post-resample) |
| ④ | Chromagram | `librosa.feature.chroma_cqt` | Pitch-class energy — harmonic and chord content |
| ⑤ | MFCC | `librosa.feature.mfcc` | Timbral envelope (20 cepstral coefficients) |
| ⑥ | Spectral Features | `spectral_centroid / bandwidth / rolloff` | Brightness, spread, energy roll-off over time |
| ⑦ | Beats & Onsets | `onset.onset_strength` + `beat.beat_track` | Rhythmic structure, onset events, BPM |
| ⑧ | HPSS | `librosa.effects.hpss` | Harmonic vs. percussive component Mel spectrograms |
| ⑨ | SSL Augmented Views | `AugmentationPipeline` (custom) | Two stochastic contrastive views (x̃ᵢ, x̃ⱼ) with augmentation labels |

**When tabs compute:** After a file loads, the **first tab (① Mel Spec)** is rendered automatically. Every **other** tab runs its computation in a **background daemon thread** the first time you select it, so the GUI stays responsive.

---

## Statistics Strip

Seven scalar descriptors are computed immediately after file load and displayed in a persistent strip above the tabs:

| Stat | Computation |
|------|------------|
| Duration | `len(y) / sr` |
| Tempo | `librosa.beat.beat_track(y=y, sr=sr)` — Ellis (2007) dynamic programming |
| Spectral Centroid | `mean(librosa.feature.spectral_centroid(y=y, sr=sr))` |
| Bandwidth | `mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))` |
| Rolloff | `mean(librosa.feature.spectral_rolloff(y=y, sr=sr))` |
| ZCR | `mean(librosa.feature.zero_crossing_rate(y))` |
| RMS Energy | `mean(librosa.feature.rms(y=y))` |

---

## Augmentation Pipeline API

The `AugmentationPipeline` class can be imported and used independently:

```python
import librosa
from Audio_Analyzer import AugmentationPipeline

aug = AugmentationPipeline(sr=22050)
y, sr = librosa.load("track.mp3", sr=22050, mono=True)

# Generate two independent augmented views
view_i, ops_i = aug(y)
view_j, ops_j = aug(y)

# view_i, view_j : np.ndarray of shape [128, T], values in [0, 1]
# ops_i, ops_j   : list of str, e.g. ['Stretch(×0.92)', 'Pitch(+2.1st)', 'TimeMask(54fr)', 'FreqMask(12bands)', 'Noise(σ=0.012)']
```

### Augmentation Details

| Transform | Domain | Parameter | Invariance Encoded |
|-----------|--------|-----------|-------------------|
| Time Stretch | Waveform | `rate ~ U[0.85, 1.15]` | Tempo variation |
| Pitch Shift | Waveform | `n_steps ~ U[−3, +3] st` | Musical key |
| Gaussian Noise | Waveform | `σ ~ U[0.005, 0.02]` | Recording environment noise |
| Frequency Masking | Spectrogram | `f ~ U[0, 30] bands` | Spectral dropout |
| Time Masking | Spectrogram | `t ~ U[0, 80] frames` | Temporal occlusion |

Each transform has an independent application probability (0.5 for pitch, 0.6 for stretch, 0.7 for noise, 0.8 for each mask). A typical call applies several of the five transforms.

---

## Using Augmented Views in PyTorch (bring your own dataset)

This repository ships the **GUI** and **`AugmentationPipeline`** only. For training, wrap `aug(y)` in a `torch.utils.data.Dataset` that loads clips, optionally crops to a fixed duration, and returns `torch.from_numpy(view).float().unsqueeze(0)` (shape `[1, 128, T]`) for each view. No `FMASSLDataset` class is included here.

---

## Organiser Evaluation Checks (Phase 1)

All three checks pass with NumPy only (no PyTorch required):

```python
import numpy as np
from Audio_Analyzer import AugmentationPipeline

aug   = AugmentationPipeline(sr=22050)
dummy = (np.random.randn(22050 * 5) * 0.1).astype(np.float32)

v1, _ = aug(dummy)
v2, _ = aug(dummy)

assert not np.allclose(v1, v2),    f"FAIL distinctness — mean diff = {np.abs(v1-v2).mean():.4f}"
assert 0.05 < v1.std() < 0.50,     f"FAIL structure — std = {v1.std():.4f}"
assert v1.shape[0] == 128,         f"FAIL shape — got {v1.shape}"

print("All Phase 1 evaluation checks PASSED")
# >> All Phase 1 evaluation checks PASSED
```

---

## Mathematical Reference

**Mel Scale** (Stevens, Volkmann & Newman, 1937):
```
m = 2595 × log₁₀(1 + f / 700)
```

**NT-Xent Contrastive Loss** (Chen et al., 2020):
```
ℓ(i,j) = −log [ exp(sim(zᵢ,zⱼ) / τ) / Σ_{k≠i} exp(sim(zᵢ,zₖ) / τ) ]

where  sim(u,v) = uᵀv / (‖u‖ · ‖v‖)  and  τ ∈ [0.1, 0.5]
```

**Cosine Similarity for Retrieval** (Phase 3):
```
score(q, v) = (q · v) / (‖q‖ · ‖v‖)
```

---

## Project Structure

```
audio-analyzer/
├── Audio_Analyzer.py    # Complete application — run this
├── requirements.txt     # pip dependencies
├── README.md
```
