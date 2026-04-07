# Audio Analyzer

> A local desktop application for interactive audio signal analysis and self-supervised learning preprocessing.
> Built on **librosa**, **Tkinter**, and **Matplotlib** — no browser, no server, no internet required.

---

## What Is This?

Audio Analyzer is a single-file Python desktop application (`echofind_gui.py`) that transforms any audio file into a suite of nine interactive signal-processing visualisations in under two seconds. It runs entirely on your local machine, embedding all computation and rendering inside a native window.

The project serves a dual purpose:

1. **Educational tool** — an interactive, code-free environment for exploring MIR (Music Information Retrieval) concepts grounded in McFee et al. (2015).
2. **Phase 1 Input Pipeline** — the complete preprocessing and augmentation stack for *EchoFind*, a self-supervised audio representation system submitted to **IEEE Impulse 2026** (Signal Processing & Deep Learning Track).

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
 STFT  (n_fft=2048, hop_length=512, Hann window)
         │
         ├──── Mel Filterbank (128 bands)
         │         │
         │         ▼
         │     power_to_dB  ──  normalise [0,1]  ──  Tensor [1, 128, T]
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
         │     (view_i, view_j)  ←  SimCLR positive pair
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
python echofind_gui.py
```

Click **Open Audio File** in the header bar, select any supported audio file, and all nine tabs populate automatically.

---

## Dependencies

```
librosa>=0.10.0       # Audio I/O, STFT, Mel filterbanks, HPSS, beat tracking
matplotlib>=3.7.0     # All figure rendering via FigureCanvasTkAgg
numpy>=1.24.0         # Array operations throughout
soundfile>=0.12.0     # Audio file backend (MP3/FLAC/OGG) for librosa
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

Each tab renders lazily — computation starts only when the tab is first selected — in a background daemon thread, keeping the GUI fully responsive at all times.

---

## Statistics Strip

Seven scalar descriptors are computed immediately after file load and displayed in a persistent strip above the tabs:

| Stat | Computation |
|------|------------|
| Duration | `len(y) / sr` |
| Tempo | `librosa.beat.beat_track(y, sr)` — Ellis (2007) dynamic programming |
| Spectral Centroid | `mean(librosa.feature.spectral_centroid(y, sr))` |
| Bandwidth | `mean(librosa.feature.spectral_bandwidth(y, sr))` |
| Rolloff | `mean(librosa.feature.spectral_rolloff(y, sr))` |
| ZCR | `mean(librosa.feature.zero_crossing_rate(y))` |
| RMS Energy | `mean(librosa.feature.rms(y=y))` |

---

## Augmentation Pipeline API

The `AugmentationPipeline` class can be imported and used independently:

```python
import librosa
from echofind_gui import AugmentationPipeline

aug = AugmentationPipeline(sr=22050)
y, sr = librosa.load("track.mp3", sr=22050, mono=True)

# Generate two independent augmented views
view_i, ops_i = aug(y)
view_j, ops_j = aug(y)

# view_i, view_j : np.ndarray of shape [128, T], values in [0, 1]
# ops_i, ops_j   : list of str, e.g. ['Stretch(x0.92)', 'Pitch(+2.1st)', 'TimeMask(54fr)']
```

### Augmentation Details

| Transform | Domain | Parameter | Invariance Encoded |
|-----------|--------|-----------|-------------------|
| Time Stretch | Waveform | `rate ~ U[0.85, 1.15]` | Tempo variation |
| Pitch Shift | Waveform | `n_steps ~ U[−3, +3] st` | Musical key |
| Gaussian Noise | Waveform | `σ ~ U[0.005, 0.02]` | Recording environment noise |
| Frequency Masking | Spectrogram | `f ~ U[0, 30] bands` | Spectral dropout |
| Time Masking | Spectrogram | `t ~ U[0, 80] frames` | Temporal occlusion |

Each transform has an independent application probability (0.50–0.80). A typical call applies 3–4 of the 5 transforms.

---

## SSL Dataset Class

For use in a PyTorch training loop:

```python
from echofind_gui import FMASSLDataset
from torch.utils.data import DataLoader
import glob

audio_paths = glob.glob("fma_small/**/*.mp3", recursive=True)

dataset = FMASSLDataset(
    audio_paths=audio_paths,
    target_sr=22050,
    duration=5.0           # seconds per clip
)

loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

for view_i, view_j, track_idx in loader:
    # view_i, view_j : FloatTensor [B, 1, 128, T]
    # Genre labels are deliberately excluded — self-supervised only
    embeddings_i = encoder(view_i)
    embeddings_j = encoder(view_j)
    loss = nt_xent_loss(embeddings_i, embeddings_j, temperature=0.2)
    ...
```

---

## Organiser Evaluation Checks (Phase 1)

Per the IEEE Impulse 2026 specification, all three checks pass:

```python
import numpy as np
import torch
from echofind_gui import AugmentationPipeline

aug   = AugmentationPipeline(sr=22050)
dummy = (np.random.randn(22050 * 5) * 0.1).astype(np.float32)

v1, _ = aug(dummy)
v2, _ = aug(dummy)
t1, t2 = torch.tensor(v1), torch.tensor(v2)

assert not torch.allclose(t1, t2),    f"FAIL distinctness — mean diff = {(t1-t2).abs().mean():.4f}"
assert 0.05 < t1.std() < 0.50,       f"FAIL structure — std = {t1.std():.4f}"
assert v1.shape[0] == 128,            f"FAIL shape — got {v1.shape}"

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
├── echofind_gui.py                  # Complete application — run this
├── requirements.txt                 # pip dependencies
├── README.md
├── notebooks/
│   └── phase1_echofind.ipynb       # Jupyter walkthrough + eval checks
└── LICENSE                          # MIT
```

---

## References

1. **McFee et al. (2015)**. *librosa: Audio and Music Signal Analysis in Python*. SciPy 2015. DOI: 10.25080/Majora-7b98e3ed-003
2. **Chen et al. (2020)**. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
3. **Park et al. (2019)**. *SpecAugment: A Simple Data Augmentation Method for ASR*. Interspeech 2019.
4. **Ellis (2007)**. *Beat Tracking by Dynamic Programming*. Journal of New Music Research, 36(1).
5. **Fitzgerald (2010)**. *Harmonic/Percussive Separation Using Median Filtering*. DAFx-10.
6. **Defferrard et al. (2017)**. *FMA: A Dataset for Music Analysis*. ISMIR 2017.
7. **IEEE Impulse 2026**. *EchoFind Problem Statement — Signal Processing & Deep Learning Track*.

---

## License

MIT License — see `LICENSE` for full terms.
