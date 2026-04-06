import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import random
import os
import numpy as np

import librosa
import librosa.display
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#09090f"
SURFACE = "#0f0f18"
PANEL   = "#13131e"
BORDER  = "#1e1e32"
ACC1    = "#a78bfa"   # purple
ACC2    = "#f472b6"   # pink
ACC3    = "#34d399"   # green
ACC4    = "#60a5fa"   # blue
TEXT    = "#e2d9f3"
MUTED   = "#6b6b8a"


# ── Augmentation Pipeline ──────────────────────────────────────────────────────
class AugmentationPipeline:
    def __init__(self, sr=22050):
        self.sr = sr

    def time_masking(self, mel, param=80):
        mel = mel.copy()
        t = random.randint(0, min(param, mel.shape[1] - 1))
        t0 = random.randint(0, mel.shape[1] - t)
        mel[:, t0:t0 + t] = mel.min()
        return mel, f"TimeMask({t}fr)"

    def frequency_masking(self, mel, param=30):
        mel = mel.copy()
        f = random.randint(0, min(param, mel.shape[0] - 1))
        f0 = random.randint(0, mel.shape[0] - f)
        mel[f0:f0 + f, :] = mel.min()
        return mel, f"FreqMask({f}bands)"

    def time_stretch(self, y, rate_range=(0.85, 1.15)):
        rate = random.uniform(*rate_range)
        return librosa.effects.time_stretch(y, rate=rate), f"Stretch(×{rate:.2f})"

    def pitch_shift(self, y, steps_range=(-3, 3)):
        n = random.uniform(*steps_range)
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n), f"Pitch({n:+.1f}st)"

    def add_noise(self, y, std_range=(0.005, 0.02)):
        std = random.uniform(*std_range)
        return np.clip(y + np.random.normal(0, std, y.shape), -1.0, 1.0), f"Noise(σ={std:.3f})"

    def __call__(self, y):
        ops = []
        wav = y.copy()
        if random.random() < 0.6:
            wav, op = self.time_stretch(wav); ops.append(op)
        if random.random() < 0.5:
            wav, op = self.pitch_shift(wav);  ops.append(op)
        if random.random() < 0.7:
            wav, op = self.add_noise(wav);    ops.append(op)

        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=128), ref=np.max)
        norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        if random.random() < 0.8:
            norm, op = self.frequency_masking(norm); ops.append(op)
        if random.random() < 0.8:
            norm, op = self.time_masking(norm);      ops.append(op)

        return norm, ops


# ── Main Application ───────────────────────────────────────────────────────────
class EchoFindApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Analyzer")
        self.geometry("1280x820")
        self.minsize(1000, 700)
        self.configure(bg=BG)

        self.audio_path = None
        self.y = None
        self.sr = 22050
        self.aug = AugmentationPipeline(sr=self.sr)

        self._build_ui()

    # ── UI Layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._style()

        # ── Header ──
        hdr = tk.Frame(self, bg=SURFACE, pady=12, padx=20)
        hdr.pack(fill="x", side="top")

        tk.Label(hdr, text="🎵 Audio Analyzer", font=("Courier", 18, "bold"),
                 bg=SURFACE, fg=ACC1).pack(side="left")
        tk.Label(hdr, text="  /  GROOVY!!!",
                 font=("Courier", 12), bg=SURFACE, fg=MUTED).pack(side="left")

        tk.Button(hdr, text="  Open Audio File  ", command=self._open_file,
                  font=("Courier", 10, "bold"),
                  bg=ACC1, fg="#0a0614", relief="flat",
                  activebackground="#c4b5fd", cursor="hand2",
                  padx=12, pady=6).pack(side="right")

        self.file_label = tk.Label(hdr, text="No file loaded",
                                   font=("Courier", 9), bg=SURFACE, fg=MUTED)
        self.file_label.pack(side="right", padx=14)

        # ── Status / progress bar ──
        status_frame = tk.Frame(self, bg=BG, pady=4)
        status_frame.pack(fill="x", padx=20)
        self.status_var = tk.StringVar(value="Open an audio file to begin.")
        tk.Label(status_frame, textvariable=self.status_var,
                 font=("Courier", 9), bg=BG, fg=MUTED).pack(side="left")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate",
                                        length=180, style="EF.Horizontal.TProgressbar")
        self.progress.pack(side="right")

        # ── Stats strip ──
        self.stats_frame = tk.Frame(self, bg=PANEL, pady=8)
        self.stats_frame.pack(fill="x", padx=20, pady=(6, 0))
        self._stat_labels = {}
        for key in ["Duration", "Tempo", "Centroid", "Bandwidth", "Rolloff", "ZCR", "RMS"]:
            col = tk.Frame(self.stats_frame, bg=PANEL, padx=18)
            col.pack(side="left")
            tk.Label(col, text=key.upper(), font=("Courier", 7),
                     bg=PANEL, fg=MUTED).pack()
            lbl = tk.Label(col, text="—", font=("Courier", 13, "bold"),
                           bg=PANEL, fg=TEXT)
            lbl.pack()
            self._stat_labels[key] = lbl

        # ── Notebook (tabs) ──
        nb_frame = tk.Frame(self, bg=BG)
        nb_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.nb = ttk.Notebook(nb_frame, style="EF.TNotebook")
        self.nb.pack(fill="both", expand=True)

        tab_defs = [
            ("① Mel Spec",   self._tab_mel),
            ("② STFT",       self._tab_stft),
            ("③ Waveform",   self._tab_waveform),
            ("④ Chroma",     self._tab_chroma),
            ("⑤ MFCC",      self._tab_mfcc),
            ("⑥ Spectral",  self._tab_spectral),
            ("⑦ Beats",     self._tab_beats),
            ("⑧ HPSS",      self._tab_hpss),
            ("⑨ SSL Views", self._tab_aug),
        ]

        self._tab_frames = {}
        self._canvases   = {}
        self._drawn      = set()

        for label, builder in tab_defs:
            frame = tk.Frame(self.nb, bg=BG)
            self.nb.add(frame, text=f"  {label}  ")
            self._tab_frames[label] = (frame, builder)

        # Draw tab on first selection
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("EF.TNotebook",            background=BG, borderwidth=0)
        s.configure("EF.TNotebook.Tab",
                    background=PANEL, foreground=MUTED,
                    font=("Courier", 9), padding=[10, 5],
                    borderwidth=0)
        s.map("EF.TNotebook.Tab",
              background=[("selected", SURFACE)],
              foreground=[("selected", ACC1)])
        s.configure("EF.Horizontal.TProgressbar",
                    troughcolor=BORDER, background=ACC1, thickness=3)

    # ── File open ─────────────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a"), ("All", "*.*")])
        if not path:
            return
        self.audio_path = path
        self.file_label.config(text=os.path.basename(path), fg=ACC3)
        self._drawn.clear()
        self._canvases.clear()
        # Clear all tab frames
        for label, (frame, _) in self._tab_frames.items():
            for w in frame.winfo_children():
                w.destroy()
        self._load_audio_async()

    def _load_audio_async(self):
        self.status_var.set("Loading & resampling audio…")
        self.progress.start(12)
        threading.Thread(target=self._load_audio, daemon=True).start()

    def _load_audio(self):
        try:
            y, sr = librosa.load(self.audio_path, sr=self.sr, mono=True)
            self.y = y
            stats = self._compute_stats(y, sr)
            self.after(0, lambda: self._on_loaded(stats))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _compute_stats(self, y, sr):
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid  = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        rolloff   = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
        zcr       = float(librosa.feature.zero_crossing_rate(y).mean())
        rms       = float(librosa.feature.rms(y=y).mean())
        return {
            "Duration":  f"{len(y)/sr:.1f}s",
            "Tempo":     f"{float(tempo):.0f} BPM",
            "Centroid":  f"{centroid:.0f} Hz",
            "Bandwidth": f"{bandwidth:.0f} Hz",
            "Rolloff":   f"{rolloff:.0f} Hz",
            "ZCR":       f"{zcr:.4f}",
            "RMS":       f"{rms:.4f}",
        }

    def _on_loaded(self, stats):
        self.progress.stop()
        self.status_var.set("✓ Loaded — select a tab to visualize")
        for key, val in stats.items():
            self._stat_labels[key].config(text=val)
        # Draw first tab automatically
        self._draw_current_tab()

    def _on_error(self, msg):
        self.progress.stop()
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("EchoFind Error", msg)

    # ── Tab switching ─────────────────────────────────────────────────────────
    def _on_tab_change(self, _event):
        self._draw_current_tab()

    def _draw_current_tab(self):
        if self.y is None:
            return
        idx   = self.nb.index(self.nb.select())
        label = list(self._tab_frames.keys())[idx]
        if label in self._drawn:
            return
        frame, builder = self._tab_frames[label]
        self.status_var.set(f"Rendering {label.strip()}…")
        self.progress.start(12)
        threading.Thread(target=lambda: self._render(label, frame, builder), daemon=True).start()

    def _render(self, label, frame, builder):
        try:
            fig = builder()
            self.after(0, lambda: self._embed_figure(label, frame, fig))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _embed_figure(self, label, frame, fig):
        self.progress.stop()
        self.status_var.set(f"✓ {label.strip()}")
        self._drawn.add(label)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.configure(bg=PANEL)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Figure builders ───────────────────────────────────────────────────────
    def _base_fig(self, rows=1, cols=1, h=5):
        fig, ax = plt.subplots(rows, cols, figsize=(12, h))
        fig.patch.set_facecolor(BG)
        return fig, ax

    def _style_ax(self, ax, title, color=ACC1):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # ① Mel Spectrogram
    def _tab_mel(self):
        fig, ax = self._base_fig(h=5)
        mel = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=self.sr, x_axis="time",
                                       y_axis="mel", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        self._style_ax(ax, "Log-Mel Spectrogram  (128 Mel bands, 22 050 Hz)", ACC1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel Frequency")
        fig.tight_layout()
        return fig

    # ② STFT
    def _tab_stft(self):
        fig, ax = self._base_fig(h=5)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis="time",
                                       y_axis="log", ax=ax, cmap="inferno")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        self._style_ax(ax, "STFT Log-Power Spectrogram  (Linear → Log Freq)", ACC2)
        fig.tight_layout()
        return fig

    # ③ Waveform
    def _tab_waveform(self):
        fig, ax = self._base_fig(h=4)
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax, color=ACC1, alpha=0.85)
        self._style_ax(ax, "Waveform  (Raw PCM @ 22 050 Hz)", ACC1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        return fig

    # ④ Chroma
    def _tab_chroma(self):
        fig, ax = self._base_fig(h=5)
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        img = librosa.display.specshow(chroma, sr=self.sr, x_axis="time",
                                       y_axis="chroma", ax=ax, cmap="plasma")
        fig.colorbar(img, ax=ax)
        self._style_ax(ax, "Chromagram  (12 Pitch Classes via CQT)", ACC3)
        fig.tight_layout()
        return fig

    # ⑤ MFCC
    def _tab_mfcc(self):
        fig, ax = self._base_fig(h=5)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20)
        img = librosa.display.specshow(mfccs, sr=self.sr, x_axis="time",
                                       ax=ax, cmap="coolwarm")
        fig.colorbar(img, ax=ax)
        self._style_ax(ax, "MFCCs  (20 Mel-Frequency Cepstral Coefficients)", ACC2)
        fig.tight_layout()
        return fig

    # ⑥ Spectral features
    def _tab_spectral(self):
        fig, ax = self._base_fig(h=4)
        ax.set_facecolor(PANEL)
        mel  = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        t    = librosa.times_like(mel, sr=self.sr)
        cent = librosa.feature.spectral_centroid(y=self.y,  sr=self.sr)[0]
        bw   = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        roll = librosa.feature.spectral_rolloff(y=self.y,   sr=self.sr)[0]
        ax.fill_between(t, cent / cent.max(), alpha=0.15, color=ACC1)
        ax.plot(t, cent / cent.max(),  color=ACC1, lw=1.6, label="Centroid")
        ax.fill_between(t, bw   / bw.max(),   alpha=0.10, color=ACC2)
        ax.plot(t, bw   / bw.max(),   color=ACC2, lw=1.6, label="Bandwidth")
        ax.fill_between(t, roll / roll.max(), alpha=0.10, color=ACC3)
        ax.plot(t, roll / roll.max(), color=ACC3, lw=1.6, label="Rolloff")
        ax.legend(facecolor=BORDER, labelcolor=TEXT, fontsize=9, framealpha=0.8)
        self._style_ax(ax, "Spectral Features Over Time  (Normalized)", ACC1)
        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    # ⑦ Beats & Onsets
    def _tab_beats(self):
        fig, ax = self._base_fig(h=4)
        ax.set_facecolor(PANEL)
        onset_env   = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times  = librosa.frames_to_time(beats, sr=self.sr)
        onset_times = librosa.times_like(onset_env, sr=self.sr)
        ax.fill_between(onset_times, onset_env / onset_env.max(),
                        alpha=0.25, color=ACC4)
        ax.plot(onset_times, onset_env / onset_env.max(),
                color=ACC4, lw=1.4, label="Onset Strength")
        for bt in beat_times:
            ax.axvline(bt, color="#f59e0b", alpha=0.55, lw=0.9)
        ax.legend(facecolor=BORDER, labelcolor=TEXT, fontsize=9)
        self._style_ax(ax,
            f"Onset Strength  +  Beat Positions   |   Tempo ≈ {float(tempo):.1f} BPM",
            ACC4)
        ax.set_xlabel("Time (s)")
        fig.tight_layout()
        return fig

    # ⑧ HPSS
    def _tab_hpss(self):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.patch.set_facecolor(BG)
        y_h, y_p = librosa.effects.hpss(self.y)
        for ax, sig, label, color in zip(
            axes, [y_h, y_p], ["Harmonic", "Percussive"], [ACC1, ACC2]
        ):
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(y=sig, sr=self.sr), ref=np.max)
            img = librosa.display.specshow(mel, sr=self.sr, x_axis="time",
                                           y_axis="mel", ax=ax, cmap="magma")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            self._style_ax(ax, f"{label} Component — Mel Spectrogram", color)
        fig.tight_layout()
        return fig

    # ⑨ SSL Augmented views
    def _tab_aug(self):
        orig_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128), ref=np.max)
        norm = (orig_mel - orig_mel.min()) / (orig_mel.max() - orig_mel.min() + 1e-8)

        v1, ops1 = self.aug(self.y)
        v2, ops2 = self.aug(self.y)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor(BG)

        specs   = [norm, v1, v2]
        titles  = ["Original Log-Mel",
                   f"View 1 (x̃ᵢ)\n{' · '.join(ops1) or 'No aug'}",
                   f"View 2 (x̃ⱼ)\n{' · '.join(ops2) or 'No aug'}"]
        cmaps   = ["magma", "inferno", "plasma"]
        accents = [ACC1, ACC2, ACC3]

        for ax, spec, title, cmap, acc in zip(axes, specs, titles, cmaps, accents):
            ax.set_facecolor(PANEL)
            im = ax.imshow(spec, aspect="auto", origin="lower",
                           cmap=cmap, interpolation="nearest")
            ax.set_title(title, color=acc, fontsize=9, fontweight="bold")
            ax.tick_params(colors=TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(acc); spine.set_linewidth(1.8)
            ax.set_xlabel("Time Frames", color=TEXT, fontsize=8)
            ax.set_ylabel("Mel Bands",   color=TEXT, fontsize=8)
            fig.colorbar(im, ax=ax)

        fig.suptitle("SSL Augmentation Pipeline — Two Distinct Views (SimCLR Protocol)",
                     color=TEXT, fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = EchoFindApp()
    app.mainloop()
