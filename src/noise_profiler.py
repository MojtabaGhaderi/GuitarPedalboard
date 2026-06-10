#!/usr/bin/env python3
"""
Spectral Noise Profiler & Subtractor
=====================================
Real-time noise reduction for guitar pedalboard using overlap-add STFT.

Usage:
    1. Record a noise profile (3s of silence) → saved as noise_profile.npz
    2. Enable spectral subtraction in the pedalboard
    3. Adjust amount if needed (0.0 = none, 1.0 = normal, up to 3.0 = aggressive)
"""

import numpy as np
import os


class SpectralSubtractor:
    """
    Overlap-add STFT spectral subtractor.

    fft_size  : STFT window length (default 2048 → 46 ms, 21.5 Hz bins)
    hop_size  : Must match the audio callback block size (default 512)
    """
    def __init__(self, fft_size=2048, hop_size=512, sample_rate=44100, profile_dir="profiles"):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window = np.hanning(fft_size)
        self.noise_profile = None
        self.amount = 1.0
        self.enabled = False
        
        # Overlap-add state buffers
        self.input_buffer = np.zeros(fft_size)
        self.output_buffer = np.zeros(fft_size)

        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Profile I/O
    # ------------------------------------------------------------------
    def load_profile(self, filename="noise_profile.npz"):
        path = os.path.join(self.profile_dir, filename)
        if os.path.exists(path):
            data = np.load(path)
            self.noise_profile = data["profile"]
            sr = int(data.get("sample_rate", self.sample_rate))
            if sr != self.sample_rate:
                print(f"Warning: profile SR {sr} != {self.sample_rate}")
            print(f"Loaded noise profile from {path}")
            return True
        return False

    def save_profile(self, filename="noise_profile.npz"):
        if self.noise_profile is not None:
            path = os.path.join(self.profile_dir, filename)
            np.savez(path, profile=self.noise_profile, sample_rate=self.sample_rate)
            print(f"Saved noise profile to {path}")
        else:
            print("No noise profile to save")

    # ------------------------------------------------------------------
    # Profile recording (offline, from collected blocks)
    # ------------------------------------------------------------------
    def record_profile(self, blocks):
        """
        Compute average magnitude spectrum from a list of raw audio blocks.
        blocks : list of 1-D numpy arrays (any length, will be concatenated)
        """
        if not blocks:
            print("No blocks provided for profiling")
            return

        audio = np.concatenate(blocks)
        n_frames = (len(audio) - self.fft_size) // self.hop_size + 1
        if n_frames < 1:
            print("Not enough audio for profiling")
            return

        mags = []
        for i in range(n_frames):
            frame = audio[i * self.hop_size : i * self.hop_size + self.fft_size]
            if len(frame) < self.fft_size:
                break
            fft = np.fft.rfft(frame * self.window)
            mags.append(np.abs(fft))

        self.noise_profile = np.mean(mags, axis=0)
        print(f"Profiled {len(mags)} frames ({len(audio)/self.sample_rate:.1f}s) "
              f"→ {len(self.noise_profile)} bins")

    # ------------------------------------------------------------------
    # Real-time processing (called once per audio callback block)
    # ------------------------------------------------------------------
    def process(self, block):
        """
        Process one hop-sized block.
        Returns a hop-sized block.
        """
        if not self.enabled or self.noise_profile is None:
            return block

        # Ensure exact hop_size
        if len(block) < self.hop_size:
            block = np.pad(block, (0, self.hop_size - len(block)), "constant")
        elif len(block) > self.hop_size:
            block = block[:self.hop_size]

        # Shift input buffer and append new block
        self.input_buffer = np.roll(self.input_buffer, -self.hop_size)
        self.input_buffer[-self.hop_size:] = block

        # STFT
        X = np.fft.rfft(self.input_buffer * self.window)
        mag = np.abs(X)
        phase = np.angle(X)

        # Spectral subtraction
        if len(self.noise_profile) != len(mag):
            self.noise_profile = np.resize(self.noise_profile, len(mag))

        mag_sub = mag - self.amount * self.noise_profile
        # Floor at 5% of original to prevent musical noise & total cancellation
        mag_sub = np.maximum(mag_sub, 0.05 * mag)

        # Reconstruct and inverse window
        X_clean = mag_sub * np.exp(1j * phase)
        y = np.fft.irfft(X_clean, n=self.fft_size) * self.window

        # Overlap-add
        self.output_buffer += y
        out = self.output_buffer[:self.hop_size].copy()
        self.output_buffer = np.roll(self.output_buffer, -self.hop_size)
        self.output_buffer[-self.hop_size:] = 0

        return out

    def reset(self):
        """Clear internal buffers (useful when toggling on mid-stream)."""
        self.input_buffer = np.zeros(self.fft_size)
        self.output_buffer = np.zeros(self.fft_size)