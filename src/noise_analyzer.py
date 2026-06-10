#!/usr/bin/env python3
"""
Advanced Noise Analyzer & Signal Monitor
==========================================
A real-time audio monitoring tool with:
- Proper dBFS-normalized spectrum (0 dB = digital full scale)
- Log-frequency spectrum axis for better musical readability
- Peak hold with gradual decay
- Rolling spectrogram (waterfall / heatmap)
- Chromatic note detection & tuner display
- Color-coded level monitoring (green / orange / red)
- Multi-second rolling waveform history

Press Ctrl+C to quit.
"""

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from collections import deque

# ========== Audio Settings ==========
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

# ========== Display Settings ==========
HISTORY_SECONDS = 3.0           # Rolling waveform length
SPECTROGRAM_SECONDS = 2.0       # Spectrogram time window

# ========== Buffers ==========
audio_buffer = deque(maxlen=1)
waveform_buffer = deque(maxlen=int(SAMPLE_RATE * HISTORY_SECONDS))

# Spectrogram ring buffer (rows = frequency bins, cols = time frames)
# This shape is required by imshow: rows -> y-axis (freq), cols -> x-axis (time)
spec_frames = int(SAMPLE_RATE * SPECTROGRAM_SECONDS / BLOCK_SIZE)
spec_buffer = None  # initialized after we know n_bins
spec_idx = 0

# Window & coherent gain for dBFS normalization
window = get_window("hann", BLOCK_SIZE)
coherent_gain = np.sum(window)

# Note mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def freq_to_note(freq):
    """Convert frequency to note name, octave, and cents deviation."""
    if freq <= 20:
        return "---", 0, 0.0
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_rounded = int(round(midi))
    note_idx = midi_rounded % 12
    octave = (midi_rounded // 12) - 1
    cents = (midi - midi_rounded) * 100
    return NOTE_NAMES[note_idx], octave, cents


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    x = indata[:, 0].copy() if indata.shape[1] >= 1 else np.zeros(frames)
    audio_buffer.append(x)
    waveform_buffer.extend(x)
    outdata.fill(0)


# ========== Setup Stream ==========
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    callback=callback
)
stream.start()

# ========== Setup Plots ==========
plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.5, 0.4, 1.2], hspace=0.35)

freqs = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)
n_bins = len(freqs)

# Initialize spectrogram buffer now that we know n_bins
# Shape: (frequency_bins, time_frames)
spec_buffer = np.zeros((n_bins, spec_frames))

# --- 1. Spectrogram (top) ---
ax_spec = fig.add_subplot(gs[0])
spec_img = ax_spec.imshow(
    np.zeros((n_bins, spec_frames)),
    aspect='auto',
    origin='lower',
    extent=[0, SPECTROGRAM_SECONDS, 0, SAMPLE_RATE / 2],
    cmap='inferno',
    vmin=-120, vmax=0
)
ax_spec.set_ylabel('Frequency (Hz)')
ax_spec.set_title('Spectrogram (History)')
ax_spec.set_ylim(20, 10000)
cbar = fig.colorbar(spec_img, ax=ax_spec, orientation='vertical', pad=0.01)
cbar.set_label('dBFS')

# --- 2. Spectrum (second) ---
ax_spectrum = fig.add_subplot(gs[1])
line_fft, = ax_spectrum.plot(freqs, np.zeros(n_bins), label='Live', linewidth=1.2)
line_peak, = ax_spectrum.plot(freqs, np.zeros(n_bins), '--', alpha=0.6, color='orange', label='Peak Hold')
line_dom = ax_spectrum.axvline(x=0, color='red', alpha=0.5, linewidth=2, label='Detected F0')
ax_spectrum.set_xlim(20, 10000)
ax_spectrum.set_ylim(-120, 6)
ax_spectrum.set_xscale('log')
ax_spectrum.set_xlabel('Frequency (Hz)')
ax_spectrum.set_ylabel('Magnitude (dBFS)')
ax_spectrum.set_title('Spectrum')
ax_spectrum.legend(loc='upper right', fontsize=8)
ax_spectrum.grid(True, which='both', linestyle='--', alpha=0.3)

# Add standard guitar frequency markers
for f in [82, 110, 147, 196, 247, 330, 440, 660, 880]:
    if 20 <= f <= 10000:
        ax_spectrum.axvline(x=f, color='cyan', linestyle=':', alpha=0.3, linewidth=0.8)

# --- 3. Tuner / Info Bar (third, compact) ---
ax_tuner = fig.add_subplot(gs[2])
ax_tuner.set_xlim(0, 1)
ax_tuner.set_ylim(0, 1)
ax_tuner.axis('off')

# Background bar (gray) and foreground bar (level meter)
bar_bg = ax_tuner.barh(0.5, 1.0, height=0.6, color='lightgray', alpha=0.3, left=0)
bar_fg = ax_tuner.barh(0.5, 0.0, height=0.6, color='green', alpha=0.8, left=0)

tuner_text = ax_tuner.text(
    0.5, 0.5, 'Initializing...',
    ha='center', va='center', fontsize=12, family='monospace',
    transform=ax_tuner.transAxes
)

# --- 4. Rolling Waveform (bottom) ---
ax_wave = fig.add_subplot(gs[3])
wave_time = np.linspace(-HISTORY_SECONDS, 0, waveform_buffer.maxlen)
line_wave, = ax_wave.plot(wave_time, np.zeros(waveform_buffer.maxlen), linewidth=0.8)
ax_wave.set_xlim(-HISTORY_SECONDS, 0)
ax_wave.set_ylim(-1.1, 1.1)
ax_wave.set_xlabel('Time (s)')
ax_wave.set_ylabel('Amplitude')
ax_wave.set_title('Waveform History')
ax_wave.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax_wave.axhline(y=-0.8, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax_wave.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
ax_wave.axhline(y=-0.99, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

# Peak hold state
peak_hold = np.full(n_bins, -120.0)
peak_decay = 0.97

print("Running Advanced Noise Analyzer...")
print("Press Ctrl+C to quit")

# ========== Main Loop ==========
try:
    while True:
        if not audio_buffer:
            plt.pause(0.01)
            continue

        x = audio_buffer[-1]

        # --- FFT with proper dBFS normalization ---
        fft = np.fft.rfft(x * window)
        mag = np.abs(fft) / coherent_gain
        mag_db = 20 * np.log10(mag + 1e-10)

        # --- Update spectrogram ---
        spec_buffer[:, spec_idx] = mag_db
        spec_idx = (spec_idx + 1) % spec_frames
        # Roll so oldest time is at column 0, newest at right edge
        rolled = np.roll(spec_buffer, -spec_idx, axis=1)
        spec_img.set_array(rolled)

        # --- Update peak hold ---
        peak_hold = np.maximum(peak_hold, mag_db)
        peak_hold *= peak_decay
        peak_hold = np.maximum(peak_hold, mag_db)  # clamp to current so it never hides live signal

        # --- Update spectrum lines ---
        line_fft.set_ydata(mag_db)
        line_peak.set_ydata(peak_hold)

        # --- Dominant frequency / Tuner ---
        valid_range = (freqs > 50) & (freqs < 5000)
        valid_db = np.where(valid_range, mag_db, -np.inf)

        if np.max(valid_db) > -60:
            dom_idx = np.argmax(valid_db)
            dom_freq = freqs[dom_idx]
            note, octave, cents = freq_to_note(dom_freq)
            line_dom.set_xdata([dom_freq])
            line_dom.set_ydata([0, 1])
            line_dom.set_visible(True)
        else:
            note, octave, cents = "---", 0, 0.0
            dom_freq = 0.0
            line_dom.set_visible(False)

        # --- RMS & Peak ---
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))

        # --- Update tuner text & level bar ---
        tuner_str = (
            f'  Note: {note}{octave}  |  '
            f'Cents: {cents:+5.1f}  |  '
            f'RMS: {rms:.4f}  |  '
            f'Peak: {peak:.4f}  '
        )
        tuner_text.set_text(tuner_str)

        # Color based on clipping level
        if peak >= 0.99:
            bar_color = 'red'
            tuner_text.set_color('red')
        elif peak >= 0.8:
            bar_color = 'orange'
            tuner_text.set_color('orange')
        else:
            bar_color = 'green'
            tuner_text.set_color('green')

        bar_fg[0].set_width(min(peak, 1.0))
        bar_fg[0].set_color(bar_color)

        # --- Update rolling waveform ---
        wave_data = np.array(waveform_buffer)
        if len(wave_data) < waveform_buffer.maxlen:
            wave_data = np.pad(
                wave_data,
                (waveform_buffer.maxlen - len(wave_data), 0),
                'constant'
            )
        line_wave.set_ydata(wave_data)
        line_wave.set_color(bar_color)

        # --- Update title ---
        if line_dom.get_visible():
            ax_spectrum.set_title(
                f'Spectrum | RMS={rms:.4f} | Peak={peak:.4f} | F0={dom_freq:.1f} Hz'
            )
        else:
            ax_spectrum.set_title(
                f'Spectrum | RMS={rms:.4f} | Peak={peak:.4f}'
            )

        # --- Draw ---
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.005)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop()
    stream.close()
    plt.close('all')