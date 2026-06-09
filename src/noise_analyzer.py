#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from collections import deque

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

audio_buffer = deque(maxlen=20)

window = get_window("hann", BLOCK_SIZE)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    x = indata[:, 0].copy()

    audio_buffer.append(x)

    outdata.fill(0)

stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    callback=callback
)

stream.start()

plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1)

freqs = np.fft.rfftfreq(BLOCK_SIZE, 1/SAMPLE_RATE)

line_fft, = ax1.plot(freqs, np.zeros(len(freqs)))
line_wave, = ax2.plot(np.zeros(BLOCK_SIZE))

ax1.set_title("Spectrum")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude (dB)")

ax2.set_title("Waveform")
ax2.set_xlabel("Sample Number")
ax2.set_ylabel("Amplitude")

ax1.set_xlim(0, 10000)
ax1.set_ylim(-120, 0)

ax2.set_ylim(-1, 1)

print("Running...")
print("Press Ctrl+C to quit")

try:
    while True:

        if not audio_buffer:
            plt.pause(0.01)
            continue

        x = audio_buffer[-1]

        rms = np.sqrt(np.mean(x**2))
        peak = np.max(np.abs(x))

        fft = np.fft.rfft(x * window)

        mag = np.abs(fft)

        mag_db = 20*np.log10(mag + 1e-10)

        line_fft.set_ydata(mag_db)
        line_wave.set_ydata(x)

        ax1.set_title(
            f"Spectrum | RMS={rms:.4f} Peak={peak:.4f}"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(0.01)

except KeyboardInterrupt:
    pass

stream.stop()
stream.close()