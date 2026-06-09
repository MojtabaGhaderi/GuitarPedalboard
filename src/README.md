# 🎸 GuitarPedalboard

A real-time, Python-based guitar pedalboard and audio analysis suite. Built for low-latency live processing, this project turns your computer + audio interface into a multi-effect rig with a noise gate, compressor, distortion, cabinet simulation, and more.

> **Status:** Active development — core DSP engine is functional. Effects, looping, and a GUI are on the roadmap.

---

## ✨ Features

### 🔊 Real-Time Pedalboard (`passthrough.py`)
A command-line guitar processor that runs as a live audio stream.

| Feature | Description |
|---------|-------------|
| **Noise Gate** | Hysteresis gate with separate open/close thresholds to eliminate hiss without chatter |
| **Compressor** | Smooth peak-detecting compressor with adjustable threshold and ratio |
| **Overdrive / Distortion** | Multi-stage tanh distortion with pre/post filtering |
| **Cabinet Simulation** | Stateful convolution with custom IR support (or a built-in synthetic IR) |
| **Acoustic → Electric** | Notch filters, peaking EQ, and low-pass shaping for acoustic guitar pickup simulation |
| **Presence** | High-shelf boost for cutting through the mix |
| **Clean Blend** | Mix dry signal back in for parallel processing |
| **Output Recording** | Record your processed output to timestamped WAV files (`recordings/`) |
| **Live Tuning** | Adjust drive, EQ, gain, volume, gate, and compressor via keyboard commands while playing |

### 📊 Spectrum Analyzer (`noise_analyzer.py`)
A real-time FFT visualizer for monitoring your signal chain.

- Live waveform display
- Frequency spectrum (0–10 kHz) in dB
- RMS and peak level metering
- Minimal latency with `sounddevice` + `matplotlib`

---

## 📁 Project Structure

```
GuitarPedalboard/
├── .venv/                  # Python virtual environment (ignored)
├── recordings/             # Saved WAV recordings (ignored)
├── src/
│   ├── passthrough.py      # Main pedalboard engine
│   ├── noise_analyzer.py   # Real-time spectrum/waveform visualizer
│   ├── .gitignore          # Git ignore rules
│   └── requirements.txt    # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Requirements

- Python 3.9+
- A working audio interface (or PipeWire / PulseAudio on Linux)
- Optional: `cabinet.wav` in the project root for custom cabinet impulse response

### Python Dependencies

```
sounddevice
numpy
scipy
matplotlib
soundfile
```

> `soundfile` is optional but required for recording output to WAV.

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/MojtabaGhaderi/GuitarPedalboard.git
cd GuitarPedalboard

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🎛️ Usage

### 1. Start the Pedalboard

```bash
cd src
python passthrough.py
```

The stream will start and print a list of available audio devices. By default it tries to use `pipewire`; edit the `device=` parameter in `main()` if you need a different backend.

### 2. Interactive Commands

While the pedalboard is running, type commands and press **Enter**:

| Command | Action |
|---------|--------|
| `d` | Toggle overdrive distortion ON/OFF |
| `ae` | Toggle acoustic-to-electric shaping ON/OFF |
| `drive <value>` | Set distortion drive (0.5 – 20.0) |
| `bass <hz>` | Set high-pass / bass-cut frequency (20 – 200 Hz) |
| `treble <hz>` | Set low-pass / treble-cut frequency (2000 – 10000 Hz) |
| `presence <0-1>` | Set presence boost amount |
| `gain <0.1-1.0>` | Set clean input gain |
| `vol <0.1-2.0>` | Set master output volume |
| `makeup <1.0-3.0>` | Set distortion make-up gain |
| `blend <0.0-1.0>` | Mix dry signal in (0 = full wet, 1 = full dry) |
| `notch <freq> [Q]` | Enable notch filter at frequency (Hz). Use `notch 0` to disable |
| `gate <threshold>` | Set noise gate open threshold (e.g., `gate 0.008`) |
| `gate_on` / `gate_off` | Enable / disable noise gate |
| `comp <threshold_db>` | Set compressor threshold in dB (e.g., `comp -18`) |
| `comp_on` / `comp_off` | Enable / disable compressor |
| `loadir` | Reload cabinet impulse response |
| `r` | Start / stop recording output to `recordings/guitar_YYYYMMDD_HHMMSS.wav` |
| `q` | Quit |

> **Tip:** Watch the `Input peak` meter printed at the bottom of the terminal. Aim for **0.3 – 0.8** for best signal-to-noise ratio.

### 3. Run the Spectrum Analyzer

In a second terminal:

```bash
cd src
python noise_analyzer.py
```

This opens a live matplotlib window showing:
- **Top:** Frequency spectrum (dB magnitude)
- **Bottom:** Time-domain waveform

Press **Ctrl+C** in the terminal to stop.

---

## 🗺️ Roadmap

- [x] Real-time audio I/O with `sounddevice`
- [x] Noise gate with hysteresis
- [x] Compressor
- [x] Multi-stage distortion & cabinet convolution
- [x] Acoustic-to-electric simulator
- [x] Live keyboard command interface
- [x] Output recording to WAV
- [x] Real-time spectrum analyzer
- [ ] **Looper** — record, overdub, and playback loops with tempo sync
- [ ] **More Effects** — delay, reverb, chorus, phaser, tremolo
- [ ] **Tuner** — chromatic pitch detection display
- [ ] **MIDI / Footswitch Support** — hands-free control via MIDI CC or GPIO
- [ ] **GUI** — desktop or web interface for visual pedal arrangement
- [ ] **Presets** — save and recall full effect chains as named presets
- [ ] **Standalone Packaging** — single executable / installer

---

## 🤝 Contributing

This is a personal project in active development. If you have ideas for DSP algorithms, UI designs, or hardware integration, feel free to open an issue or pull request.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.