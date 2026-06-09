#!/usr/bin/env python3
"""
Professional Guitar Pedalboard – with hysteresis gate and output recording.
- Noise gate with separate open/close thresholds
- Press 'r' to start/stop recording output to file (saved as guitar_YYYYMMDD_HHMMSS.wav)
- All previous improvements: stateful cabinet convolution, fixed compressor, extra low-pass, etc.
"""

import sounddevice as sd
import numpy as np
import time
import threading
import sys
import os
from scipy.signal import butter, sosfilt, iirnotch, lfilter
from datetime import datetime

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Install soundfile for output recording: pip install soundfile")

# ========== Audio Settings ==========
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
LATENCY = 'high'          # change to 'low' if CPU permits

# ========== Effect Parameters ==========
GAIN = 0.6
DISTORTION_ENABLED = False
DRIVE = 6.0
BASS_CUT_HZ = 80.0
TREBLE_CUT_HZ = 4000.0
PRESENCE = 0.15
MASTER_VOLUME = 1.2
MAKEUP_GAIN = 1.6
CLEAN_BLEND = 0.0
ACOUSTIC_TO_ELECTRIC = False

# Noise Gate with Hysteresis
NOISE_GATE_ENABLED = True
NOISE_GATE_OPEN_THRESHOLD = 0.008     # RMS level to open gate (higher than close)
NOISE_GATE_CLOSE_THRESHOLD = 0.004    # RMS level to close gate (lower, prevents chatter)
NOISE_GATE_ATTACK_MS = 5
NOISE_GATE_RELEASE_MS = 80
gate_gain = 0.0
attack_coef = 0.0
release_coef = 0.0
gate_state = 'closed'                 # 'open' or 'closed'

# Compressor
COMPRESSOR_ENABLED = True
COMP_THRESHOLD_DB = -18.0
COMP_RATIO = 2.0
COMP_ATTACK_MS = 10
COMP_RELEASE_MS = 100
comp_gain = 1.0
comp_state = 0.0
comp_attack_coef = 0.0
comp_release_coef = 0.0

# Output recording
RECORDING = False
RECORDINGS_DIR = "../recordings"
record_buffer = []        # list of numpy arrays (block data)
record_file = None

# ========== Filter Design ==========
def design_hp(cutoff, fs, order=2):
    return butter(order, cutoff, btype='high', fs=fs, output='sos')

def design_lp(cutoff, fs, order=4):
    return butter(order, cutoff, btype='low', fs=fs, output='sos')

def design_high_shelf(gain_db, freq, fs, q=0.707):
    if gain_db <= 0:
        return ([1.0], [1.0])
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * q)
    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    return ([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])

# Main filters
hp_sos = design_hp(BASS_CUT_HZ, SAMPLE_RATE, order=2)
hp_zi = np.zeros((hp_sos.shape[0], 2))
lp_sos = design_lp(TREBLE_CUT_HZ, SAMPLE_RATE, order=4)
lp_zi = np.zeros((lp_sos.shape[0], 2))

# Acoustic pre‑shaping low‑pass (6 kHz)
ae_lp_sos = design_lp(6000, SAMPLE_RATE, order=2)
ae_lp_zi = np.zeros((ae_lp_sos.shape[0], 2))

# Inter‑stage filters (multi‑stage distortion)
stage_lp_sos = design_lp(6000, SAMPLE_RATE, order=2)
stage1_zi = np.zeros((stage_lp_sos.shape[0], 2))
stage2_zi = np.zeros((stage_lp_sos.shape[0], 2))

# Extra low‑pass after cabinet (3.5 kHz, 2‑pole)
post_cab_lp_sos = design_lp(3500, SAMPLE_RATE, order=2)
post_cab_lp_zi = np.zeros((post_cab_lp_sos.shape[0], 2))

# Presence (high‑shelf at 3.5 kHz)
presence_b = [1.0]
presence_a = [1.0]
presence_state = np.zeros(2)

def update_presence_filter():
    global presence_b, presence_a, presence_state
    gain_db = PRESENCE * 12.0
    presence_b, presence_a = design_high_shelf(gain_db, freq=3500, fs=SAMPLE_RATE, q=0.707)
    presence_state = np.zeros(len(presence_a)-1)

update_presence_filter()

def apply_presence(x):
    if PRESENCE <= 0:
        return x
    global presence_state
    y, presence_state = lfilter(presence_b, presence_a, x, zi=presence_state)
    return y

# Notch filters for acoustic body
def init_notch_filters(freqs, Q, fs):
    filters = []
    for f in freqs:
        b, a = iirnotch(f, Q, fs)
        state = np.zeros(max(len(b), len(a)) - 1)
        filters.append((b, a, state))
    return filters

NOTCH_FREQS = [180, 350, 700]
NOTCH_Q = 4.0
notch_filters = init_notch_filters(NOTCH_FREQS, NOTCH_Q, SAMPLE_RATE)

def apply_notch_filters(x):
    for i, (b, a, state) in enumerate(notch_filters):
        y, new_state = lfilter(b, a, x, zi=state)
        notch_filters[i] = (b, a, new_state)
        x = y
    return x

# Peaking EQ (acoustic mid boost)
ae_eq_state = np.zeros(2)

def peaking_eq(x, freq, gain_db, q, state):
    w0 = 2.0 * np.pi * freq / SAMPLE_RATE
    alpha = np.sin(w0) / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = [b0/a0, b1/a0, b2/a0]
    a = [1.0, a1/a0, a2/a0]
    y, new_state = lfilter(b, a, x, zi=state)
    return y, new_state

# ========== Cabinet Impulse Response (Stateful Convolution) ==========
CAB_IR = None
cab_delay_line = None

def load_cabinet_ir(path="cabinet.wav", max_len=512):
    global CAB_IR, cab_delay_line
    if HAS_SOUNDFILE and os.path.exists(path):
        try:
            ir, sr = sf.read(path)
            if sr != SAMPLE_RATE:
                print(f"ERROR: Cabinet IR sample rate {sr} != {SAMPLE_RATE}. Using synthetic.")
                raise ValueError
            if ir.ndim > 1:
                ir = ir.mean(axis=1)
            ir = ir.astype(np.float32)
            ir /= np.max(np.abs(ir))
            if len(ir) > max_len:
                ir = ir[:max_len]
            elif len(ir) < max_len:
                ir = np.pad(ir, (0, max_len - len(ir)), 'constant')
            CAB_IR = ir
            cab_delay_line = np.zeros(len(CAB_IR) - 1)
            print(f"Loaded cabinet IR from {path} ({len(CAB_IR)} taps)")
            return
        except:
            pass
    # Synthetic IR: simple resonant low‑pass
    b, a = butter(2, 4000, btype='low', fs=SAMPLE_RATE, output='ba')
    ir = np.zeros(max_len)
    ir[0] = 1.0
    for i in range(1, max_len):
        ir[i] = b[0]*ir[i-1] + b[1]*(ir[i-2] if i>=2 else 0) - a[1]*ir[i-1] - a[2]*ir[i-2]
    ir = ir / np.max(np.abs(ir))
    t = np.arange(max_len) / SAMPLE_RATE
    comb = 0.5 + 0.5 * np.cos(2 * np.pi * 150 * t)
    ir = ir * comb
    ir = ir / np.sum(np.abs(ir))
    CAB_IR = ir.astype(np.float32)
    cab_delay_line = np.zeros(len(CAB_IR) - 1)
    print("Using synthetic cabinet IR (place cabinet.wav for real IR)")

def apply_cabinet(x):
    global cab_delay_line
    full = np.concatenate((cab_delay_line, x))
    conv = np.convolve(full, CAB_IR, mode='full')
    out = conv[:BLOCK_SIZE]
    cab_delay_line = full[-(len(CAB_IR)-1):]
    return out

# ========== Multi‑Stage Distortion ==========
def multistage_distortion(x, drive, makeup):
    y = np.tanh(x * drive * 0.7)
    global stage1_zi
    y, stage1_zi = sosfilt(stage_lp_sos, y, zi=stage1_zi)
    y = np.tanh(y * 2.2)
    global stage2_zi
    y, stage2_zi = sosfilt(stage_lp_sos, y, zi=stage2_zi)
    y = np.tanh(y * 1.8)
    return y * makeup

# ========== Noise Gate with Hysteresis ==========
def update_gate_coefficients():
    global attack_coef, release_coef
    attack_coef = np.exp(-1.0 / (NOISE_GATE_ATTACK_MS * 0.001 * SAMPLE_RATE))
    release_coef = np.exp(-1.0 / (NOISE_GATE_RELEASE_MS * 0.001 * SAMPLE_RATE))

update_gate_coefficients()

def apply_noise_gate(x):
    global gate_gain, gate_state
    rms = np.sqrt(np.mean(x**2) + 1e-10)
    # Hysteresis logic
    if gate_state == 'closed':
        if rms > NOISE_GATE_OPEN_THRESHOLD:
            gate_state = 'open'
            target = 1.0
        else:
            target = 0.0
    else:  # 'open'
        if rms < NOISE_GATE_CLOSE_THRESHOLD:
            gate_state = 'closed'
            target = 0.0
        else:
            target = 1.0
    # Smooth gain
    if target > gate_gain:
        gate_gain = attack_coef * gate_gain + (1 - attack_coef) * target
    else:
        gate_gain = release_coef * gate_gain + (1 - release_coef) * target
    return x * gate_gain

# ========== Compressor (Fixed Logic) ==========
def update_comp_coefficients():
    global comp_attack_coef, comp_release_coef
    comp_attack_coef = np.exp(-1.0 / (COMP_ATTACK_MS * 0.001 * SAMPLE_RATE))
    comp_release_coef = np.exp(-1.0 / (COMP_RELEASE_MS * 0.001 * SAMPLE_RATE))

update_comp_coefficients()

def apply_compressor(x):
    global comp_gain, comp_state
    peak = np.max(np.abs(x))
    # Smooth envelope
    if peak > comp_state:
        comp_state = comp_attack_coef * comp_state + (1 - comp_attack_coef) * peak
    else:
        comp_state = comp_release_coef * comp_state + (1 - comp_release_coef) * peak
    db = 20 * np.log10(comp_state + 1e-10)
    if db > COMP_THRESHOLD_DB:
        reduction_db = (db - COMP_THRESHOLD_DB) * (1 - 1/COMP_RATIO)
        target_gain = 10 ** (-reduction_db / 20)
    else:
        target_gain = 1.0
    # Attack when gain reduces (target_gain < comp_gain)
    if target_gain < comp_gain:
        comp_gain = comp_attack_coef * comp_gain + (1 - comp_attack_coef) * target_gain
    else:
        comp_gain = comp_release_coef * comp_gain + (1 - comp_release_coef) * target_gain
    return x * comp_gain

# ========== Output Recording ==========
def start_recording():
    global RECORDING, record_buffer
    if RECORDING:
        print("Already recording. Stop first.")
        return
    RECORDING = True
    record_buffer = []
    print("Recording started... Press 'r' again to stop and save.")

def stop_recording():
    global RECORDING, record_buffer
    if not RECORDING:
        print("Not recording.")
        return
    if not record_buffer:
        print("No audio captured.")
        RECORDING = False
        return
    # Concatenate all recorded blocks
    audio = np.concatenate(record_buffer, axis=0)
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"guitar_{timestamp}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    if HAS_SOUNDFILE:
        sf.write(filepath, audio, SAMPLE_RATE)
        print(f"Recording saved to {filepath}")
    else:
        print("soundfile not installed, cannot save WAV.")
    RECORDING = False
    record_buffer = []

# ========== Input Meter ==========
input_peak = 0.0

# ========== Audio Callback ==========
def callback(indata, outdata, frames, time, status):
    global hp_zi, lp_zi, ae_lp_zi, input_peak, ae_eq_state, post_cab_lp_zi
    global RECORDING, record_buffer
    if status:
        print(f"Status: {status}", flush=True)

    x = indata[:, 0].copy() if indata.shape[1] >= 1 else np.zeros(frames)

    # Input peak meter
    peak = np.max(np.abs(x))
    input_peak = max(peak, input_peak * 0.999)

    # Clean gain
    x = x * GAIN

    # Noise gate (first)
    if NOISE_GATE_ENABLED:
        x = apply_noise_gate(x)

    # Compressor (before distortion)
    if COMPRESSOR_ENABLED:
        x = apply_compressor(x)

    # Acoustic pre‑shaping
    if ACOUSTIC_TO_ELECTRIC:
        x = apply_notch_filters(x)
        x, ae_lp_zi = sosfilt(ae_lp_sos, x, zi=ae_lp_zi)
        x, ae_eq_state = peaking_eq(x, 1200.0, 5.0, 0.8, ae_eq_state)

    # Distortion chain
    dry = indata[:, 0].copy() if CLEAN_BLEND > 0 else None
    if DISTORTION_ENABLED:
        x, hp_zi = sosfilt(hp_sos, x, zi=hp_zi)
        x = multistage_distortion(x, DRIVE, MAKEUP_GAIN)
        x, lp_zi = sosfilt(lp_sos, x, zi=lp_zi)
        if CAB_IR is not None:
            x = apply_cabinet(x)
        x, post_cab_lp_zi = sosfilt(post_cab_lp_sos, x, zi=post_cab_lp_zi)
        x = apply_presence(x)

    # Clean blend
    if CLEAN_BLEND > 0 and dry is not None:
        dry = dry * GAIN * CLEAN_BLEND
        x = x * (1 - CLEAN_BLEND) + dry

    # Master volume and safety clip
    x = x * MASTER_VOLUME
    x = np.clip(x, -0.99, 0.99)

    outdata[:] = x.reshape(-1, 1)

    # Recording if enabled
    if RECORDING:
        record_buffer.append(x.copy())

# ========== Keyboard Listener ==========
def input_listener():
    global DISTORTION_ENABLED, DRIVE, BASS_CUT_HZ, TREBLE_CUT_HZ, PRESENCE
    global GAIN, MASTER_VOLUME, MAKEUP_GAIN, CLEAN_BLEND, ACOUSTIC_TO_ELECTRIC
    global hp_sos, lp_sos, hp_zi, lp_zi, stage1_zi, stage2_zi, ae_lp_zi
    global notch_filters, ae_eq_state, presence_state, NOISE_GATE_ENABLED
    global NOISE_GATE_OPEN_THRESHOLD, NOISE_GATE_CLOSE_THRESHOLD
    global COMPRESSOR_ENABLED, COMP_THRESHOLD_DB

    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        if cmd == 'd':
            DISTORTION_ENABLED = not DISTORTION_ENABLED
            print(f"Overdrive {'ON' if DISTORTION_ENABLED else 'OFF'}")
        elif cmd == 'ae':
            ACOUSTIC_TO_ELECTRIC = not ACOUSTIC_TO_ELECTRIC
            print(f"Acoustic→Electric {'ON' if ACOUSTIC_TO_ELECTRIC else 'OFF'}")
        elif cmd == 'drive' and len(parts) > 1:
            DRIVE = max(0.5, min(20.0, float(parts[1])))
            print(f"Drive = {DRIVE:.2f}")
        elif cmd == 'bass' and len(parts) > 1:
            BASS_CUT_HZ = max(20.0, min(200.0, float(parts[1])))
            hp_sos = design_hp(BASS_CUT_HZ, SAMPLE_RATE, order=2)
            hp_zi = np.zeros((hp_sos.shape[0], 2))
            print(f"Bass cut = {BASS_CUT_HZ:.0f} Hz")
        elif cmd == 'treble' and len(parts) > 1:
            TREBLE_CUT_HZ = max(2000.0, min(10000.0, float(parts[1])))
            lp_sos = design_lp(TREBLE_CUT_HZ, SAMPLE_RATE, order=4)
            lp_zi = np.zeros((lp_sos.shape[0], 2))
            print(f"Treble cut = {TREBLE_CUT_HZ:.0f} Hz")
        elif cmd == 'presence' and len(parts) > 1:
            PRESENCE = max(0.0, min(1.0, float(parts[1])))
            update_presence_filter()
            print(f"Presence = {PRESENCE:.2f}")
        elif cmd == 'gain' and len(parts) > 1:
            GAIN = max(0.1, min(1.0, float(parts[1])))
            print(f"Clean gain = {GAIN:.2f}")
        elif cmd == 'vol' and len(parts) > 1:
            MASTER_VOLUME = max(0.1, min(2.0, float(parts[1])))
            print(f"Master volume = {MASTER_VOLUME:.2f}")
        elif cmd == 'makeup' and len(parts) > 1:
            MAKEUP_GAIN = max(1.0, min(3.0, float(parts[1])))
            print(f"Make‑up gain = {MAKEUP_GAIN:.2f}")
        elif cmd == 'blend' and len(parts) > 1:
            CLEAN_BLEND = max(0.0, min(1.0, float(parts[1])))
            print(f"Clean blend = {CLEAN_BLEND:.2f}")
        elif cmd == 'notch' and len(parts) > 1:
            try:
                freq = float(parts[1])
                q = float(parts[2]) if len(parts) > 2 else 4.0
                if freq <= 0:
                    notch_filters = []
                    print("Notch filters disabled")
                else:
                    notch_filters = init_notch_filters([freq], q, SAMPLE_RATE)
                    print(f"Notch filter: {freq} Hz, Q={q}")
            except:
                print("Usage: notch freq [Q]")
        elif cmd == 'gate' and len(parts) > 1:
            try:
                val = float(parts[1])
                NOISE_GATE_OPEN_THRESHOLD = max(0.0, min(0.5, val))
                # Set close threshold to half of open by default
                NOISE_GATE_CLOSE_THRESHOLD = NOISE_GATE_OPEN_THRESHOLD * 0.5
                print(f"Gate open={NOISE_GATE_OPEN_THRESHOLD:.4f}, close={NOISE_GATE_CLOSE_THRESHOLD:.4f}")
            except:
                print("Usage: gate 0.008")
        elif cmd == 'gate_off':
            NOISE_GATE_ENABLED = False
            print("Noise gate OFF")
        elif cmd == 'gate_on':
            NOISE_GATE_ENABLED = True
            print("Noise gate ON")
        elif cmd == 'comp' and len(parts) > 1:
            try:
                val = float(parts[1])
                COMP_THRESHOLD_DB = max(-40.0, min(0.0, val))
                print(f"Compressor threshold = {COMP_THRESHOLD_DB:.1f} dB")
            except:
                print("Usage: comp -18")
        elif cmd == 'comp_off':
            COMPRESSOR_ENABLED = False
            print("Compressor OFF")
        elif cmd == 'comp_on':
            COMPRESSOR_ENABLED = True
            print("Compressor ON")
        elif cmd == 'loadir':
            load_cabinet_ir()
        elif cmd == 'r':
            if not RECORDING:
                start_recording()
            else:
                stop_recording()
        elif cmd == 'q':
            print("Quitting...")
            sys.exit(0)
        else:
            print("Commands: d, ae, drive, bass, treble, presence, gain, vol, makeup, blend, notch, gate, gate_off, gate_on, comp, comp_off, comp_on, loadir, r, q")

def main():
    print("Available audio devices:")
    print(sd.query_devices())
    print("\nOpening stream...")
    load_cabinet_ir()

    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()

    try:
        with sd.Stream(
            device='pipewire',   # or 'default'
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            callback=callback,
            latency=LATENCY
        ) as stream:
            print(f"✅ Stream active: {int(stream.samplerate)} Hz, {stream.blocksize} samples/block")
            print("Ultimate Guitar Pedalboard – with hysteresis gate and recording")
            print("   • Noise gate: hysteresis (open/close thresholds)")
            print("   • Press 'r' to start/stop recording output to WAV")
            print("Commands: d, ae, drive, bass, treble, presence, gain, vol, makeup, blend, notch, gate, comp, loadir, r, q\n")
            while True:
                time.sleep(0.5)
                print(f"Input peak: {input_peak:.3f} (aim 0.3-0.8) | Gate state: {'OPEN' if gate_gain>0.5 else 'CLOS'}", end='\r')
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()