#!/usr/bin/env python3
"""
Freeverb-style Algorithmic Reverb
==================================
A classic Schroeder reverb implementation optimized for real-time
block-based audio processing (guitar pedalboard).

Architecture:
    8 parallel comb filters → sum → 4 cascaded all-pass filters → output
    Each comb has a one-pole low-pass in the feedback loop (damping).

All delay lines use circular buffers so state persists across callbacks.
"""

import numpy as np

# Freeverb delay lengths at 44100 Hz (prime numbers to reduce beating)
COMB_DELAYS_44100 = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
ALLPASS_DELAYS_44100 = [225, 556, 441, 341]


class CombFilter:
    """Feedback comb filter with damping (one-pole LPF in feedback loop)."""
    def __init__(self, delay_samples, sample_rate=44100):
        self.delay = int(delay_samples)
        self.buffer = np.zeros(self.delay)
        self.idx = 0
        self.filter_state = 0.0  # one-pole LPF state
        self.damping = 0.5
        self.feedback = 0.5
        self.sample_rate = sample_rate

    def set_params(self, feedback, damping):
        """feedback: 0.0-1.0 (room size), damping: 0.0-1.0 (high freq loss)"""
        self.feedback = feedback
        self.damping = damping

    def process(self, x):
        """Process a block of samples. Returns same-length array."""
        out = np.empty_like(x)
        for i in range(len(x)):
            # Read delayed sample
            delayed = self.buffer[self.idx]

            # Damping: one-pole LPF on feedback
            self.filter_state = delayed * (1.0 - self.damping) + self.filter_state * self.damping

            # Write new sample (input + filtered feedback)
            self.buffer[self.idx] = x[i] + self.filter_state * self.feedback

            # Output is the delayed sample (comb filter output)
            out[i] = delayed

            # Advance circular buffer
            self.idx = (self.idx + 1) % self.delay
        return out

    def reset(self):
        self.buffer.fill(0.0)
        self.idx = 0
        self.filter_state = 0.0


class AllPassFilter:
    """All-pass filter for diffusing the comb output into a smooth tail."""
    def __init__(self, delay_samples):
        self.delay = int(delay_samples)
        self.buffer = np.zeros(self.delay)
        self.idx = 0
        self.feedback = 0.5

    def set_feedback(self, feedback):
        self.feedback = feedback

    def process(self, x):
        out = np.empty_like(x)
        for i in range(len(x)):
            delayed = self.buffer[self.idx]
            # All-pass formula: output = delayed - feedback * input
            #                    buffer = input + feedback * delayed
            out[i] = delayed - self.feedback * x[i]
            self.buffer[self.idx] = x[i] + self.feedback * delayed
            self.idx = (self.idx + 1) % self.delay
        return out

    def reset(self):
        self.buffer.fill(0.0)
        self.idx = 0


class Freeverb:
    """
    Complete reverb processor.

    Parameters (all 0.0–1.0 unless noted):
        room_size   : scales comb filter feedback (0=small closet, 1=cathedral)
        damping     : high-frequency absorption (0=bright, 1=dark)
        wet         : wet/dry mix (0=dry only, 1=wet only)
        width       : stereo spread (mono for now, placeholder)
        pre_delay_ms: delay before reverb starts (0–100 ms)
    """
    def __init__(self, sample_rate=44100, block_size=512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.enabled = False

        # Scale delays to current sample rate
        scale = sample_rate / 44100.0
        comb_delays = [int(d * scale) for d in COMB_DELAYS_44100]
        ap_delays = [int(d * scale) for d in ALLPASS_DELAYS_44100]

        self.combs = [CombFilter(d, sample_rate) for d in comb_delays]
        self.allpasses = [AllPassFilter(d) for d in ap_delays]

        # Pre-delay buffer
        self.max_pre_delay_ms = 100.0
        self.pre_delay_samples = int(self.max_pre_delay_ms * 0.001 * sample_rate)
        self.pre_buffer = np.zeros(self.pre_delay_samples)
        self.pre_idx = 0

        # Default params
        self.room_size = 0.5
        self.damping = 0.5
        self.wet = 0.3
        self.width = 0.5
        self.pre_delay_ms = 0.0

        self._update_params()

    def _update_params(self):
        """Push parameter changes down to individual filters."""
        # Room size → comb feedback (0.28 to 0.7 range, classic Freeverb scaling)
        feedback = self.room_size * 0.28 + 0.7
        for c in self.combs:
            c.set_params(feedback, self.damping)

        # All-pass feedback is fixed at 0.5 for diffusion
        for ap in self.allpasses:
            ap.set_feedback(0.5)

        # Pre-delay
        self.pre_delay_samples = int(self.pre_delay_ms * 0.001 * self.sample_rate)

    def set_room_size(self, val):
        self.room_size = np.clip(val, 0.0, 1.0)
        self._update_params()

    def set_damping(self, val):
        self.damping = np.clip(val, 0.0, 1.0)
        self._update_params()

    def set_wet(self, val):
        self.wet = np.clip(val, 0.0, 1.0)

    def set_width(self, val):
        self.width = np.clip(val, 0.0, 1.0)

    def set_pre_delay(self, ms):
        self.pre_delay_ms = np.clip(ms, 0.0, 100.0)
        self._update_params()

    def process(self, x):
        if not self.enabled:
            return x

        # Ensure exact block size
        if len(x) < self.block_size:
            x = np.pad(x, (0, self.block_size - len(x)), "constant")
        elif len(x) > self.block_size:
            x = x[:self.block_size]

        # --- Pre-delay ---
        if self.pre_delay_samples > 0:
            delayed = np.empty_like(x)
            for i in range(len(x)):
                delayed[i] = self.pre_buffer[self.pre_idx]
                self.pre_buffer[self.pre_idx] = x[i]
                self.pre_idx = (self.pre_idx + 1) % len(self.pre_buffer)
            x_delayed = delayed
        else:
            x_delayed = x

        # --- Comb filters (parallel) ---
        comb_out = np.zeros(self.block_size)
        for c in self.combs:
            comb_out += c.process(x_delayed)
        comb_out *= 0.125  # normalize by 1/8

        # --- All-pass filters (cascade) ---
        ap_out = comb_out
        for ap in self.allpasses:
            ap_out = ap.process(ap_out)

        # --- Wet/dry mix ---
        wet_gain = self.wet
        dry_gain = 1.0 - self.wet
        out = x * dry_gain + ap_out * wet_gain

        return out

    def reset(self):
        """Clear all buffers (call when toggling on or changing sample rate)."""
        for c in self.combs:
            c.reset()
        for ap in self.allpasses:
            ap.reset()
        self.pre_buffer.fill(0.0)
        self.pre_idx = 0

    def status(self):
        return (
            f"Reverb: {'ON' if self.enabled else 'OFF'} | "
            f"room={self.room_size:.2f} | damp={self.damping:.2f} | "
            f"wet={self.wet:.2f} | pre={self.pre_delay_ms:.0f}ms"
        )