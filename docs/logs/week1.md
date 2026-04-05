# Week 1 - Software OFDM transmitter and receiver

## System configuration

The `OFDMConfig` dataclass defines all system parameters. Here are the key values and why each matters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_fft` | 64 | FFT size - determines the number of subcarriers |
| `n_cp` | 16 | Cyclic prefix length (25% of FFT) - must exceed channel delay spread |
| `n_data` | 48 | Data subcarriers carrying user information |
| `n_pilots` | 4 | Pilot subcarriers for channel estimation |
| `pilot_indices` | [11, 25, 39, 53] | Positions follow 802.11a placement (evenly spaced, ~14 bins apart) |
| `guard bands` | [0–5, 59–63] | 11 null subcarriers preventing spectral leakage into adjacent channels |
| `DC null` | bin 32 | Nulled to avoid DC offset from direct-conversion receivers |
| `mod_order` | 4 (QPSK) | 2 bits per symbol; extensible to 16-QAM (4 bits per symbol) |
| `n_symbols` | 20 | OFDM symbols per frame |
| `bits_per_frame` | 1920 | 48 subcarriers × 2 bits × 20 symbols |

**Question I had:** At what actual frequency do the pilot subcarriers sit?

**Answer:** The pilot positions [11, 25, 39, 53] are FFT bin indices, not absolute frequencies. Their physical frequency depends on the sample rate: at 1 MSPS, each bin is spaced 1e6/64 = 15,625 Hz apart, so pilot at bin 11 sits at 11 × 15,625 Hz = 171.875 kHz relative to the center frequency. The specific bin positions are defined by the 802.11a standard and are chosen to be roughly evenly distributed across the active subcarriers for good interpolation coverage.


## Subcarrier mapping

The 64 FFT bins are divided into four categories:

- **Data subcarriers (48):** Carry QPSK symbols, this is where the user's bits live. Bins 6–58, excluding DC and pilots.
- **Pilot subcarriers (4):** Carry known value `1+0j`. Used by the receiver to estimate the channel. The receiver compares what it received at these bins to what it knows was sent, computing `H = Y/X`.
- **DC null (1):** Bin 32, always zero. Real SDR hardware (like the PlutoSDR's AD9363) has a DC offset from the mixer that would corrupt data placed here.
- **Guard bands (11):** Bins 0–5 and 59–63, always zero. Prevent spectral leakage into adjacent frequency channels. In the code, these stay at zero because `freq_bins` is initialized with `np.zeros()` and only data/pilot indices are written.


## QPSK and 16-QAM normalization

Constellation points are scaled so the average symbol energy equals 1. This normalization is important because it makes SNR calculations and BER comparisons meaningful, without it, different modulation schemes would have different transmit powers.

**QPSK normalization - divide by √2:**

Unnormalized QPSK points are {1+j, 1−j, −1+j, −1−j}. Each has energy |1±j|² = 1² + 1² = 2, so average symbol energy E_s = 2. Dividing by √E_s = √2 gives unit average energy.

**16-QAM normalization - divide by √10:**

For Gray-coded 16-QAM, I and Q each take values {−3, −1, +1, +3}. Average energy per axis: E[I²] = (9+1+1+9)/4 = 5. Total average symbol energy: E_s = E[I²] + E[Q²] = 5 + 5 = 10. Dividing by √10 gives unit average energy.

**Bug I encountered:** My initial QPSK mapper used a Gray coding table where bit 0 controlled the real sign and bit 1 controlled "which diagonal," but the demapper assumed bit 0 = sign of real, bit 1 = sign of imaginary. This mismatch caused BER to sit at ~0.5 regardless of SNR - the classic symptom of a modulation/demodulation inconsistency. Fix: ensure both mapper and demapper use the same convention: `bit 0 → real sign, bit 1 → imaginary sign`.


## Cyclic prefix

The cyclic prefix (CP) is a copy of the last `n_cp = 16` samples of the OFDM symbol, prepended to the front. This converts the channel's linear convolution into circular convolution, which is what the FFT-based receiver assumes.

**Why it works:** When a delayed multipath copy reaches back into the CP region, it finds samples identical to the symbol's tail - as if the signal wrapped around. From the receiver's perspective (after discarding the CP), the channel performed circular convolution, and the circular convolution theorem guarantees Y[k] = X[k] × H[k] - simple pointwise multiplication in the frequency domain.

**In the code:**
```python
# TX: prepend last n_cp samples
ofdm_symbol = np.concatenate([time_samples[-n_cp:], time_samples])

# RX: discard first n_cp samples
ofdm_no_cp = ofdm_with_cp[n_cp:]
```

**Guard interval in 802.11a context:** The 802.11a standard uses 800 ns guard interval within a 4.0 μs total symbol duration, designed to handle typical indoor multipath delay spreads of 50–200 ns. In our simulation at 1 MSPS, the CP of 16 samples corresponds to 16 μs - much longer than typical indoor delays, providing a comfortable margin.


## Frame synchronization - Zadoff-Chu preamble

The preamble is a known sequence prepended to each frame that lets the receiver determine exactly where the frame starts. We use a Zadoff-Chu (ZC) sequence:

$$z[n] = e^{-j\pi r \cdot n(n+1)/N}$$

where n is the sample index, r = 25 is the root index, and N = 63 is the sequence length.

**Why Zadoff-Chu specifically:**

1. **Constant amplitude** - |z[n]| = 1 for all n, so no DAC clipping risk and favorable PAPR.
2. **Ideal autocorrelation** - periodic autocorrelation is a perfect impulse (zero at all non-zero lags), meaning no false detection peaks.
3. **Noise resilience** - cross-correlation provides ~18 dB processing gain (10×log10(63)), allowing detection even at negative SNR values.

**In the code:** `detect_preamble()` cross-correlates the received signal with the stored ZC preamble. The peak location marks the frame start, and data begins at `peak_idx + len(preamble)`.

**Connection to real systems:** LTE and 5G NR use Zadoff-Chu sequences for their Primary Synchronization Signal (PSS) - the exact same concept for the exact same reason.


## Channel estimation and equalization

**The concept:** Pilots are known symbols at known subcarrier positions. By comparing what was sent vs. what was received at pilot bins, the receiver estimates the channel's frequency response H[k].

**The process:**
1. Extract received pilot values: `rx_pilots = freq_bins[pilot_indices]`
2. Compute H at pilot frequencies: `h_pilots = rx_pilots / tx_pilots` (least-squares estimate)
3. Interpolate H to all data subcarriers using `np.interp` (separately for magnitude and phase)
4. Equalize: `rx_data = freq_bins[data_indices] / h_interp`

**Interpolation example:** For a subcarrier at bin 18 (between pilots at bins 11 and 25):

$$H[18] = H[11] \cdot \frac{25 - 18}{25 - 11} + H[25] \cdot \frac{18 - 11}{25 - 11}$$

This is a weighted average where closer pilots contribute more - standard linear interpolation.

**Limitation I observed:** The measured BER curve sits ~1.5–2 dB above the theoretical QPSK curve. This gap comes from: (a) noisy pilot estimates corrupting the channel estimate, (b) linear interpolation missing rapid channel variations between pilots, and (c) finite simulation length (only 38,400 bits per SNR point).


## AWGN channel model

The `awgn_channel()` function simulates thermal noise - the only channel impairment in Phase 1. The key computation:

```python
noise_power = sig_power / (10 ** (snr_db / 10))
sigma = sqrt(noise_power / 2)
noise = sigma * (randn(N) + 1j * randn(N))
```

**Why divide by 2:** The total noise power splits equally between I and Q components. Each gets variance = noise_power/2, so total power = noise_power.

**Important realization:** The AWGN function is purely for simulation - in Phase 2 with real PlutoSDRs, the physical channel provides the noise. The function exists to validate the receiver before hardware is available.


## Results

| SNR (dB) | BER (measured) | Errors / Total |
|----------|---------------|----------------|
| 0 | 2.19e-01 | 8401 / 38400 |
| 2 | 1.50e-01 | 5743 / 38400 |
| 4 | 9.20e-02 | 3534 / 38400 |
| 6 | 3.88e-02 | 1491 / 38400 |
| 8 | 1.40e-02 | 538 / 38400 |
| 10 | 2.89e-03 | 111 / 38400 |
| 12 | 1.56e-04 | 6 / 38400 |
| 14+ | 0 | 0 / 38400 |

The BER curve follows the expected waterfall shape, matching QPSK theory within ~2 dB. Zero errors above 14 dB confirms the receiver is working correctly.


## Files produced
- `constellation_15dB.png` - QPSK constellation at 15 dB SNR
- `constellations_multi_snr.png` - constellation comparison at 5, 10, 15, 20 dB
- `ofdm_spectrum.png` - transmit power spectral density
- `ber_curve_qpsk.png` - measured BER vs theoretical QPSK