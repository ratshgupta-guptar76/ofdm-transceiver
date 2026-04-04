"""
OFDM Transceiver - Phase 1 Starter Code
========================================
A complete software OFDM TX/RX chain with AWGN simulation.
This code is a starting point for implementing an OFDM transceiver.
Built as a pre-research skill builder for Optical Satelite comms work.

System parameters:
    - 64-point FFT (matching PlutoSDR pipeline for Phase 2)
    - 48 data subcarriers + 4 pilots + DC null + guard bands
    - QPSK modulation (extensible to 16-QAM)
    - Cyclic prefix length of 16 samples (25% of FFT size)
    - Zadoff-Chu preamble for frame synchronization
    - Least-squares channel estimation via pilot subcarriers

Author: Ratish Gupta
Course Context: McMaster University ECE - EE 3TR4, CE 3DY4, EE 3SM4
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ──────────────────────────────────
# System Configuration
# ──────────────────────────────────

@dataclass
class OFDMConfig:
    """All parameters for the OFDM system."""
    n_fft: int = 64                 # FFT size
    n_cp: int = 16                  # Cyclic prefix length
    n_data: int = 48                # Number of data subcarriers
    n_pilots: int = 4               # Number of pilot subcarriers
    pilot_value: complex = 1 + 0j   # Known pilot symbol (BPSK, +1)
    n_symbols: int = 20             # OFDM symbols per frame
    mod_order: int = 4              # QPSK = 4, 16-QAM = 16
    zc_root: int = 25               # Zadoff-Chu root index for preamble
    zc_length: int = 63             # Zadoff-Chu sequence length (odd < n_fft)

    @property
    def bits_per_symbol(self):
        """
        Returns the number of bits that can be decoded from each symbol
        """
        return int(np.log2(self.mod_order))

    @property
    def pilot_indices(self):
        """ Recover the pilot indices which matches the `802.11a` style of placement\n

            The array added to is a standard for `802.11a`, and the values determine
            the position of the pilots as an offset from the DC null sub-carrier.
            `[-21, -7, 7, 21]`
        """
        return np.array([-21, -7, 7, 21]) + self.n_fft // 2

    @property
    def data_indices(self):
        """
            Returns the Data Subcarrier positions i.e. everything except of pilots, DC
            and guards.
        """
        usable = set(range(6, 59))
        usable.discard(32)
        return np.array(sorted(usable - set(self.pilot_indices)))

    @property
    def bits_per_ofdm_symbol(self):
        return self.n_data * self.bits_per_symbol
    
    @property
    def bits_per_frame(self):
        return self.n_symbols * self.bits_per_ofdm_symbol
    

"""
------- Transmitter -------
"""
# Modulation - Transmitter
def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """
    Map the bit pairs to QPSK
    Gray Coded\n
    | 00 | 01 | 11 | 10 |
    |----|----|----|----|
    | `1 + 1j` | `1 - 1j` | `-1 - 1j`  | `-1 + 1j`  |

    Normalized to unit average power
    """
    assert len(bits) % 2 == 0, "QPSK needs even number of bits"

    pairs = bits.reshape(-1, 2)

    greyCode_map = {
        (0, 0): (1 + 1j),
        (0, 1): (1 - 1j),
        (1, 0): (-1 + 1j),
        (1, 1): (-1 - 1j),
    }

    symbols = np.array([greyCode_map[tuple(p)] for p in pairs]) / np.sqrt(2)
    return symbols

def qam16_map(bits: np.ndarray) -> np.ndarray:
    """
    Map the 4 bits to 16-QAM
    Gray Coding is used on each sample

    Normalized to unit average power

    Uses a 2-bit In-Phase and 2-bit Quadrature approach 
    to map the bits to the respective symbols
    """
    assert len(bits) % 4 == 0, "16-QAM needs groups of 4 bits"
    
    quads = bits.reshape(-1, 4)

    greyCode_map = {
        (0, 0): -3,
        (0, 1): -1,
        (1, 0):  1,
        (1, 1):  3,
    }

    symbols = np.array([
        greyCode_map[tuple(q[:2])] + 1j * greyCode_map[tuple(q[2:])]
        for q in quads
    ]) / np.sqrt(10)
    return symbols

# Generate Zadoff-Chu
def generate_preamble(cfg: OFDMConfig) -> np.ndarray:
    """
    Generate a Zadoff-Chu sequence for frame synchronization.
    ZC sequences have constant amplitude and ideal autocorrelation
    properties, this is perfect for preamble detection in noisy channels

    This is the same family of sequences used in LTE/5G NR for primary
    synchronization signals (PSS)

    The Zadoff-Chu formula is as follows, `exp[-j * 25 pi * n(n+1)]`
    """
    n = np.arange(cfg.zc_length)
    zc = np.exp(-1j * np.pi * cfg.zc_root * n * (n+1) / cfg.zc_length)

    # Zero-pad to FFT size and create time-domain preamble
    freq_domain = np.zeros(cfg.n_fft, dtype=complex)
    freq_domain[1:cfg.zc_length+1] = zc
    time_domain = np.fft.ifft(freq_domain)
    
    # Add cyclic prefix
    preamble = np.concatenate([time_domain[-cfg.n_cp:], time_domain])
    return preamble

# OFDM Transmitter
def ofdm_transmit(bits: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """
    Full OFDM transmitter chain:

     bits -> modulation -> subcarrier mapping -> iFFT -> add CP -> frame assmebly

    Returns the complete time-domain frame (preamble + OFDM symbols)
    """
    assert len(bits) == cfg.bits_per_frame, \
        f"Expected {cfg.bits_per_frame} bits, got {len(bits)}"
    
    if cfg.mod_order == 4:
        modulate = qpsk_map
    elif cfg.mod_order == 16:
        modulate = qam16_map
    else:
        raise ValueError(f"Unsupported modulation order: {cfg.mod_order}")
    
    # Generate preamble = generate_preamble(cfg)
    preamble = generate_preamble(cfg)
    frame_samples = [preamble]

    bits_per_ofdm_sym = cfg.bits_per_ofdm_symbol

    for i in range(cfg.n_symbols):
        # Extract bits for curr OFDM symbol
        sym_bits = bits[i*bits_per_ofdm_sym : (i+1)*bits_per_ofdm_sym]

        # Modulate to constellation points
        data_symbols = modulate(sym_bits)

        freq_bins = np.zeros(cfg.n_fft, dtype=complex)
        # Assign Freq bins for data subcarriers
        freq_bins[cfg.data_indices] = data_symbols
        # Assign Freq bins for pilot subcarriers
        freq_bins[cfg.pilot_indices] = cfg.pilot_value

        # IFFT -> time domain
        time_samples = np.fft.ifft(freq_bins)

        # Cyclic prefix
        ofdm_symbol = np.concatenate([
            time_samples[-cfg.n_cp:], # last n_cp samples to the front
            time_samples
        ]) 

        frame_samples.append(ofdm_symbol)

    return np.concatenate(frame_samples)


"""
------- Receiver -------
"""
# Demodulation - Receiver
def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """
    Hard Decisions Demapping for 16-QAM
    """
    bits = np.zeros(len(symbols) * 2, dtype=int)

    bits[0::2] = (symbols.real < 0).astype(int)
    bits[1::2] = (symbols.imag < 0).astype(int)
    return bits

def qam16_demap(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 4, dtype=int)

    for i, q in enumerate(symbols):
        real = q.real * np.sqrt(10)
        imag = q.imag * np.sqrt(10)
        
        for val, bit, idx in [(real, bits, 4*i), (imag, bits, 4*i+2)]:
            if val < -2:
                bit[idx], bit[idx+1] = 0, 0
            elif val < -1:
                bit[idx], bit[idx+1] = 0, 1
            elif val < 1:
                bit[idx], bit[idx+1] = 1, 1
            else:
                bit[idx], bit[idx+1] = 1, 0

    return bits

# Decode/Detect Zadoff-Chu
def detect_preamble(signal: np.ndarray, cfg: OFDMConfig) -> int:
    """
    Compares signal samples to known preamble and starts from there.\n
    Returns the index at which the OFDM symbols start.
    """
    zadoff_chu_preamble = generate_preamble(cfg)

    corr = np.abs(np.correlate(signal, zadoff_chu_preamble, mode="valid"))
    # I didn't do this, I don't understand this
    # corr /= (np.sqrt(np.sum(zadoff_chu_preamble**2) * np.convolve(np.abs(signal**2),
    #                                                               np.ones(len(zadoff_chu_preamble)),
    #                                                               mode="valid"
    #                                                               )
    #                  )
    #          )
    
    peak_idx = np.argmax(corr)
    frame_start = peak_idx + len(zadoff_chu_preamble)
    return frame_start

# OFDM Receiver
def ofdm_receive(rx: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
    """
    Decodes the bits from the received signal which contains
    awgn and multipath channels
    """
    # Find frame start
    start_idx = detect_preamble(rx, cfg)
    rx = rx[start_idx:]

    # Choose demodulator
    if cfg.mod_order == 16:
        demodulate = qam16_demap
    elif cfg.mod_order == 4:
        demodulate = qpsk_demap
    else:
        raise ValueError(f"Unsupported modulation order: {cfg.mod_order}")
    
    raw_symbols = []
    recovered_bits = []

    for i in range(cfg.n_symbols):
        start = i*(cfg.n_cp + cfg.n_fft)
        end = start + (cfg.n_cp + cfg.n_fft)

        if end > len(rx):
            break
        
        ofdm_symbol = rx[start+cfg.n_cp : end]

        freq_bins = np.fft.fft(ofdm_symbol)

        y_pilots = freq_bins[cfg.pilot_indices]
        x_pilots = cfg.pilot_value * np.ones(cfg.n_pilots)
        h_pilots = y_pilots / x_pilots

        h_interp = np.interp(
            cfg.data_indices,
            cfg.pilot_indices,
            np.abs(h_pilots)
        ) * np.exp(1j * np.interp(
            cfg.data_indices,
            cfg.pilot_indices,
            np.angle(h_pilots)
        ))

        rx_data = freq_bins[cfg.data_indices] / h_interp
        raw_symbols.append(rx_data)

        sym_bits = demodulate(rx_data)
        recovered_bits.append(sym_bits)

    return np.concatenate(recovered_bits), np.concatenate(raw_symbols)


"""
Simulation Functions
"""
def AWGN(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Simulates the Thermal noise from the hardware front end
    using `Additive White Gaussian Noise`
    """
    sig_power = np.mean(pow(np.abs(signal), 2))
    snr = pow(10, snr_db / 10)

    noise_power = sig_power / snr
    a = np.sqrt(noise_power / 2) # Divide by 2 since power is equally distributed among I and Q
    real = np.random.randn(len(signal))
    imag = np.random.randn(len(signal))

    noise = a * (real + 1j * imag)

    return signal + noise   # Add the simulated noise to the signal

def multipath_channel(signal: np.ndarray, filter_taps: np.ndarray = None) -> np.ndarray:
    """
    Simulates multiple reflected paths i.e. reflected signals that
    may arrive late in time with a different amplitude and/or phase
    Implements a convolution in the time domain to simulate the 
    multiple paths (Low number of taps [2-5])
    """
    if filter_taps is None:
        filter_taps = np.array([1.0, 0.4 * np.exp(1j*0.5), 0.2 * np.exp(1j*1.3)])
    
    return np.convolve(signal, filter_taps, mode="same")

"""
Plotting Functions
"""
# ─────────────────────────────────────────────
# Performance measurement
# ─────────────────────────────────────────────

def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Compute Bit Error Rate."""
    min_len = min(len(tx_bits), len(rx_bits))
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    return errors / min_len


def theoretical_ber_qpsk(snr_db: np.ndarray) -> np.ndarray:
    """Theoretical BER for QPSK in AWGN (same as BPSK per-bit)."""
    from scipy.special import erfc
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_lin))


def theoretical_ber_16qam(snr_db: np.ndarray) -> np.ndarray:
    """Approximate theoretical BER for 16-QAM in AWGN."""
    from scipy.special import erfc
    snr_lin = 10 ** (snr_db / 10)
    return (3 / 8) * erfc(np.sqrt(snr_lin * 2 / 5))


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_constellation(rx_symbols: np.ndarray, title: str = "Received constellation"):
    """Plot received symbols on IQ plane."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(rx_symbols.real, rx_symbols.imag, s=4, alpha=0.5, c='#3498db')
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_ber_curve(snr_range: np.ndarray, ber_measured: np.ndarray,
                   mod_order: int = 4):
    """Plot measured BER vs theoretical."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Theoretical
    snr_fine = np.linspace(snr_range[0], snr_range[-1], 100)
    if mod_order == 4:
        ber_theory = theoretical_ber_qpsk(snr_fine)
        label_theory = "QPSK theoretical"
    else:
        ber_theory = theoretical_ber_16qam(snr_fine)
        label_theory = "16-QAM theoretical"

    ax.semilogy(snr_fine, ber_theory, 'k--', linewidth=1.5, label=label_theory)
    ax.semilogy(snr_range, ber_measured, 'o-', color='#e74c3c',
                markersize=6, linewidth=1.5, label="Measured (AWGN sim)")

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("BER vs SNR — OFDM System")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-5, 1])
    plt.tight_layout()
    return fig


def plot_spectrum(tx_signal: np.ndarray, fs: float = 1.0):
    """Plot the power spectral density of the transmitted signal."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_signal), 1/fs))
    psd = np.fft.fftshift(np.abs(np.fft.fft(tx_signal)) ** 2)
    psd_db = 10 * np.log10(psd / np.max(psd) + 1e-12)

    ax.plot(freqs, psd_db, linewidth=0.8, color='#2ecc71')
    ax.set_xlabel("Frequency (normalized)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("OFDM Transmit Spectrum")
    ax.set_ylim([-60, 5])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────

def run_ber_simulation(snr_range_db=None, n_trials=10, mod_order=4):
    """
    Run full BER simulation across SNR range.
    This is the main experiment — generates the BER curve for your report.
    """
    if snr_range_db is None:
        snr_range_db = np.arange(0, 22, 2)

    cfg = OFDMConfig(mod_order=mod_order)
    ber_results = []

    print(f"\n{'='*55}")
    print(f"  OFDM BER Simulation — {'QPSK' if mod_order == 4 else '16-QAM'}")
    print(f"  FFT size: {cfg.n_fft} | Data subcarriers: {cfg.n_data}")
    print(f"  Symbols/frame: {cfg.n_symbols} | Bits/frame: {cfg.bits_per_frame}")
    print(f"  Trials per SNR: {n_trials}")
    print(f"{'='*55}\n")

    for snr_db in snr_range_db:
        total_errors = 0
        total_bits = 0

        for trial in range(n_trials):
            # Generate random bits
            tx_bits = np.random.randint(0, 2, cfg.bits_per_frame)

            # Transmit
            tx_signal = ofdm_transmit(tx_bits, cfg)

            # Channel
            rx_signal = AWGN(tx_signal, snr_db)

            # Receive
            rx_bits, _ = ofdm_receive(rx_signal, cfg)

            # Count errors
            min_len = min(len(tx_bits), len(rx_bits))
            total_errors += np.sum(tx_bits[:min_len] != rx_bits[:min_len])
            total_bits += min_len

        ber = total_errors / total_bits if total_bits > 0 else 0
        ber_results.append(max(ber, 1e-6))  # Floor for log plot
        print(f"  SNR = {snr_db:5.1f} dB  |  BER = {ber:.2e}  |  Errors: {total_errors}/{total_bits}")

    return snr_range_db, np.array(ber_results)




def main():
    """Run the full Phase 1 demo."""
    cfg = OFDMConfig(mod_order=4)

    # ── Single-frame demo at specific SNR ──
    print("\n" + "─" * 55)
    print("  Single-frame demo at SNR = 15 dB")
    print("─" * 55)

    tx_bits = np.random.randint(0, 2, cfg.bits_per_frame)
    tx_signal = ofdm_transmit(tx_bits, cfg)
    rx_signal = AWGN(tx_signal, snr_db=15.0)
    rx_bits, rx_symbols = ofdm_receive(rx_signal, cfg)

    ber = compute_ber(tx_bits, rx_bits)
    print(f"  Transmitted: {cfg.bits_per_frame} bits")
    print(f"  Recovered:   {len(rx_bits)} bits")
    print(f"  BER:         {ber:.2e}")

    # Plot constellation
    fig1 = plot_constellation(rx_symbols, "QPSK constellation @ 15 dB SNR")
    fig1.savefig("constellation_15dB.png", dpi=150)
    print("  → Saved constellation_15dB.png")

    # Plot spectrum
    fig2 = plot_spectrum(tx_signal)
    fig2.savefig("ofdm_spectrum.png", dpi=150)
    print("  → Saved ofdm_spectrum.png")

    # ── BER curve simulation ──
    snr_range, ber_measured = run_ber_simulation(
        snr_range_db=np.arange(0, 22, 2),
        n_trials=20,
        mod_order=4
    )

    fig3 = plot_ber_curve(snr_range, ber_measured, mod_order=4)
    fig3.savefig("ber_curve_qpsk.png", dpi=150)
    print("\n  → Saved ber_curve_qpsk.png")

    # Show constellations at multiple SNR levels
    fig4, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, snr in zip(axes, [5, 10, 15, 20]):
        tx_bits_demo = np.random.randint(0, 2, cfg.bits_per_frame)
        tx_sig = ofdm_transmit(tx_bits_demo, cfg)
        rx_sig = AWGN(tx_sig, snr)
        _, rx_syms = ofdm_receive(rx_sig, cfg)
        ax.scatter(rx_syms.real, rx_syms.imag, s=3, alpha=0.4, c='#3498db')
        ax.set_title(f"SNR = {snr} dB", fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)

    fig4.suptitle("QPSK constellation at various SNR levels", fontsize=13)
    plt.tight_layout()
    fig4.savefig("constellations_multi_snr.png", dpi=150)
    print("  → Saved constellations_multi_snr.png")

    plt.show()
    print("\n  ✓ Phase 1 complete. All plots generated.\n")



if __name__ == "__main__":
    main()