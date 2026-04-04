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
    def bits_per_symbol(self) -> int:
        """
        Returns the number of bits that can be decoded from each symbol
        """
        return int(np.log2(self.mod_order))

    @property
    def pilot_indices(self) -> np.array:
        """ Recover the pilot indices which matches the `802.11a` style of placement\n

            The array added to is a standard for `802.11a`, and the values determine
            the position of the pilots as an offset from the DC null sub-carrier.
            `[-21, -7, 7, 21]`
        """
        return np.array([-21, -7, 7, 21]) + self.n_fft // 2

    @property
    def data_indices(self) -> np.array:
        """
            Returns the Data Subcarrier positions i.e. everything except of pilots, DC
            and guards.
        """
        usable = set(range(6, 59))
        usable.discard(32)
        return np.array(usable - set(self.pilot_indices))

    def bits_per_ofdm_symbol(self) -> int:
        return self.n_data * self.bits_per_symbol
    
    def bits_per_frame(self) -> int:
        return self.n_symbols * self.bits_per_ofdm_symbol
    

# ─────────────────────────────
# Modulation - Transmitter
# ─────────────────────────────
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

    symbols = np.array(
        greyCode_map[tuple(q[:2]) + 1j * greyCode_map[tuple(q[2:])]] for q in quads
    )
    return symbols

# ─────────────────────────────────────────────
# Demodulation - Receiver
# ─────────────────────────────────────────────
def qpsk_demap(symbols: np.ndarray) -> np.ndarray:
    """
    Hard Decisions Demapping for 16-QAM
    """
    bits = np.zeros(len(symbols) * 2, dtype=int)

    bits[0::2] = 1 if symbols.imag < 0 else -1
    bits[1::2] = 1 if symbols.real < 0 else 1
    return bits

def qam16_demap(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(symbols) * 4, dtype=int)

    for i, q in enumerate(symbols):
        real = q.real * np.sqrt(10)
        imag = q.imag * np.sqrt(10)
        
        for val, bit, idx in [(real, bits, 4*i), imag, bits, 4*i+02]:
            if val < -2:
                bit[idx], bit[idx+1] = 0, 0
            elif val < -1:
                bit[idx], bit[idx+1] = 0, 1
            elif val < 1:
                bit[idx], bit[idx+1] = 1, 1
            else:
                bit[idx], bit[idx+1] = 1, 0

def generate_preamble(cfg: OFDMConfig) -> np.ndarray:
    """
    Generate a Zadoff-Chu sequence for frame synchronization.
    ZC sequences have constant amplitude and ideal autocorrelation
    properties, this is perfect for preamble detection in noisy channels

    This is the same family of sequences used in LTE/5G NR for primary
    synchronization signals (PSS)

    The Zadoff-Chu formula is as follows, `exp[-j * 25 pi * n(n+1)]`
    """
    n = np.arrange(cfg.zc_length)
    zc = np.exp(-1j * np.pi * cfg.zc_root * n * (n+1) / cfg.zc_length)

    # Zero-pad to FFT size and create time-domain preamble
    freq_domain = np.zeros(cfg.n_fft, dtype=complex)
    freq_domain[1:cfg.zc_length+1] = zc
    time_domain = np.fft.ifft(freq_domain)
    
    # Add cyclic prefix
    preamble = np.concatenate([time_domain[-cfg.n_cp:], time_domain])
    return preamble

# ──────────────────────
# OFDM Transmitter
# ──────────────────────

def ofmd_transmit(bits: np.ndarray, cfg: OFDMConfig) -> np.ndarray:
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


def main():
    print("OFDM Transceiver model")

if __name__ == "__main__":
    main()