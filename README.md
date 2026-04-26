# ofdm-transceiver
Portfolio Project to build a complete OFDM wireless link prospectively using two ADALM-PlutoSDR boards. Essentially building a simplified version of the physical layer used in 4G/5G/Wi-Fi

Built as a pre-research skill builder for optical satellite communication work at McMaster University

# OFDM Wireless Transceiver with FPGA Acceleration
 
> A complete OFDM physical layer — from Python simulation to real over-the-air 
> transmission on PlutoSDR to hardware-accelerated DSP on a Zynq 7010 FPGA.
 
![System Architecture](docs/results/system_architecture.png)
 
## Key results
 
| Metric | Value |
|--------|-------|
| Modulation | QPSK and 16-QAM |
| Over-the-air BER @ 15 dB SNR | < 10⁻⁴ |
| FPGA FFT latency vs Python | 340x faster |
| FPGA resource utilization | 47% LUT, 38% BRAM |
| Real-time throughput | 1.2 Mbps |
 
![BER Curve](docs/results/ber_curves/ber_qpsk_measured_vs_theory.png)
![Constellations](docs/results/constellations/multi_snr_comparison.png)
 
## What this project demonstrates
 
- **End-to-end wireless system design**: bit generation → modulation → 
  OFDM framing → RF transmission → synchronization → channel estimation 
  → equalization → demodulation → BER measurement
- **Real RF hardware**: two ADALM-PlutoSDR boards transmitting at 915 MHz 
  ISM band with real multipath, CFO, and noise
- **FPGA acceleration**: custom 64-point pipelined FFT and preamble 
  correlator in Verilog on the Pluto's Zynq 7010, integrated via AXI-Stream
- **Rigorous validation**: simulated BER curves match QPSK/16-QAM theory 
  within 1 dB; real measurements annotated with channel conditions
 
## System architecture
 
```
TX Chain:                          RX Chain:
┌──────────┐                       ┌──────────────┐
│ Bit Gen  │                       │ Preamble Det │◄── Zadoff-Chu correlator
└────┬─────┘                       └──────┬───────┘
     ▼                                    ▼
┌──────────┐                       ┌──────────────┐
│QPSK/16QAM│                       │  CP Removal  │
└────┬─────┘                       └──────┬───────┘
     ▼                                    ▼
┌──────────┐     ┌──────────┐      ┌──────────────┐
│ IFFT+CP  │────►│ PlutoSDR │─RF──►│  FFT (FPGA)  │
└──────────┘     │   TX     │      └──────┬───────┘
                 └──────────┘             ▼
                                   ┌──────────────┐
                                   │  Ch. Est.    │◄── LS via pilots
                                   └──────┬───────┘
                                          ▼
                                   ┌──────────────┐
                                   │  Equalize +  │
                                   │  Demap       │
                                   └──────────────┘
```
 
## Project structure
 
```
src/python/          Phase 1: software OFDM simulation
src/gnuradio/        Phase 2: GNU Radio flowgraphs for PlutoSDR
src/verilog/         Phase 3: FPGA RTL (FFT, correlator, AXI wrapper)
docs/journal/        Engineering journal (daily progress + decisions)
docs/theory/         Technical write-ups on each DSP concept
docs/results/        All measurement data and plots
hardware/            Setup photos and bill of materials
```
 
## Quick start
 
```bash
# Phase 1: Run simulation
pip install numpy scipy matplotlib
python src/python/ofdm_transceiver.py
 
# Phase 2: GNU Radio (requires PlutoSDR + pyadi-iio)
gnuradio-companion src/gnuradio/ofdm_tx.grc
 
# Phase 3: FPGA (requires Vivado WebPACK)
cd src/verilog && make synth
```
 
## Technologies
 
Python · NumPy · SciPy · GNU Radio · ADALM-PlutoSDR · Verilog · 
Vivado · Zynq 7010 · AXI-Stream · libiio · pyadi-iio
 
## Context
 
Built as a pre-research skill builder before joining optical satellite 
communication research at McMaster University (OGRE ground station project, 
Ottawa). The synchronization, channel estimation, and noise analysis 
techniques here directly apply to free-space optical link acquisition 
and performance evaluation.
 
## Documentation
 
- [Engineering Journal](docs/journal/) — daily progress and decision log
- [Design Decisions](docs/design_decisions.md) — why I chose X over Y
- [Theory Write-ups](docs/theory/) — deep dives on each DSP concept
- [References](docs/references.md) — papers, textbooks, videos used