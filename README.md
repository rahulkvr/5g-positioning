# 5G Indoor Positioning: RANSAC-UKF Tracking Pipeline

Robust indoor positioning using 5G NR Positioning Reference Signals (PRS) with multipath rejection and Kalman filtering.

## Results

| Metric | RANSAC-only | RANSAC + UKF |
|--------|-------------|--------------|
| Median error | 0.67 m | 1.23 m |
| 95th percentile | 2.41 m | **1.85 m** |

**23% reduction in worst-case error** — the UKF trades slight median bias for significantly tighter tails.

## Method

```
PRS Signals → TDoA Estimation → RANSAC Outlier Rejection → UKF Tracking → Position
```

1. **TDoA Estimation** — Correlate received 5G PRS with local replicas to get time-of-arrival
2. **RANSAC** — Sample anchor triplets, reject multipath/NLOS outliers via consensus
3. **Gauss-Newton** — Solve hyperbolic intersection on inlier set
4. **UKF** — Constant-velocity motion prior smooths trajectory, resets on gross outliers

## Simulation Setup

- **Environment:** 3D ray-traced train station (STL mesh)
- **Transmitters:** 12 gNodeBs at 3.5 GHz
- **Bandwidth:** 100 RB PRS (30 kHz SCS)
- **Channel:** Ray tracing with 2 reflections, 1 diffraction
- **SNR sweep:** -10 dB to 25 dB
- **Runs:** 1,300 Monte Carlo trials

## Quick Start

```matlab
% 1. Open MATLAB in repo root
% 2. Configure parameters in PositioningConfig.m
% 3. Run main script
main_5g_positioning
```

Results logged to `positioning_sweep_results.csv`. Generate plots with:

```bash
python generate_positioning_plots.py
```

## Key Files

| File | Description |
|------|-------------|
| `main_5g_positioning.m` | Entry point, runs sweep |
| `PositioningConfig.m` | All simulation parameters |
| `generateEnhancedPRSWaveform.m` | 5G NR PRS signal generation |
| `analyze_positioning_results.py` | Post-processing & plots |

## Tech Stack

`MATLAB` `5G Toolbox` `Ray Tracing` `RANSAC` `Unscented Kalman Filter` `Python` `Plotly`

## References

- 3GPP TS 38.211 — NR Physical channels and modulation
- 3GPP TR 38.855 — Study on NR positioning support
- Fischler & Bolles (1981) — RANSAC
- Julier & Uhlmann (2004) — Unscented filtering

---

*Research project @ TU Hamburg, Institut für Hochfrequenztechnik*
