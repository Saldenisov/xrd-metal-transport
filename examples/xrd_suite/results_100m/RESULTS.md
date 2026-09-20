# Geant4 versus Metal: X-ray diffraction examples

Each row compares 100,000,000 independent histories per backend. Profile p-values use the full pyFAI pixel-splitting covariance.

| Case | Water/fat/collagen | Thickness | Geant4 | Metal | Speedup | Profile χ²/ν | p | Figure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `collagen_50mm` | 0/0/1 | 50 mm | 307.7 s | 1.095 s | 281× | 1.017 | 0.412 | [plot](collagen_50mm/detector_comparison.png) |
| `equal_ternary_50mm` | 0.333/0.333/0.333 | 50 mm | 317.4 s | 1.096 s | 290× | 0.931 | 0.778 | [plot](equal_ternary_50mm/detector_comparison.png) |
| `fat_50mm` | 0/1/0 | 50 mm | 328.7 s | 1.242 s | 265× | 0.975 | 0.602 | [plot](fat_50mm/detector_comparison.png) |
| `fat_collagen_50mm` | 0/0.5/0.5 | 50 mm | 354.2 s | 1.126 s | 314× | 0.988 | 0.541 | [plot](fat_collagen_50mm/detector_comparison.png) |
| `g50_100mm` | 0.386/0.51/0.104 | 100 mm | 286.0 s | 0.959 s | 298× | 0.996 | 0.505 | [plot](g50_100mm/detector_comparison.png) |
| `g50_25mm` | 0.386/0.51/0.104 | 25 mm | 317.6 s | 1.301 s | 244× | 1.261 | 0.00288 | [plot](g50_25mm/detector_comparison.png) |
| `g50_50mm` | 0.386/0.51/0.104 | 50 mm | 294.1 s | 1.150 s | 256× | 0.954 | 0.691 | [plot](g50_50mm/detector_comparison.png) |
| `water_100mm` | 1/0/0 | 100 mm | 310.6 s | 1.143 s | 272× | 0.917 | 0.825 | [plot](water_100mm/detector_comparison.png) |
| `water_10mm` | 1/0/0 | 10 mm | 313.2 s | 1.690 s | 185× | 1.025 | 0.377 | [plot](water_10mm/detector_comparison.png) |
| `water_50mm` | 1/0/0 | 50 mm | 293.3 s | 1.406 s | 209× | 0.975 | 0.601 | [plot](water_50mm/detector_comparison.png) |
| `water_collagen_50mm` | 0.5/0/0.5 | 50 mm | 344.4 s | 1.394 s | 247× | 1.041 | 0.313 | [plot](water_collagen_50mm/detector_comparison.png) |
| `water_fat_50mm` | 0.5/0.5/0 | 50 mm | 296.4 s | 1.099 s | 270× | 0.955 | 0.686 | [plot](water_fat_50mm/detector_comparison.png) |

## Interpretation

11/12 primary profile tests have p ≥ 0.05. The lowest value is `g50_25mm` at p = 0.00288; its four channel differences are within 1.04σ and its integrated profile difference is -0.037%.

Across the 12 primary cases, the largest absolute channel difference is 2.62σ, integrated profile differences span -0.222% to 0.314%, and observed wall-time speedups span 185× to 314×. These are descriptive results for this hardware and model, not universal performance or equivalence claims.

An independent 100-million-history `g50_25mm` repeat with new Geant4 and Metal seeds gives p = 0.331, with all four channels within 0.89σ. The low primary p-value did not reproduce.

[Open the independent `g50_25mm` repeat](replicates/g50_25mm_seed2/detector_comparison.png).

![Profile overview](profile_overview.png)

![Parity overview](parity_overview.png)

The comparisons cover the finite homogeneous sample and ideal detector implemented by both engines. Remaining model limitations are recorded in every case summary.
