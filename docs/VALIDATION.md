# Validation record and interpretation

These are **independent** Geant4 11.4.2 CPU and Apple Metal photon histories.
The saved JSON reports in `tutorial/saxs_keele/gpu_transport/results/` are
historical records from the source project. Raw detector images and ROOT
histories were discarded; a saved p-value cannot be recalculated from JSON
alone. Fresh runs use the scripts in [EXAMPLES.md](EXAMPLES.md).

## Checks by mechanism

| Check | Conditions | Observation | What it establishes |
|---|---|---|---|
| RNG | Three Random123 Philox4x32-10 known-answer vectors | Bit-for-bit match (`philox_metal_kat.swift`) | RNG transformation is implemented correctly; says nothing about photon physics. |
| Compton final state | Water, 22.0220601 keV; 297,016 events per engine | 32 × 32 angle/energy histogram p = 0.856 | No detectable difference at this count and binning. |
| Compton final state | g50 mixture, 22.162917 keV; 372,488 events per engine | p = 0.825 | Same bounded statement for this composition and energy. |
| Compton final state | g50 mixture, 30 keV; 499,573 events per engine | p = 0.490 | Same bounded statement at 30 keV. |
| Full water transport | 100 mm thick, 120 × 120 mm box; 22.0220601 keV; 200 M incident photons per engine; independent seeds | 200-bin full-covariance profile χ²/ν = 0.947, p = 0.693; sample direct/single-Rayleigh/multiple-Rayleigh/Compton differences 0.18/1.09/0.12/−1.29σ | Profile and channels are statistically compatible in this run. It lacks precision to exclude a ≈0.23% direct-channel offset. |
| Full water transport, retained high-statistics reference | 10 × 200 M Geant4 histories versus 10 × 200 M Metal histories | Direct Metal −0.229%, z = −2.74; Compton −0.290%, z = −1.25 | Full channel parity remains unresolved. The old CPU image was discarded, so no formal full-covariance profile p-value is claimed for this pairing. |
| Changed material | 45/45/10 water/fat/collagen by mass; 50 mm box; 22.162917 keV; 20 M photons per engine | 256-bin profile χ²/ν = 1.061, p = 0.242; four channel differences within 2σ; q = 2–28 nm⁻¹ relative RMS = 5.75% | Agreement at tested count and geometry; not a proof for every mixture or thickness. |

The water setup has a parallel 100 µm diameter circular source, 100 mm air
after the sample and an ideal 1000² detector with 100 µm pixels. Both images
use the same 150 mm mean-plane distance, 1–30 nm⁻¹ q range, 200 points and
`correct_solid_angle=False`. The 45/45/10 setup has a focused source,
110 mm downstream air and a 135 mm mean-plane distance. For both, the same
pyFAI CSR pixel-splitting operator supplies the adjacent-bin covariance;
fractionally split bins are not treated as independent Poisson counts.

The water [fresh independent comparison](../tutorial/saxs_keele/gpu_transport/results/water_100mm_finite_box_independent_cpu_1x200m/summary.json),
[retained high-statistics comparison](../tutorial/saxs_keele/gpu_transport/results/water_100mm_finite_box_metal_10x200m/summary.json),
and [changed-material comparison](../tutorial/saxs_keele/gpu_transport/results/trial_45_45_10_finite_box_20m/validation.json)
contain exact counts and geometry. The Compton records are in
`water_penelope_compton_joint/`, `g50_penelope_compton_joint/` and
`g50_30kev_penelope_compton_joint/` under the same `results/` directory.

## Water thickness series against an external reference

A separate colleague-supplied joblib was compared at 20, 30 and 40 mm water
thickness, λ = 0.0563 nm and 50 M simulated incident photons per thickness.
The CPU profiles had shape correlation 0.9963/0.9955/0.9939 against the
colleague profile. Their near-axis relative intensities at 20/30/40 mm were
1/0.5252/0.2759, compared with 1/0.5193/0.2695 in the colleague file.
This agreement depends on an **inferred** 1000² detector with ≈240 µm pitch:
the reference specifies neither pixel geometry, mask nor incident flux.
Its absolute scale is therefore undetermined. The joblib itself is not
redistributed. The [summary](../tutorial/saxs_keele/gpu_transport/results/colleague_water_cpu_gpu_50m/summary.json)
records the assumptions and profile metrics.

## Published 100-million-history composition suite

The public example suite contains 12 primary comparisons with 100 million
incident histories per backend and case: pure water, fat and collagen; all
three 50/50 binary mixtures; an equal ternary mixture; and water and g50
thickness series. The detector distance from the mean scattering plane is
fixed at 150 mm while sample thickness varies.

Eleven primary profile tests have p ≥ 0.05. The `g50_25mm` primary comparison
has χ²/ν = 1.261 and p = 0.00288, although its integrated profile differs by
−0.037% and all four channel counts differ by at most 1.04σ. A second
100-million-history comparison with independent Geant4 and Metal seeds gives
χ²/ν = 1.036 and p = 0.331, with channel differences below 0.89σ. The low
primary p-value therefore did not reproduce. Across the 12 primary cases, the
largest channel difference is 2.62σ and integrated profile differences range
from −0.222% to +0.314%.

The profile operator uses one temporary guard bin below and above the requested
1–30 nm⁻¹ range. pyFAI's CSR builder otherwise clips radial underflow and
overflow into the first and last bins, producing artificial edge jumps in a
ring-sum profile. The two guard rows are discarded before profile statistics.

These observations bound the tested compositions, energy and geometry. They
do not establish exact equality of the samplers or validate a measured detector
response. Full tables, covariance tests and figures are in
[`examples/xrd_suite/results_100m`](../examples/xrd_suite/results_100m/RESULTS.md).

## Timing and remaining tests

In the maintained 100-million-history suite, complete-process wall-time ratios
range from 185× to 314× on the recorded M4 Pro host. A separate 20-million-
history image-output record gives 400.5×, but Geant4 writes CSV while Metal
writes a raw binary image. Warm Metal transport-and-reduction intervals give
364–372× in two older 20-million-history records; those ratios exclude Metal
startup and are not symmetric end-to-end measurements. Exact definitions,
records and limitations are reported in [PERFORMANCE.md](PERFORMANCE.md).

These values compare different implementations on one host. They do not
predict arbitrary Geant4 workloads or clinical inverse-model throughput.

The strongest unresolved test is a fresh independent multi-billion-photon
Geant4/Metal pair with retained covariance, followed by single-scatter
Rayleigh angular histograms. It should determine whether the residual comes
from the different Rayleigh samplers, interpolated cross sections, finite-box
transport or stochastic fluctuation. Until then, "Geant4-equivalent" would
overstate the evidence.
