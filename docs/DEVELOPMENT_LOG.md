# Historical development log

The text below records the sequence of prototype experiments before this
repository was separated from the Keele workspace. Its old commands and
relative `results/` paths are not the maintained interface. Use the root
README, `docs/EXAMPLES.md` and `docs/VALIDATION.md` for current commands,
data locations and bounded conclusions. Some experiments in this log used
intermediate physics implementations and are intentionally not release claims.

# Keele photon transport on Apple Metal

## Modular transport and GPU radial reduction (2026-09-20)

The 829-line notebook prototype combined an embedded Metal shader, device
selection, physics-table construction, transport dispatch and radial reduction
in one Swift file. The maintained repository now separates those concerns into
four Metal files and six Swift files. The numerical transport path was checked
bit for bit against the previous executable for 200,000 fixed-seed histories.

The notebook's multi-device selection and sparse GPU radial reducer were
retained. A shared `SlabDetectorGeometry` now emits matched inputs for Geant4
and Metal validation runs. This removes duplicated dimensions but does not
make Metal execute arbitrary Geant4 geometry. A 200,000-history check found
identical channel counts between image and radial modes, and exact agreement
between GPU radial sums and multiplication of the serialized sparse operator
by the full detector image.

On the M4 Pro, one fixed-seed 5-million-history water run took 0.1261 s in
image mode and 0.0505 s in radial mode, including process startup and shader
compilation (2.50-fold wall-time reduction in this single run). The transferred
result decreased from a 4,000,000-byte detector image to 1,600 bytes for 200
double-precision radial sums. The maximum difference from multiplying the
full image by the same pyFAI operator was 1.96 × 10⁻⁶ counts (7.18 × 10⁻⁸
relative). These timings describe this hardware and geometry; they are not a
general performance guarantee.

## Independent 100 mm water parity benchmark

`validate_water_100mm_cpu_metal.py` runs independent Geant4 CPU and Apple
Metal histories at 22.0220601 keV in a 100 mm water box. Both use a parallel,
uniform circular 100 µm diameter source, 100 mm of downstream air, and an
ideal 1000 × 1000 detector with 100 µm pixels.
The energy corresponds to the colleague water data's 0.0563 nm wavelength;
it is not the 22.162917 keV Ag Kα1 line.

Geant4 image-only CSV mode records four sample-interaction history classes
alongside its detector image: direct,
single Rayleigh, multiple Rayleigh, and any Compton. Both backends
transport photons through water and air, but Geant4's channel counters
include sample interactions only. Metal flags air interactions separately.
No ROOT file is created. Temporary detector images are deleted
after integration.

Both detector images are integrated with **the same XRD-preprocessing call**:
200 q points over 1–30 nm⁻¹, `correct_solid_angle=False`, and the 150 mm
distance from the slab's mean scattering plane (100 mm exit gap + half of the
100 mm slab). The formal profile parity test uses pyFAI's *same* CSR
pixel-splitting operator, with its full adjacent-bin covariance. It does not
pretend fractional `sum_signal` bins are independent Poisson counts.

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python -m pytest -q geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/test_validate_water_100mm_cpu_metal.py
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/validate_water_100mm_cpu_metal.py --photons 200000000 --repetitions 10 --threads 14 --metal-physics geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/results/water_100mm_cpu_metal_10x200m/physics_exact.json --output-dir geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/results/water_100mm_cpu_metal_philox_10x200m_full
```

The initial 10 × 200 M Geant4/Metal run used `xorshift32`, a sample Rayleigh
law for air, and six-node interpolated cross sections. It gave full-covariance
200-bin profile χ²/ν = 0.956 (p = 0.660) but a direct-detector-count deficit
of 0.568% on Metal (two-sample Poisson z = −6.80). Thus the profile alone did
not establish channel parity. Its persistent results are in
`results/water_100mm_cpu_metal_10x200m/`.

After that run, Metal was changed to counter-based Philox4x32-10, atomic
N/O Rayleigh sampling for air, and full-path interaction classes. A one-event
Geant4 run measured exact cross sections at 22.0220601 keV without ROOT,
replacing the nearby interpolation node (`measure_exact_water_xs.py`). Ten new
200 M Metal runs against the retained Geant4 reference gave *approximate*
200-bin χ²/ν = 1.096 (p = 0.168), because the discarded Geant4 pixel image
prevents exact post-hoc covariance reconstruction. Direct count still differs
by −0.296% (z = −3.54); Compton count differs by +0.987% (z = +4.24), and
multiple Rayleigh by +1.134% (z = +2.90). Do **not** call the full transport
physics Geant4-equivalent yet. These results are in
`results/water_100mm_metal_philox_exact_xs_10x200m/`.
Future full CPU baselines retain the small 200 × 200 Geant4 covariance matrix
in `profiles.npz`, permitting exact later GPU-only Poisson comparisons without
retaining detector images.

The 10 Geant4 transports took 2932.4 s on 14 CPU threads; the corrected
Metal transports took 6.25 s in total on the M4 Pro (469× transport-only
timing ratio). The full Metal validation pass, including image integration,
took 7.32 s. These are different physics implementations; the ratio is a
prototype throughput result, not a speed claim for a validated clinical
inverse model.

### CUDA reference audit

The [FDA MC-GPU v1.3 source](https://github.com/DIDSR/MCGPU) was inspected at
commit `cb16a5f52661ea5342d6f8291c78b7584dfba60c`. Its `track_particles`
kernel batches independent histories per thread; it uses voxel Woodcock
tracking, material-specific Rayleigh sampling, and shell-aware Compton
sampling including Doppler broadening. The repository provides example inputs
and a CPU/CUDA compilation path, but no automated parity test suite to import.
The [VICTRE_MCGPU reference](https://github.com/DIDSR/VICTRE_MCGPU) also
documents mammography examples and a single-precision near-axis artifact.

Before the shell/Doppler port below, Metal already parallelized independent
histories but used an incident-energy Compton angle CDF and a free-electron
energy shift. Copying MC-GPU's atomic Rayleigh form-factor tables would
replace the Keele molecular scattering model, not validate it. Once full
transport parity is established, detector scoring can be reduced directly
to a 1D profile to avoid moving a 1000² image for each inverse-model trial.

### Penelope shell/Doppler port (2026-09-17)

The Metal scattering module now samples shell-aware Compton angle and outgoing
energy using the Penelope-2008 algorithm in Geant4's
`G4PenelopeComptonModel.cc`, adapted to Metal's float arithmetic and
independent Philox photon histories. `SAXSRunAction.cc` exports the *actual*
Geant4 oscillator strengths, ionisation energies and Hartree factors for the
sample and air. `prepare_multi_physics.py` stores these in the compact physics
JSON while deleting its temporary ROOT file. The Keele MIFF Rayleigh kernel
is unchanged. FDA MC-GPU was used as the GPU transport design reference, but
its different PENELOPE-2006 shell tables were not substituted for Geant4's.

`metal_forward.py` now asks Geant4 for a fresh oscillator table whenever an
inverse-model trial changes water/fat/collagen composition. It caches tables
by composition; exporting one new composition takes about 0.5 s on this host
and creates no ROOT file. This setup cost must be included in cold-trial
end-to-end timings. The 22.162917 and 30 keV optimizer references now contain
Penelope shell tables; arbitrary residual shape changes at fixed composition
reuse the cache. The GPU still does not implement voxel geometry, electron
transport, fluorescence or a non-ideal detector.

Independent Geant4/Metal Compton final-state checks (same sample material,
independent histories) found no detectable difference in 32×32 joint
angle/outgoing-energy histograms: water at 22.0220601 keV, 297,016 events per
backend, p=0.856; g50 at 22.162917 keV, 372,488 events, p=0.825; g50 at
30 keV, 499,573 events, p=0.490. See `results/*penelope_compton_joint/`.
The fresh 200M+200M water transport comparison gave the exact 200-bin pyFAI
CSR-covariance profile χ²/ν=1.023, p=0.399; Geant4 transport 298.5 s and
Metal transport/process wall 0.82 s. See
`results/water_100mm_penelope_exact_1x200m/`.

For higher-statistics channels, 10×200M new Metal histories compared with the
retained 10×200M Geant4 CPU reference gave direct −0.187% (z=−2.24), single
Rayleigh −0.020% (z=−0.13), multiple Rayleigh +0.768% (z=1.97) and Compton
−0.822% (z=−3.55). These are **sample-only** interaction classes, matching
`SAXSSteppingAction.cc`; air interactions still affect paths and are reported
separately. Full Geant4 transport parity is not yet established. The old CPU
pixel images were discarded, so this retained-CPU comparison deliberately
reports no formal profile p-value. See
`results/water_100mm_penelope_sample_channels_10x200m/`.

A changed 45/45/10 water/fat/collagen material, 50 mm thick, was generated
independently by Geant4 and Metal with 20M incident photons each. All four
detector-channel differences were within 2σ; the 256-point radial profile
had 5.7% RMS relative difference across q=2–28 nm⁻¹ at this count. See
`results/trial_45_45_10_penelope_transport_20m/`. No ROOT files are retained.

### Finite sample-box parity correction (2026-09-17)

The previous Metal kernel transported the sample as an *infinite lateral slab*,
while every Geant4 benchmark used a **120 × 120 mm finite box**. A photon that
left a Geant4 side face continued through air; Metal kept it in tissue. This
was a forward-model geometry error, independent of the inverse reconstruction.
The Metal kernel now tracks all six sample faces, possible air-to-sample
re-entry, and the same 10 m Geant4 world bounds. A `sample_lateral_mm` physics
field makes this explicit; existing Geant4-matched tables default to 120 mm.
The photon loop cap increased from 32 to 128 for process-ablation diagnostics.

A 1M-photon **Compton-only, no-photoelectric** diagnostic made the side-face
effect visible: the old infinite-slab Metal result had 34,657 detected Compton
histories versus Geant4's 25,606 (+35.3%); after the finite-box correction it
had 25,832 (+0.88%, z=1.00). The 1M Rayleigh-only direct and single/multiple
Rayleigh channel differences after correction were all below 0.6σ. This
ablation is a geometry diagnostic, not a physical measurement configuration.

Against the retained 2B-photon Geant4 water reference, two independent new
2B-photon Metal runs gave Compton differences −0.290% (z=−1.25) and −0.494%
(z=−2.13), improved from the infinite-slab −0.822% (z=−3.55). Single-Rayleigh
differences were within 0.8σ, multiple-Rayleigh within 1.7σ. The direct
channel remained approximately −0.229% (z≈−2.74) in both runs, so **full
transport parity is still unproven**. The old Geant4 pixel images were discarded;
these retained-reference comparisons cannot provide a formal 1D-profile p-value.
The new finite-box test is `test_finite_box_compton_ablation_matches_geant4`.

`measure_exact_water_xs.py --audit-energy-kev` now compares Metal's six-node
post-Compton cross-section interpolation with Geant4's exact model *and*
tracking lambda table without creating ROOT files. At 21.5 keV, sample
photoelectric/Rayleigh interpolation errors were +0.282%/+0.459%; this is a
remaining approximation to address, not a demonstrated explanation for the
direct-channel difference at the 22.0220601-keV source energy, where all three
cross sections agree exactly.

A **new independent CPU/GPU seed pair** (`--seed-run-offset 100`, 200M photons
per engine) gave the exact 200-bin XRD-preprocessing pyFAI CSR-covariance
profile χ²/ν=0.947, p=0.693. Direct, single-Rayleigh, multiple-Rayleigh and
Compton detector-channel differences were respectively 0.18σ, 1.09σ, 0.12σ
and −1.29σ. Geant4 transport took 289.05 s on 14 CPU threads; Metal transport
and detector tally took 0.813 s on the M4 Pro. See
`results/water_100mm_finite_box_independent_cpu_1x200m/`. This new run does
not reproduce the retained 2B-CPU direct-channel offset, but its 200M count
has insufficient precision to rule out a 0.23% effect. A new independent
multi-billion-photon Geant4 baseline would be needed to settle that residual.
Neither engine's detector response is modeled here; both score ideal photon
entrance crossings. No ROOT or raw image files were retained.

Higher-statistics process ablations without photoelectric absorption were
also run with no ROOT files: at 20M photons, Compton-only detector counts were
516,030 Geant4 versus 514,948 Metal (−0.210%, z=−1.07). Across three
independent 20M-photon Rayleigh-only pairs, multiple-Rayleigh counts were
1,581,872 versus 1,574,986 (−0.435%, z=−3.88); direct counts were within
0.6σ pooled. This small Rayleigh-only multi-scatter discrepancy is **not
resolved**. The custom-MIFF Geant4 sampler uses a RITA table and spline in
log(q²)/log(F²), whereas Metal integrates a finite angular CDF from linearly
interpolated file amplitudes. That algorithmic difference is a candidate, not
a proven cause; it should be isolated with single-scatter angular histograms
before claiming parity for arbitrary thicknesses or mixtures.

The 50 mm **45/45/10 water/fat/collagen** mixture was revalidated after the
finite-box fix with 20M incident Ag-Kα1 photons per engine, a focused beam,
110 mm of downstream air and the same ideal 1000² detector. The Geant4
comparison now uses image-only CSV; **no ROOT file is created**. Both images
go through XRD-preprocessing with the 135 mm mean-plane distance, explicit
`correct_solid_angle=False`, and 256 q points over 1–30 nm⁻¹. The same pyFAI
CSR operator gives a full-covariance profile χ²/ν=1.061, p=0.242; all four
detector-channel differences are within 2σ. Geant4 wall time was 33.66 s
on 14 CPU threads; Metal process wall time was 0.171 s, including startup and
image output (197× for this benchmark). The figure and report are in
`results/trial_45_45_10_finite_box_20m/`. This confirms parity at the tested
count and geometry, not a proof for every material or thickness.

MC-GPU also partitions RANECU streams by skip-ahead. Our earlier Metal
`xorshift32` state was too short for billion-history independence; its initial
benchmark p-value is therefore provisional. The current kernel uses
Philox4x32-10 with photon ID and draw-block counter, and distinct run keys.
The CPU/GPU physics parity caveat above remains despite this RNG fix.

The [Random123 Philox4x32-10 reference](https://github.com/DEShawResearch/random123)
has official known-answer vectors. `philox_metal_kat.swift` passes all three
vectors bit-for-bit, and the same counter transformation is now integrated
into `Metal/Random.metal`. The vector test checks RNG correctness, not
Rayleigh/Compton physics.

## Multi-collision forward model (experimental)

The `transport` kernel launches one independent photon history per Metal
thread. For a homogeneous 50 mm slab it transports repeated sample Rayleigh
and Compton interactions, energy-dependent photoelectric attenuation, air,
and an ideal 1000 × 1000 detector. Rayleigh angles come from the trial Keele
MIFF (`F/sqrt(W) = sqrt(S)`); macroscopic cross sections come from Geant4.
The Compton angular CDF is calibrated using temporary Geant4 Penelope
histories. `metal_forward.py` updates cross sections for each trial mixture and
`S(q)` and can be selected with `--backend metal` in
`synthetic_shape_free_residual_newton.py`. ROOT and raw images are temporary.

### Colleague water comparison (20/30/40 mm)

The provided joblib has wavelength 0.0563 nm, 500 mm entrance-plane distance,
and 1000 radial samples, but no incident flux, detector pixel geometry or mask.
The comparison infers a 1000², approximately 240 µm/pixel detector from its
reported q maximum. The upstream air path is 90/85/80 mm and the downstream
gap is 480/470/460 mm. It runs independent Geant4 CPU and Metal GPU histories
with the same source, MIFF and ideal entrance-crossing detector, then uses
XRD-preprocessing with thickness correction (490/485/480 mm effective distance).
Absolute scale is unknown; shape errors use q=5–30 nm⁻¹ area normalization.

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/prepare_multi_physics.py --material water --energy-kev 22.0220601 --photons 2000000 --output geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/results/multi_transport/water_22p022_physics.json
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/compare_colleague_water_cpu_gpu.py --photons 50000000 --threads 14
```

For 50 million incident photons per thickness, Geant4 CPU took 79.1/77.3/75.8 s
and Metal GPU 0.199/0.193/0.206 s. Independent CPU/GPU integer annular counts
at q=5–30 nm⁻¹ gave reduced two-sample Poisson chi-square 0.97/1.02/1.10.
Area-normalized shape RMS errors against the colleague file were 3.7/4.1/4.6%
for CPU and 4.0/4.1/4.8% for GPU. This supports GPU parity with the current
Geant4 model, not the correctness of the inferred detector metadata or a
claim that the colleague file is an absolutely calibrated experiment.
The near-axis intensity ratios for 20:30:40 mm were 1:0.519:0.269 in the
colleague file, 1:0.525:0.276 on CPU and 1:0.525:0.275 on GPU.

Reproduce from the Keele notebook directory with `eosdx13`:

```sh
./scripts/build.sh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/prepare_multi_physics.py --photons 1000000 --output geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/results/multi_physics_g50.json
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/run_multi_transport.py --photons 200000000
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/validate_multi_transport.py --photons 20000000 --threads 8
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/check_trial_xs.py
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python synthetic_shape_free_residual_newton.py --backend metal --iterations 2 --fit-photons 5000000 --validation-photons 100000000
```

On this M4 Pro, the matched 20-million-photon detector-image workload took
42.34 s in 8-thread Geant4 image-only CSV mode and 0.106 s in Metal including
process startup (400× wall-clock speedup; output serialization differs). A
200-million-photon Metal image took 0.58 s for transport and pixel histogram;
the Python integration and figure took additional time. At 20 million
independent photons, the Geant4/Metal thickness-corrected radial profiles
had 0.18% mean relative bias and 5.2% raw 256-bin RMS relative difference
(1.5% after 4-bin Gaussian smoothing). These are *simulation comparisons*,
not a claim of equivalence for arbitrary breast measurements.

For the earlier target of 50 million detected Rayleigh photons, a 3.8-billion-
incident-photon Metal run yielded 46,460,671 one-Rayleigh and 3,813,914
multi-Rayleigh hits (50,274,585 total), with 9.88 s process wall time for
transport and detector histogram. The final thickness-adjusted figure took
12.8 s including Python integration and plotting.

The three-case inverse run made 142 new Metal forward calls in 28.2 s, but
the unknown residual remains incompletely recovered: a lower detector-space
deviance can accompany a worse true component estimate. More photons per
Jacobian did not remove that ambiguity.

An additional bounded, LM-style damped Gauss–Newton experiment is in
`synthetic_shape_free_trust_region.py`. On the same three synthetic targets,
5M-photon fit calls and independent 100M-photon validation, it used 180 Metal
forward calls in 28.2 s. Total-S relative errors were 1.56/1.61/2.13%,
versus 1.72/1.68/2.17% for the previous undamped run. Errors on the unknown
nonnegative extra profile remained 35.9/34.7/25.4%, versus 47.3/40.9/26.6%.
The condition number of the local weighted response Jacobian was about
1,200–1,450: damping helped modestly but did not resolve composition/residual
ambiguity. Only the comparison figure and JSON report were retained; trial
images were temporary and no ROOT files were created.

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python synthetic_shape_free_trust_region.py --iterations 2 --fit-photons 5000000 --validation-photons 100000000
```

The physical extra component is constrained nonnegative during fitting; a
signed residual from post-hoc least-squares projection is not equivalent to
that component. If the extra has different elemental composition, its
attenuation and Compton cross sections must be parameterized separately.
The current synthetic model treats it as extra coherent scattering with the
three-component material mixture still determining non-Rayleigh transport.

This is a local damped Gauss–Newton comparison, not a proof that it is globally
optimal or faster per accurate recovery than the undamped method. The fixed
damping candidates are 0.1, 1 and 10; adaptive trust-radius updates and
coarse-to-fine residual bases remain future tests.

### Paired 22.162917/30 keV synthetic inverse test

`synthetic_dual_energy_prepare.py` pairs the existing three 200M-photon
Geant4 Ag-Kα1 targets with three new, independent 200M-photon 30 keV Geant4
images at the same 50 mm geometry and the same unknown S(q). The 30 keV
Compton angular CDF was calibrated from 1M Geant4 photons; both energies use
energy-dependent Geant4 cross-section tables and separate photoelectric,
Compton, Rayleigh and air transport. XRD-preprocessing integrates each image
into 256 points over q=1–30 nm⁻¹ with the same 135 mm thickness-corrected
distance. No ROOT files are retained.

`synthetic_dual_energy_fit.py` uses the *same* 23-control nonnegative residual
parameterization and damped Gauss–Newton search for single-energy and joint
fits. The two-energy objective sums the separate absolute radial-count and
direct-transmission Poisson deviances; it does not average profiles with
different wavelengths. Fitting used 50M incident photons per forward call per
energy, followed by independent 200M-photon validation per energy. GPU random
streams are independent across energies but shared across finite-difference
perturbations within an energy.

| Unknown excess | Extra-S error, 22 only → 22+30 | Total-S error | Max fraction error |
|---|---:|---:|---:|
| One peak | 39.7% → 14.3% | 1.23% → 0.70% | 1.53 → 0.66 percentage points |
| Two separated peaks | 39.9% → 16.7% | 0.97% → 0.82% | 1.64 → 0.38 points |
| Two overlapping peaks | 17.3% → 9.5% | 1.11% → 0.88% | 1.07 → 0.63 points |

Across all three cases the inversion used 282 new 22-keV and 183 new 30-keV
Metal calls in 185.4 s total. Geant4 generation of the three 30-keV targets
took 326.7/334.8/327.5 s. At *generating truth*, independent Metal and
Geant4 targets gave joint per-energy 256-bin-plus-direct Poisson deviances
230–251 at 22 keV and 263–266 at 30 keV; this is a post-fit forward-model
audit, not a parameter-selection oracle. Its agreement supports the paired
test within the present ideal detector model, not full Geant4 equivalence.

The 30 keV energy contributes information beyond merely duplicating 22 keV
in the *local linearized* response: at equal incident-photon count, the
regularized residual-control variance trace is 0.86–0.88 times that of two
22-keV exposures; the two independent material-coordinate variance trace is
0.26–0.32 times. These are local Fisher approximations, not an empirical
same-dose control with a second independent Geant4 22-keV target. Both
energies also double the incident dose relative to one exposure. The extra
component has no separate elemental composition in this synthetic test; it
changes coherent scattering only.

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/prepare_multi_physics.py --material g50 --energy-kev 30 --photons 1000000 --threads 14 --output geant4-xray-diffraction/tutorial/saxs_keele/gpu_transport/results/multi_transport/g50_30kev_physics.json
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python synthetic_dual_energy_prepare.py --threads 14
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python synthetic_dual_energy_fit.py --iterations 2 --fit-photons 50000000 --validation-photons 200000000
```

Therefore this backend is opt-in;
do not use it as the final calibrated clinical or Geant4-equivalent model.
It omits Penelope Compton Doppler broadening, fluorescence and electron
secondaries, material-dependent Compton CDF changes, full air angular law,
detector energy response, and heterogeneous/voxel geometry. The provided
validation scripts explicitly quantify this gap.

## Earlier single-Rayleigh feasibility stage

The original `single_rayleigh.swift` tests whether photon histories can be
parallelized on the Apple M4 Pro GPU. It implements the direct and **exactly one Rayleigh interaction**
channels, with a 50 mm box, focused circular source, air gap, finite detector,
Geant4-measured total/Rayleigh attenuation coefficients, and the same Keele
MIFF file (`F`, used as `F²` in the angular law). The source is Ag Kα1 at
22.162917 keV. The CPU and Metal paths use the same xorshift32 streams.

This first-stage kernel is **not** a port of Geant4. First Compton/photoelectric and second
interactions are counted but not propagated. Detector energy response,
secondaries, electron transport, arbitrary voxel geometries, and 2D pixel
images remain to be implemented. The present benchmark must not be used as
the forward model for composition or `S(q)` inversion.

## Reproduce

From `tutorial/saxs_keele`:

```sh
swiftc -O -framework Metal gpu_transport/single_rayleigh.swift -o gpu_transport/single_rayleigh
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python gpu_transport/benchmark.py --cases 100 --photons 10000000
/opt/homebrew/Caskroom/miniconda/base/envs/eosdx13/bin/python gpu_transport/benchmark_g4.py --photons 100000 --metal-photons 1000000 --threads 8
```

The Geant4 comparison requires the already built
`geant4-xray-diffraction/build/saxs_keele/saxs` executable. Its temporary
ROOT files are deleted immediately after reading; the retained outputs are
`gpu_transport/results/*.csv`, `*.json`, `*.npz`, and `*.png`.

`SAXS_DUMP_XS=1` enables an opt-in Geant4 printout of process-specific
macroscopic cross sections for the current material and air. At 22.162917 keV
for the baseline breast mixture, Geant4 reported (mm⁻¹): photoelectric
0.0250599, Compton 0.018054, Rayleigh 0.00651873. The latter two are not
interchangeable with arbitrary material-independent coefficients.

## Measured result on this M4 Pro

- 100 CPU/Metal geometries × 10 million incident photons each: median Metal
  kernel speedup 6.49× over the equivalent single-thread CPU kernel. Mean
  kernel times were 0.272 s (CPU) and 0.041 s (Metal) per 10 million.
- Same-seed CPU/Metal: only 74 total class-count L1 and 229 q-bin L1 counts
  differed across all one billion histories. These differences arise at
  floating-point boundaries; exact bitwise equality is not required.
- 100 independent Geant4 geometries × 100,000 photons: direct and single-
  Rayleigh yields differed from the CPU/Metal predictions by +0.297% and
  +0.279% in pooled totals. The pooled 256-bin q-profile gave reduced
  χ² ≈ 0.90 with the appropriate unequal-sample Poisson variance.
- A separate 3-million-photon 50 mm Geant4 run gave 248,894 direct and
  36,717 single-Rayleigh hits versus 249,261 and 36,472 on Metal; the
  256-bin q-profile reduced χ² was 1.04.

The 100 cases vary slab thickness (10–60 mm), air gap (60–180 mm), source
radius, focal distance and material density. The **molecular composition is
fixed** to the `keele_breast_g50.dat` mixture; later multi-collision and
composition audits are described above.

Reference implementation for a CUDA rather than Metal GPU:
[FDA MC-GPU](https://github.com/DIDSR/MCGPU). Its source is useful for
transport design, but cannot run unmodified on Apple Silicon.
