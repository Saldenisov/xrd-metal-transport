# Performance: definitions, measurements and limits

## Question

The performance question is narrowly defined: how much faster can the
restricted Metal implementation repeat a forward X-ray diffraction transport
calculation that has first been checked against the Geant4 CPU reference?
The measurements do not describe arbitrary Geant4 applications, detector
simulation, inverse fitting or clinical throughput.

For comparable complete runs, the reported ratio is

```text
end-to-end speedup = Geant4 process wall time / Metal process wall time.
```

Both terms include process startup and requested output for their respective
programs. A second diagnostic divides complete Geant4 wall time by the warm
Metal transport-and-reduction interval. That value isolates repeated GPU work,
but is not a symmetric end-to-end comparison.

## Maintained 100-million-history suite

The primary evidence is the public 12-case suite in
[`examples/xrd_suite/results_100m`](../examples/xrd_suite/results_100m/RESULTS.md).
Every case uses 100 million incident photons in Geant4 and another independent
100 million in Metal. Geant4 uses seven CPU threads. The recorded host is a
MacBook Pro with an Apple M4 Pro, 14 CPU cores and 48 GB RAM.

| Observation | Geant4 | Metal | End-to-end ratio |
|---|---:|---:|---:|
| Fastest ratio: fat/collagen 50/50, 50 mm | 354.2 s | 1.126 s | 314× |
| Slowest ratio: water, 10 mm | 313.2 s | 1.690 s | 185× |
| Range over all 12 cases | 286.0–354.2 s | 0.959–1.690 s | 185–314× |

The values come directly from
[`summary.csv`](../examples/xrd_suite/results_100m/summary.csv); software and
hardware metadata are frozen in
[`run_manifest.json`](../examples/xrd_suite/results_100m/run_manifest.json).
The range is descriptive for this machine, configuration and workload.

## Where the 300–400× statement applies

Some supported tasks reach this range under their recorded timing definition:

| Saved benchmark | Photons | CPU threads | Compared intervals | Ratio |
|---|---:|---:|---|---:|
| Fat/collagen 50/50 suite case | 100 M | 7 | complete Geant4 / complete Metal | 314× |
| Saved `g50` Penelope transport; composition not recorded | 20 M | 14 | complete Geant4 / warm Metal transport and reduction | 364× |
| 45/45/10 Penelope transport | 20 M | 14 | complete Geant4 / warm Metal transport and reduction | 372× |
| Detector-image benchmark | 20 M | 8 | complete Geant4 CSV / complete Metal raw image | 400.5× |

The two 364–372× values exclude Metal startup from the denominator while the
Geant4 numerator remains a complete process time. The corresponding symmetric
end-to-end ratios are 248.6× and 253.3×. They quantify the repeated warm GPU
calculation rather than total latency.

The 400.5× record includes startup on both sides and complete detector-image
transport, but its output formats differ: Geant4 writes CSV and Metal writes a
raw binary image. Serialization cost is therefore a confound. The JSON record
also lacks a hardware identifier. This measurement supports the statement that
approximately 400× was observed for one image-output task; it is not a portable
or format-matched performance guarantee.

Exact records:

- [20 M detector-image benchmark](../tutorial/saxs_keele/gpu_transport/results/multi_transport/fair_benchmark.json)
- [20 M `g50` Penelope transport](../tutorial/saxs_keele/gpu_transport/results/g50_penelope_transport_20m/validation.json)
- [20 M 45/45/10 Penelope transport](../tutorial/saxs_keele/gpu_transport/results/trial_45_45_10_penelope_transport_20m/validation.json)

## Mechanism of acceleration

The measured acceleration has a direct computational explanation:

1. Photon histories are independent and map to many concurrent Metal threads.
2. Counter-based Philox streams avoid shared random-number state between
   threads.
3. The supported geometry uses analytic intersections with one rectangular
   box instead of the general Geant4 solid and navigation framework.
4. Cross sections, Rayleigh angular tables and Penelope shell data are prepared
   once and placed in compact GPU buffers.
5. Detector channels use compact integer tallies. Radial-output mode can apply
   the pyFAI sparse integration operator on the GPU and return only the reduced
   profile.

These choices accelerate the specified forward problem by reducing generality.
They do not imply that the full Geant4 physics and geometry stack has been
ported to either GPU backend.

## CUDA-to-Metal diagnostic

The CUDA backend has one compiler check on the recorded Apple M4 Pro host. For
10 million g50 histories, CuMetal 0.5.0 required 3.914 s for synchronized
kernel launches after a discarded compilation warm-up. Native Metal reported
0.0460 s of GPU time, giving a diagnostic ratio of 85.1. CuMetal GPU tracing
confirmed the same order of kernel runtime. The intervals are not perfectly
symmetric because CuMetal uses synchronized host wall time and native Metal
uses command-buffer GPU timestamps.

Complete timed transport was 4.004 s for CuMetal and 0.133 s for native Metal,
a ratio of 30.2. The translated MSL is functionally close but inefficient for
this branch-heavy transport kernel. This confirms the platform choice: native
Metal remains the maintained performance path on Apple silicon, while the
CuMetal run checks common CUDA source behavior.

No NVIDIA device was available on this host. Native CUDA speedup, kernel time
and cross-platform reproducibility remain unmeasured; they must be reported
from an NVIDIA system with GPU, driver, CUDA toolkit, compiler flags and output
mode recorded. CuMetal timing must not be presented as native CUDA timing.

The record is
[`cuda_cumetal_validation_10m.json`](../tutorial/saxs_keele/gpu_transport/results/cuda_cumetal_validation_10m.json).

## Factors that change the ratio

- Apple GPU and CPU model, thermal state and concurrent system load;
- Geant4 build options, compiler, number of CPU threads and thread affinity;
- photon count and the relative contribution of startup and Metal compilation;
- sample thickness, composition and number of interactions per history;
- full detector image versus GPU radial reduction;
- text, compressed or binary output and filesystem performance;
- additional geometry, detector response or secondary-particle physics.
- backend and compiler: native Metal, native CUDA and CUDA translated by
  CuMetal have different startup, memory-transfer and math behavior.

Performance and physical agreement are separate tests. A faster run is useful
only within the regime where the Metal result remains compatible with the
reference observables. The present evidence is bounded by the compositions,
energies, thicknesses, finite-box geometry and ideal detector described in
[VALIDATION.md](VALIDATION.md).
