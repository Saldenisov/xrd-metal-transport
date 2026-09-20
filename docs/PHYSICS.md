# Physics and units

## Question and model boundary

The forward model asks how a specified, isotropic molecular-interference
scattering law appears after photon attenuation, repeated scattering, finite
sample geometry, air transport and detector integration. It does **not** infer
atomic coordinates from a diffraction curve. This distinction matters because
the input form factor already contains the nanoscale structure; the Geant4 or
Metal geometry describes millimetre-scale sample and instrument boundaries.

Both engines propagate X-ray photons through photoelectric absorption,
incoherent Compton scattering and coherent Rayleigh scattering. The implemented
Metal geometry is one homogeneous rectangular sample box in air with an ideal
planar photon-entrance detector. Geant4 remains the reference implementation
for this comparison, not an oracle for unmeasured sample properties.
`geometry.py` supplies the same dimensions to both engines. Geant4 constructs
its own solids and Metal navigates its own finite box; no general Geant4 solid
or navigation state is executed on the GPU.

## Molecular-interference form factors

The six-column input `data/xrd_components.txt` contains `q` in nm⁻¹ and
`I(q) = F(q)²/W` for fat, collagen and water. Geant4's
`G4PenelopeRayleighModelMI` accepts `q/(mₑc)` and `F(q)/√W`:

```text
q_Geant4 = q_nm⁻¹ × 3.8615926744 × 10⁻⁴
FF_Geant4 = √I(q)
q_nm⁻¹ = 4π sin(θ/2) / λ_nm
```

`prepare_keele_ff.py` checks the component tables, converts these units and
appends low- and high-q tails from Geant4 G4EMLOW molecular form-factor tables.
This prevents endpoint clamping when Geant4 integrates the full angular
cross section. The measured ranges are 0.09925–30.00878 nm⁻¹ for fat (449
points), 0.42678–29.90264 nm⁻¹ for collagen (181 points), and
0.76029–29.97690 nm⁻¹ for water (129 points). The table's immediate
provenance is the Keele project; the G4EMLOW tails trace to the published
tissue tables discussed by [Tartari et al.](https://doi.org/10.1088/0031-9155/47/1/312)
and [Paternò et al.](https://doi.org/10.1088/1361-6560/aba7d2).

For a mass-fraction mixture, the current model uses

```text
S_mix(q) = Σᵢ aᵢ Sᵢ(q),       Σᵢ aᵢ = 1
1/ρ_mix = Σᵢ aᵢ/ρᵢ.
```

`S` denotes `F²/W` here. This rule omits inter-component interference terms.
It is a testable material approximation, not a universal mixing law.

## Metal transport mapping

1. `SAXSRunAction.cc` exports Geant4 macroscopic photoelectric, Compton and
   Rayleigh cross sections at six energy nodes, plus the actual Penelope
   oscillator strength, ionization energy and Hartree-factor triples for
   sample and air. `prepare_multi_physics.py` stores them in JSON.
2. `Sources/MetalTransport/` validates the JSON and forms a 4096-bin angular
   Rayleigh CDF for every energy node. Its sample law is proportional to
   `(1 + cos²θ) × F(q)² × sinθ`. The air law uses the nitrogen and oxygen
   atomic form-factor tables from `G4LEDATA`. Metal interpolates cross
   sections logarithmically between nodes after Compton energy loss.
3. `Metal/` assigns one Philox4x32-10 stream to each photon
   history. It draws exponential flights, tests the six faces of the sample
   box, permits exit and re-entry, and continues transport in air until
   absorption, world exit or detector entrance.
4. For Compton scattering, the kernel samples shell-aware angle and outgoing
   energy with a Metal translation of the Penelope-2008 final-state algorithm
   in Geant4 11.4.2 `G4PenelopeComptonModel.cc`. The older calibrated angular
   CDF and free-electron shift remain a fallback for legacy fixtures without
   shell tables. The current examples use shell tables.
5. The detector image stores entrance counts, not deposited energy. The
   category is based on interactions **inside the sample**: direct, one
   Rayleigh, multiple Rayleigh or any Compton. Air interactions are tracked
   separately. The CPU and GPU images are integrated by the same
   XRD-preprocessing/pyFAI operation and geometry.
6. In radial-output mode, the same sparse pyFAI pixel-splitting operator is
   applied on the GPU. Each pixel contributes to at most three bins. The
   image-output path remains available for detector-level validation.

Metal uses an angular CDF with linear form-factor interpolation, while Geant4
MI uses RITA sampling and interpolation in log(q²)/log(F²). This algorithmic
difference is a plausible source of remaining multi-scatter residuals, but
the existing tests do not isolate it. The six-node cross-section table is
another approximation away from its exact source-energy node.

## Geometry and signal conventions

The main water parity case uses a 100 mm thick, 120 × 120 mm water box,
22.0220601 keV photons (λ = 0.0563 nm), a parallel uniform circular beam of
100 µm diameter, 100 mm downstream air and an ideal 1000 × 1000 detector with
100 µm pixels. The 1D profile uses an effective 150 mm distance from the
sample mean scattering plane, `q = 1–30 nm⁻¹`, 200 points and
`correct_solid_angle=False`.

The mixture case uses 45% water, 45% fat and 10% collagen by mass, a 50 mm
thick finite box, focused 22.162917 keV Ag Kα1 photons, 110 mm downstream
air and the same ideal detector. Its profile uses a 135 mm mean-plane
distance and 256 q points over 1–30 nm⁻¹.

The separate colleague-water comparison uses 20, 30 and 40 mm water slabs.
Its reference joblib supplies wavelength and 500 mm entrance-plane distance,
but lacks detector pixel geometry, mask and incident flux. The 1000² detector
and ≈240 µm pixel pitch were inferred from its q maximum. Thus comparisons
to those profiles test shape under a stated geometry assumption, not
independently calibrated absolute intensity.

## Limits

- The MIFF is isotropic: oriented collagen anisotropy is outside this model.
- Metal omits electron transport, fluorescence, voxels and detector response.
- The component mixture omits interfacial cross terms.
- The source is monochromatic and the detector ideal in the saved parity runs.
- Geant4/Metal channel equivalence remains unproven at multi-billion-history
  precision; see [validation](VALIDATION.md).
