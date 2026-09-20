# Provenance, attribution and code ownership

This product includes software developed by Members of the Geant4 Collaboration
(http://cern.ch/geant4). The copied/adapted SAXS example is redistributed under
the [Geant4 Software License](../GEANT4_LICENSE). The example in this repository
is **`saxs_keele`**, a modified derivative of the
[Geant4 11.4.2 SAXS example](https://github.com/Geant4/geant4/tree/v11.4.2/examples/extended/exoticphysics/saxs),
and must not be represented as the original Geant4 toolkit.

| Component | Origin and change |
|---|---|
| `tutorial/saxs_keele/include`, `src`, `saxs.cc` | Geant4 11.4.2 `examples/extended/exoticphysics/saxs`; geometry, event scoring, image-only CSV mode, cross-section/oscillator export and Keele material paths were added or changed locally. Unchanged example files retain their Geant4 headers. |
| `tutorial/saxs_keele/prepare_keele_ff.py` and material macros | Project adaptation of the Geant4 MI input convention to Keele water/fat/collagen tables. |
| `gpu_transport/Metal` | Original Metal implementation for this project. Its shell-aware Compton final-state equations were translated from [Geant4 11.4.2 `G4PenelopeComptonModel.cc`](https://github.com/Geant4/geant4/blob/v11.4.2/source/processes/electromagnetic/lowenergy/src/G4PenelopeComptonModel.cc). Its parallel independent-history architecture was informed by [FDA MC-GPU v1.3](https://github.com/DIDSR/MCGPU), but the MC-GPU CUDA source and its PENELOPE-2006 shell tables were not copied. Molecular Rayleigh uses the Keele MIFF, not MC-GPU's atomic tables. |
| Philox4x32-10 implementation | Algorithm and known-answer vectors from [Random123](https://github.com/DEShawResearch/random123); see `philox_metal_kat.swift`. |
| Radial profile integration | External [XRD-preprocessing](https://github.com/Eos-Dx/XRD-preprocessing) and [pyFAI](https://github.com/silx-kit/pyFAI), called consistently for both detector images. |
| `data/xrd_components.txt` and `tutorial/saxs_keele/data/keele_*.dat` | Keele project input and derived MIFF files. Public redistribution rights and exact source attribution require review before release. |
| `data/pure_cross_sections.json` | Six-node pure-component photoelectric, Compton and Rayleigh tables exported by this project's Geant4 11.4.2 SAXS application; the original console logs are not distributed. |

For a file-by-file comparison of the modified SAXS application against the
downloaded Geant4 release, use:

```bash
diff -ru /path/to/geant4-11.4.2/examples/extended/exoticphysics/saxs \
  tutorial/saxs_keele
```

The additional GPU implementation is **not** Geant4 running on a GPU.
The Geant4 reference and Metal implementation independently sample histories
from related, but not identical, numerical physics procedures.

## Scientific references

1. [Geant4 11.4.2 source and license](https://geant4.web.cern.ch/download/)
   and [Physics Reference Manual, Penelope Rayleigh](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/gamma_incident/elastic/penelope_rayleigh.html).
2. G. Paternò et al., “Geant4 implementation of inter-atomic interference
   effect in small-angle coherent X-ray scattering for materials of medical
   interest,” *Physica Medica* **51**, 64–70 (2018).
   [doi:10.1016/j.ejmp.2018.04.395](https://doi.org/10.1016/j.ejmp.2018.04.395).
3. G. Paternò et al., “Comprehensive data set to include interference effects
   in Monte Carlo models of x-ray coherent scattering inside biological
   tissues,” *Physics in Medicine & Biology* (2020).
   [doi:10.1088/1361-6560/aba7d2](https://doi.org/10.1088/1361-6560/aba7d2).
4. A. Tartari et al., “Updating of form factor tabulations for coherent
   scattering of photons in tissues,” *Physics in Medicine & Biology* **47**,
   163–175 (2002). [doi:10.1088/0031-9155/47/1/312](https://doi.org/10.1088/0031-9155/47/1/312).
5. A. Badal and A. Badano, “Accelerating Monte Carlo simulations of photon
   transport in a voxelized geometry using a massively parallel graphics
   processing unit,” *Medical Physics* **36**, 4878–4880 (2009).
   [doi:10.1118/1.3231824](https://doi.org/10.1118/1.3231824).
6. J. K. Salmon et al., “Parallel Random Numbers: As Easy as 1, 2, 3,”
   *SC11* (2011). [doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405).
