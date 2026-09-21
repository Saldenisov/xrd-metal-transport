# Public release and reuse status

The public repository supports scientific review and exact reruns of the
included examples. Public visibility does not itself grant a reuse license.

- Copied and adapted Geant4 SAXS files retain their headers and are distributed
  under `GEANT4_LICENSE` with the required attribution.
- No separate license is granted for original Swift, Metal, CUDA/C++ and Python files.
- No separate license is granted for `data/xrd_components.txt` or derived
  `keele_*.dat` molecular-interference form-factor tables. Their scientific
  provenance is documented in `PROVENANCE.md`. Contact the repository owner
  before redistribution or incorporation into another work.
- `CITATION.cff` records software authorship and the repository citation. No
  software DOI or associated journal publication is claimed.
- The maintained example manifest records source commit, seeds, software
  versions, host class and timing conditions. Each case records input hashes.
- Generated summaries contain no patient data. The colleague-supplied joblib
  used for one historical comparison is not distributed.

The unresolved physics limits in `VALIDATION.md` remain part of the release.
Public availability does not change their interpretation.
