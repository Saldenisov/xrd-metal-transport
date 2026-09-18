# Checks before public release

- Confirm ownership and redistribution rights for the Keele six-column
  `data/xrd_components.txt` and the derived `keele_*.dat` MIFF files. If
  rights are limited, replace them with licensed public examples and keep
  private validation tables outside the public repository.
- Select a license for original Swift, Metal and Python additions. Preserve
  Geant4 headers, the full `GEANT4_LICENSE` and its required attribution
  notice for the copied/adapted SAXS example. Check whether any further
  license notices apply to redistributed data.
- Decide authorship and citation metadata; no software DOI or journal claim
  has been assigned.
- Rerun an independent high-statistics water reference with a retained
  profile covariance and resolve or bound the −0.229% direct-channel residual.
- Record Geant4, G4EMLOW, Swift, macOS, pyFAI and XRD-preprocessing versions,
  source commit, seed policy, host model and wall-clock timing conditions in
  a release manifest.
- Review the historical JSON records for private paths and verify that no
  patient-level or confidential colleague material is present.
