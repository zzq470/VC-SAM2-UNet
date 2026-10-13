# Source Provenance

The model and protocol source was selected from the final D10 v3 experiment export and checked against the formal XEC source manifests. Private absolute paths, split manifests, run logs, checkpoints, and case-level data are intentionally omitted.

| Repository component | Source role | Verified XEC source SHA-256 |
|---|---|---|
| `models/baseline/SAM2UNet.py` | Formal SAM2-UNet baseline model | `52cad0313cdffce96706fba8c0ffbdaeab8edcedb76602525cb702da2b32b19c` |
| `models/full/SAM2UNet.py` | Formal view-conditioned SAM2-UNet model | `29fa56b4dd929ff8ebda551c4d2aee9bad6d2e340560bb400135796c677cdc35` |
| `models/full/metadata_film_global_dpbifpn.py` | Full-model Metadata-FiLM + DPBiFPN composition | `bb5af66f07646ec830b6ef464863680e939c2432d747231879861ab754118bbf` |
| `models/full/selective_dpbifpn_common.py` | Selective Dual-Path BiFPN implementation | `03d2d8ecf2256923e498eca93cf176a5b3c1d7e2c31f5dec283fe0f42de74325` |
| `data_processing/dataset_d10_multilabel.py` | 14-channel dataset and partial-label handling | `5de8d15c3ec118e55734e9286cbe56340de72f56c076e8813331df832687d253` |
| `models/view_classifier/view_models.py` | View-classifier model definitions | `9fb3c9f8a57e17972b465794ad54a5a8a72a435a194883fc62996e45e3e1a1db` |

The first table lists verified XEC source hashes. Repository hashes and adaptation notes are in `docs/release_manifest.csv`; five rows are byte-identical to their source hashes. Package-qualified imports were applied to the full-model composition and A1/A2 model wrapper after a CLI import test exposed a collision between baseline and full files both named `SAM2UNet.py`; these edits do not change model parameters or state-dict keys. Training path arguments were made portable. The A1/A2 trainer is adapted from the formal source. The fixed-test runner is a privacy-safe release wrapper aligned with the audited Stage T1 protocol, not a byte-identical copy of the original test script; it omits case identifiers and prediction exports. Source and release hashes/status are recorded separately in `docs/release_manifest.csv`.

## SAM 2 Dependency Note

The original private training environment resolved `sam2.build_sam` through a third-party fork whose license could not be verified from its repository metadata. That fork is not redistributed here. The repository instead documents a pinned official Meta SAM 2 installation, which is Apache-2.0 licensed. This source substitution has not been validated against the original trained environment; exact checkpoint compatibility and numerical reproduction remain conditional on confirming the authorized original runtime source or a validated equivalent.
