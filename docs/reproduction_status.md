# Reproduction Status

The repository contains source-level entrypoints and the frozen paper protocol. It does not contain the D10 images/masks, split manifests, SAM2 Hiera initialization weights, trained segmentation checkpoints, or the frozen View-classifier checkpoint. Therefore a third party cannot reproduce the reported numbers from this repository alone.

The fixed-test metrics were verified against the final XEC SeedMax summary and component-ablation paper table. The ten-seed validation values are the recorded mean and sample standard deviation for the corresponding four configurations; fixed-test values use the checkpoint selected by validation. No score was converted algebraically between Dice and IoU.

The original runtime used a SAM 2 fork without a verifiable license declaration. It is excluded. A public, licensed SAM 2 dependency is documented, but its exact equivalence to that private runtime has not been tested. Full numerical reproducibility is therefore **not yet certified**.

No training, validation pass, or test inference was run while preparing this repository. Python bytecode compilation, non-clinical content scans, seven CPU-only synthetic contract/metric/FiLM/view-interface tests, and `--help` import checks for the baseline/full trainer, A1/A2 trainer, View-classifier trainer, and evaluator passed in the existing XEC Python 3.10.20 / PyTorch 2.5.1+cu124 / torchvision 0.20.1+cu124 environment. No SAM2-backed segmentation model was instantiated, no checkpoint was loaded, no image data was read, and CUDA was not used. The local Windows Python lacks PyTorch and therefore was not used for those runtime checks.
