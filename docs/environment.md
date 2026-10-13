# Environment Record

## Original XEC training environment

Read from the existing `sam2unet_xec_py310` environment on the XEC login node; no training job or GPU task was launched for this check.

| Component | Version / source |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| PyTorch CUDA build | 12.4 |
| torchvision | 0.20.1+cu124 |
| SAM2 source | Project-local Polyp-SAM-2 fork, commit `c2549321706fa290b10f9e94227b71ebf7512524`; not redistributed because its license could not be verified |

The official Meta SAM 2 dependency pinned in the installation command is an Apache-2.0 alternative, **not a verified binary or numerical equivalent** of the training fork above. `sam2.build_sam` must be importable after installation, and the `sam2_hiera_l.yaml` configuration plus authorized Hiera Large initialization weights must be supplied separately. Strict checkpoint compatibility has not been certified with the official alternative.

The current local Windows Python is 3.12.10 and does not have PyTorch, torchvision, matplotlib, Hydra, or SAM2 installed. Use the recorded Linux environment or install compatible dependencies before invoking model/training/evaluation commands.
