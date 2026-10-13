# Third-Party Notices

## SAM2-UNet

The adapted baseline and decoder source in `models/baseline/` and `models/full/` is derived from [WZH0120/SAM2-UNet](https://github.com/WZH0120/SAM2-UNet), distributed under Apache License 2.0. The license text is retained at `third_party/licenses/SAM2-UNet-Apache-2.0.txt`. Source modifications and formal experiment provenance are documented in `docs/source_provenance.md`.

## SAM 2

The runtime dependency is installed from the official [facebookresearch/sam2](https://github.com/facebookresearch/sam2) repository at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, under Apache License 2.0. Its license text is retained at `third_party/licenses/SAM2-Apache-2.0.txt`.

The private training environment used a different SAM2 fork whose redistribution terms could not be verified. That fork is not included. Checkpoint/numerical compatibility with the official dependency has not been certified.
