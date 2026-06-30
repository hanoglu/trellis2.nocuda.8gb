# trellis2.c

`trellis2.c` is a native C + ggml + CUDA inference project for TRELLIS.2 image-to-3D generation.

The goal is to run the TRELLIS.2 pipeline without Python or PyTorch:

- encode an input image with DINOv3
- run stage1 sparse-structure denoising
- decode and visualize voxels at each stage1 step
- run stage2 shape latent denoising
- decode and visualize a mesh at each stage2 step

The main app is `trellis-live`, a raylib viewer that shows the generation process as it runs.

## Status

Implemented:

- CUDA-backed ggml runtime setup
- safetensors loading
- DINOv3 image conditioning
- stage1 sparse-structure flow and decoder
- stage2 shape flow
- shape decoder with FlexiDualGrid mesh extraction
- raylib voxel and mesh visualization
- custom CUDA kernels for the parts not covered by ggml

Still in progress:

- Python parity tuning
- performance optimization for sparse convolution and decode
- texture/PBR output
- full end-to-end model quality validation

## Dependencies

- CMake
- CUDA Toolkit
- a CUDA GPU
- Git submodules

`ggml` is included as a submodule from:

```sh
https://github.com/Wimacs/ggml.git
```

`raylib` is included as a submodule for visualization.

Clone with submodules:

```sh
git clone --recursive git@github.com:Wimacs/trellis2.c.git
```

## Build

```sh
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

Place the TRELLIS.2 and DINOv3 checkpoints next to the project, then run:

```sh
./build/trellis-live \
  --model ../TRELLIS.2-4B \
  --dino ../dinov3-vitl16-pretrain-lvd1689m \
  --image ../assets/example_image/T.png
```

`trellis-live` preloads model weights first, opens a raylib window, then visualizes stage1 voxels and stage2 meshes step by step.

## Useful Tools

- `trellis-live`: full image-to-3D live viewer
- `trellis-info`: inspect checkpoint coverage
- `trellis-stage1`: stage1 validation helpers
- `trellis-stage2`: stage2 tensor and decoder helpers
- `trellis-infer`: stage1 image-to-voxel debug path
- `trellis-voxel-viewer`: replay voxel snapshots
- `trellis-mesh-viewer`: replay mesh snapshots

## Layout

```text
include/       public C API
src/           inference, loaders, CUDA/ggml execution
src/custom_ops custom CUDA kernels
tools/         CLI tools and raylib viewers
tests/         unit tests
docs/          implementation notes
3rd/           ggml and raylib submodules
```
