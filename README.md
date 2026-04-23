# JuliaMapbuilder — live shadow generation (GPU, cross-platform)

Lunar shadow-map generator. This branch (`live-only`) has only the
live raycasting GPU pipeline — the precomputed-horizons codepath and
the CPU live codepath were removed. See `docs/algorithms.md` on the
`master` branch for the historical comparison.

## Bit-exactness across hardware

The GPU kernel uses only IEEE-754-mandated ops (+, −, *, /, sqrt, fma)
plus LUT-based transcendentals. **The output is bit-identical on any
backend KernelAbstractions supports**: Apple Metal, NVIDIA CUDA, AMD
ROCm, Intel oneAPI, and the CPU fallback. The SHA-256 of an output
directory is a deterministic function of (DEM, SPICE kernels, git SHA).

## Install

Library deps are backend-agnostic (no Metal / CUDA in `Project.toml`):

```
git clone …
cd JuliaMapbuilder
git checkout live-only
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Then add the backend that matches your hardware to your **global**
Julia env (`@v1.x`). `using Metal` / `using CUDA` will find it from
there without polluting this project's Project.toml.

```
# Apple Silicon
julia -e 'using Pkg; Pkg.add("Metal")'

# NVIDIA (Linux / Windows)
julia -e 'using Pkg; Pkg.add("CUDA")'

# AMD
julia -e 'using Pkg; Pkg.add("AMDGPU")'
```

## Usage

Scripts in `scripts/` have a backend selection block at the top —
uncomment the one matching your hardware. Example:

```
# scripts/smoke_test_gpu_live.jl
using Metal
const BACKEND    = Metal.MetalBackend()
const DEVICE_ARR = Metal.MtlArray
```

For NVIDIA, comment that out and use:

```
using CUDA
const BACKEND    = CUDA.CUDABackend()
const DEVICE_ARR = CUDA.CuArray
```

Then run:

```
julia --project scripts/smoke_test_gpu_live.jl
```

The smoke test prints the SHA-256 of a single 128×128 frame. **You
should see the same SHAs on every backend** — that's how you verify
bit-exactness.

Known-good SHAs (2026-01-01T00:00:00, origin 9000,18600, 128×128):
```
sun: a468f36f2fa98c87fbf301a31580ffa1c3f9b89c66043c9eee687937188935a7
dsn: 215894806c716a15949a243ee1a83fb31f32376fe8e5d90b6eb964144073fa93
```

## Library API

```julia
using JuliaMapbuilder
using Metal   # or CUDA, AMDGPU

ldem = load_ldem("data/inputs/ldem_80s_20m.img")
init_spice("kernels")
max_mm, min_mm = build_ldem_mipmaps_minmax(ldem.data)

et = datetime_to_et(DateTime(2027, 6, 1, 0, 0, 0))
sun_pos   = Tuple(get_body_position(NAIF_SUN,   et))
earth_pos = Tuple(get_body_position(NAIF_EARTH, et))

sun, dsn = generate_live_shadow_frame_gpu(
    ldem.data, 8960, 18432, 512, 896,
    sun_pos, earth_pos, 0.0;
    max_mipmaps = max_mm, min_mipmaps = min_mm,
    backend     = Metal.MetalBackend(),
    DeviceArray = Metal.MtlArray,
    workgroup_size = 512,
)

save_indexed_png(sun, SUN_PALETTE, "sun.png")
save_indexed_png(dsn, DSN_PALETTE, "dsn.png")
```

## Scripts

- `scripts/smoke_test_gpu_live.jl` — single-frame bit-exactness check
- `scripts/bench_workgroup.jl` — sweep workgroup sizes {128, 256, 512}
- `scripts/generate_live_year.jl` — full year (2h cadence) with per-ts
  timing and SSIM vs an optional precomputed reference directory
- `scripts/fetch_test_data.jl` — pull the LDEM and SPICE kernels
- `scripts/azimuth_range.jl`, `scripts/select_test_timestamps.jl` —
  diagnostic helpers

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

Covers deterministic-math LUTs, Float32 stereographic projection,
mipmap pyramid shape. GPU byte-exactness is tested via
`scripts/smoke_test_gpu_live.jl` (since the GPU backend is caller-
supplied, not built into the package).
