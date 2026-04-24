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

Library deps are backend-agnostic (no Metal / CUDA in `Project.toml`).
Fresh clone to running tests is three commands:

```
git clone …
cd JuliaMapbuilder
git checkout live-only
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
```

The test command auto-downloads the 1.85 GB LDEM from the PDS on the
first run (SHA-verified), then runs the full 20-timestamp bit-exactness
regression. Subsequent test runs are a no-op on the data — no re-download.

If you want to pre-seed the data without running tests yet (or to also
derive the Nobile GTiff used by diagnostic scripts):

```
julia --project scripts/fetch_test_data.jl
```

Both paths call the same `JuliaMapbuilder.ensure_ldem!()` /
`ensure_test_data!()` functions, so they're safe to mix.

### GPU backends (optional)

The CPU KernelAbstractions backend works out of the box. For GPU runs,
add the matching backend to your **global** Julia env (`@v1.x`) so
`using Metal` / `using CUDA` resolves without polluting this project's
Project.toml:

```
# Apple Silicon
julia -e 'using Pkg; Pkg.add("Metal")'

# NVIDIA (Linux / Windows)
julia -e 'using Pkg; Pkg.add("CUDA")'

# AMD
julia -e 'using Pkg; Pkg.add("AMDGPU")'
```

## Usage

To verify bit-exactness end-to-end, run the tests:

```
julia --project -e 'using Pkg; Pkg.test()'
```

This runs `test/bitexact.jl`, which exercises the full pipeline on the
KA CPU backend at 20 representative timestamps (full 896×512 each) and
verifies:
- SHA-256 of every intermediate (sun, dsn, palette-applied RGB,
  Float32 diagnostics, azel precompute) against a hardcoded known-good
  table, and
- pixel equality of the decoded PNG against 40 committed reference
  fixtures in `test/fixtures/bitexact/`.

Runtime: ~10-15 minutes. The same invariants hold bit-for-bit on
Apple Metal and NVIDIA CUDA — see
[`docs/cross-vendor-determinism.md`](docs/cross-vendor-determinism.md).

For a cross-vendor forensic audit (actual Metal or CUDA hardware +
raw .bin buffers), use `scripts/bitexact_test.jl`:

```
JM_BACKEND=metal julia --project scripts/bitexact_test.jl   # or cuda / cpu
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

- `scripts/bitexact_test.jl` — cross-vendor forensic harness: 20
  timestamps × full 896×512 × backend-of-your-choice, SHAs + raw .bin
  buffers + PNGs. Pair with `scripts/diff_bitexact_{shas,pixels}.jl`
  for N-way pairwise comparison across `data/outputs/bitexact/`.
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
mipmap pyramid shape, and — the canonical cross-platform regression —
20-timestamp bit-exactness check on the KA CPU backend, including
SHA-256 of all intermediates + pixel-exact comparison against 40
committed PNG fixtures. See
[`docs/cross-vendor-determinism.md`](docs/cross-vendor-determinism.md)
for the theory and verification protocol.
