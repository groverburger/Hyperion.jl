# Cross-vendor bit-exact verification — 2026-05-01

Snapshot of the three-backend audit run on the post-precision-fix kernel
(commit `300f557` on branch `1m-shadows`), before adding the dual-DEM
farfield handoff. Result: **fully bit-exact across Apple Silicon CPU,
Windows x86 CPU, and NVIDIA CUDA**.

For the *what / why / how* of cross-vendor determinism in this codebase,
see `cross-vendor-determinism.md`. This file is the audit trail for one
specific run.

## Backends covered

| Backend dir | Platform                  | Toolchain                                         |
|-------------|---------------------------|---------------------------------------------------|
| `cpu/`      | Apple Silicon arm64 (Mac) | Julia 1.11.5, KernelAbstractions 0.9.41           |
| `win_cpu/`  | x86_64 Windows            | Julia 1.11.5, KernelAbstractions 0.9.41           |
| `win_cuda/` | x86_64 Windows + NVIDIA   | Julia 1.11.5, KernelAbstractions 0.9.41, CUDA 5.9.0 |

All three ran the same `scripts/bitexact_test.jl` over the canonical
20-timestamp LDEM region (`origin=(8960,18432)`, `size=512×896`).

## How the audit was produced

1. **Mac, before drive moved:** `test/bitexact.jl` + legacy 1m site-only SHAs
   were freshly pinned on Apple Silicon (the canonical pin).
2. **Windows machine** (drive plugged into a Windows PC with NVIDIA GPU):
   ran `scripts/cross_vendor_test.ps1` which:
   - Stage 0: `Pkg.instantiate()`
   - Stage A: `Pkg.test()` — full regression on Windows CPU, including
     the 20m LDEM bit-exact pin and the 1m site DEM pin.
   - Stage B: `bitexact_test.jl` with `HYP_BACKEND=cuda`
   - Stage C: `bitexact_test.jl` with `HYP_BACKEND=cpu`
   Then renamed `data/outputs/bitexact/cuda/` → `win_cuda/` and
   `data/outputs/bitexact/cpu/` → `win_cpu/` so they wouldn't be
   overwritten by the upcoming Mac run.
3. **Mac, drive returned:** ran `scripts/bitexact_test.jl` to repopulate
   `data/outputs/bitexact/cpu/`.
4. **Comparison:**
   - `scripts/diff_bitexact_shas.jl` — SHA-table all-pairs diff.
   - `scripts/diff_bitexact_pixels.jl` — per-pixel UInt8 diff of the
     `sun`, `dsn`, `sun_rgb`, `dsn_rgb` raw buffers.

## What passed

### A) Pinned regression tests on Windows CPU

Stage A of the PowerShell driver ran `Pkg.test()` on Windows. Both
pinned suites passed:

- `test/bitexact.jl` — 20-timestamp 20m LDEM SHAs identical to the
  Apple-Silicon pin.
- Legacy 1m site-only pin — 256×256 1m nobile_1m.tif SHAs identical to the
  Apple-Silicon pin.

This proves Windows x86 CPU Float32 is byte-identical to Apple Silicon
CPU Float32 for the canonical kernel runs.

### B) SHA all-pairs (`diff_bitexact_shas.jl`)

| Pair                         | SHAs matching |
|------------------------------|---------------|
| `cpu` ↔ `win_cpu`            | 280 / 280     |
| `cpu` ↔ `win_cuda`           | 280 / 280     |
| `win_cpu` ↔ `win_cuda`       | 280 / 280     |

Each side covers 20 timestamps × 14 SHA keys per timestamp (data
sub-block, azel precompute, sun, dsn, de, d_0…d_7).

### C) Pixel-level diff (`diff_bitexact_pixels.jl`)

19 timestamps were common across all three backends (one timestamp
present on Mac but not yet completed on Windows side at the moment of
that run). For every timestamp and every channel:

- `sun`     (UInt8, 512×896)
- `dsn`     (UInt8, 512×896)
- `sun_rgb` (UInt8, 3×512×896 — palette-applied, what the user sees)
- `dsn_rgb` (UInt8, 3×512×896)

result:

- 228 / 228 pairwise pixel-buffer comparisons reported `BIT-EXACT`
  (zero differing bytes).

## Kernel state being verified

This audit covers the kernel as of commit `300f557`, which includes:

- dz-cancellation fixes (`bfda7f6`, `533f9ed`, `facfe85`) — Float32
  catastrophic cancellation in `dz` was the root cause of the concentric
  ring artefact, fixed by reordering to fma(q_elev_m − telev_m, 0.001f0,
  fma(scale, two_u2, −qz_pos)).
- Mipmap-pool / skip / d-alignment fixes (`b048496`, `3a57940`):
  - 3×3 halo'd max/min pool (was 2×2) so bilinear footprint never reads
    past pooled cell.
  - `n_skip` loop accumulates `d` via `d += base_step` instead of
    multiplying, keeping skip-step `d` values bit-identical to the
    no-skip version.
  - Removed the `cmin` early-termination shortcut (it was inconsistent
    with the level-0 path's `max_num` accounting).
- Default `mipmap_base = 100` for the site driver (`9ecfb43`).
- LDEM + 1m site SHAs re-pinned at `122e31d`.

## Reproducing later

```bash
# Mac: regenerate cpu/ baseline
julia --project scripts/bitexact_test.jl

# Windows (drive on Windows PC): regenerate win_cpu/ + win_cuda/
.\scripts\cross_vendor_test.ps1

# Compare (any machine with all three dirs present)
julia --project scripts/diff_bitexact_shas.jl
julia --project scripts/diff_bitexact_pixels.jl
```

The diff scripts auto-discover any subdirectory of
`data/outputs/bitexact/` that contains a `SHAs.txt`, so additional
backends (e.g. AMD ROCm) can be dropped in without script changes.

## What this does NOT cover

- The 1m site DEM is pinned only on Apple Silicon CPU + Windows CPU
  (Stage A). It was not run as `bitexact_test.jl` on CUDA — that test
  uses the LDEM driver, not the site driver. Site cross-vendor coverage
  remains a "tested at one CPU pin level" property; expanding it to a
  CUDA + multi-timestamp regression is a future task.
- Performance is not measured here; only output equivalence.
