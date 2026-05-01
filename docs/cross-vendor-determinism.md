# Cross-vendor bit-exactness: Metal ↔ CUDA

`docs/algorithms.md` documented seven bugs fixed to make the live shadow
pipeline bit-exact between CPU and Metal GPU on Apple Silicon. That
establishes *intra-vendor* determinism. Cross-vendor (Metal ↔ CUDA) is
a stricter bar: different LLVM backends compiling the same Julia source
make independent optimization choices, and several of those choices
flip rounding behavior at the 1-ULP level.

This doc covers the additional classes of divergence we discovered and
fixed while making the kernel bit-exact between Apple Metal and NVIDIA
CUDA at full 896×512 scale. Read the bugs section of `algorithms.md`
first — it establishes the foundation this extends.

## TL;DR

- **The `fma()` principle from algorithms.md #5 isn't strict enough.**
  Writing `fma(a, b, c)` where you have an `a*b + c` *expression* is
  necessary. It's also not sufficient: any `(a*b) ± c` reconstructible
  across connected statements is a platform-dependent fusion coin flip.
  Fix: wrap those too.
- **Every in-loop division and sqrt is a cross-vendor risk.** `div.approx.f32`
  / `sqrt.approx.f32` vs IEEE rn is a per-backend default that can vary
  by compile flag. Fix: eliminate from the hot path via algebraic
  rewrites (squared-form comparisons, orthonormality, module-const
  reciprocals).
- **Float32 scalars derived from JIT-time constants belong in module
  `const`, not in kernel args.** Constant propagation inlines them as
  immediates; kernel args go through Metal's indirect argument buffer
  (31-slot limit).

## Verification protocol

The canonical regression check is **`] test` / `julia --project -e 'using Pkg; Pkg.test()'`**.
It runs `test/bitexact.jl` on the KernelAbstractions CPU backend, which
(a) runs the IEEE-math guard, (b) computes every stage of the pipeline
for 20 representative timestamps, (c) asserts every SHA matches a
hardcoded known-good table, and (d) decodes the 40 committed PNG
fixtures and verifies pixel equality against the kernel's palette-applied
RGB output. A regression fails the test and points at specific
timestamps you can then visually diff via the reference PNG.

Runtime on CPU backend: ~10-15 min for the 20-timestamp regression.

For cross-vendor audits (actual Metal or CUDA hardware) and for
regenerating baselines after an intentional algorithm change,
`scripts/bitexact_test.jl` is the forensic harness. It produces the
same 20-timestamp SHA table the test checks against, plus raw .bin
buffers for byte-level diffing via `diff_bitexact_shas.jl` /
`diff_bitexact_pixels.jl`. To refresh committed baselines after a
deliberate change: run on every backend, confirm cross-vendor match,
copy `data/outputs/bitexact/metal/<ts>/{sun,dsn}.png` to
`test/fixtures/bitexact/<ts>_{sun,dsn}.png`, and update the
`KNOWN_GOOD` table in `test/bitexact.jl`.

The harness spans three parts: an IEEE-math guard, SHA/raw-buffer
production, and N-way pairwise diffing.

### IEEE-math guard

Before any frames are generated, `bitexact_test.jl` runs a tiny kernel
on the target backend that computes three known-tricky operations and
compares bit patterns against hard-coded IEEE-rn expectations:

| Op | Inputs | Expected bits |
|---|---|---|
| `fma(a, a, -1)` with `a = 1 + 2⁻¹²` | tests single-rounded FMA vs. naive mul-then-add | `0x3a000400` |
| `1.0f0 / 3.0f0` | tests IEEE-rn `/` (vs `div.approx.f32`) | `0x3eaaaaab` |
| `sqrt(2.0f0)` | tests IEEE-rn `sqrt` (vs `sqrt.approx.f32`) | `0x3fb504f3` |

Any mismatch hard-aborts before SHAs are written — so a compiler-flag
shift or math-mode change can't silently corrupt the audit output. The
host Julia CPU is self-checked against the same hard-coded values first,
so a broken host doesn't mask a broken backend.

### Production + cross-vendor shuttle

1. **`scripts/bitexact_test.jl`** runs 20 representative timestamps
   spread across 2027 × full 896×512 frames: night (full-frame
   twilight-skip), twilight (terminator crossing the frame, hardest case
   for bit-exactness), high-sun (long rays), plus a dozen more spread by
   month and hour to exercise varied sun/earth geometries. Emits SHA-256
   of every stage of the pipeline:

   - **`sun`, `dsn`** — UInt8 kernel output (the raw frames).
   - **`sun_rgb`, `dsn_rgb`** — palette-applied 3×H×W RGB bytes (the
     actual user-visible content). Cross-platform invariant by
     construction since UInt8 is bit-exact and the palette is a module
     `const`. End-to-end audit endpoint.
   - **`de`, `d_0..d_7`, `azel`** — Float32 diagnostic buffers for
     debugging.

   Plus raw `.bin` buffers to `data/outputs/bitexact/<backend>/<tag>/`
   for offline pixel-level diff. Toolchain triplet (julia, platform,
   KernelAbstractions, backend pkg) is recorded in the SHAs.txt header
   for regression localization.

   For each frame the test also writes the actual PNG file
   (`sun.png`, `dsn.png`) via `Hyperion.save_indexed_png`, immediately reads
   it back via `FileIO.load`, and verifies the decoded RGB bytes equal
   the in-memory `sun_rgb` / `dsn_rgb` hash. Hard-aborts on mismatch.
   This catches PNG-encoder regressions per-platform.

   PNG file bytes themselves are *not* hashed for cross-platform
   comparison because PNG encoders (deflate impls, filter heuristics,
   metadata chunks) are not vendor-stable — but PNG is lossless, so
   decoded content equals `sun_rgb` / `dsn_rgb` on any platform that
   reads the file.
2. Run on all three backends: `julia --project scripts/bitexact_test.jl`
   with `HYP_BACKEND ∈ {metal, cuda, cpu}`. Metal and cpu run on Mac;
   cuda runs on Windows (plug the drive in, `HYP_BACKEND=cuda julia
   --project scripts/bitexact_test.jl`). A removable drive shuttles
   outputs between the two machines.

### N-way comparison

3. **`scripts/diff_bitexact_shas.jl`** — auto-discovers every backend
   present in `data/outputs/bitexact/`, echoes each toolchain header
   (so a regression can be pinned to a specific version bump), then
   reports all-pairs MATCH/DIFFER per intermediate per timestamp. For
   three backends that's three pairs (cpu↔metal, cpu↔cuda, metal↔cuda)
   × 20 timestamps × 14 intermediates = 840 comparisons total.
4. **`scripts/diff_bitexact_pixels.jl`** — for each pair of backends
   with UInt8 buffers on disk, per-timestamp diff stats: count of
   differing pixels, min / median / mean / max |Δ|. Essential for
   distinguishing three different fix paths: *bit-exact*, *isolated
   pixels drifting*, *widespread 1-ULP drift*.

SHA match ≠ coincidence: a single bit off anywhere in a 458,752-pixel
buffer produces a different hash. MATCH across all frames × all UInt8
intermediates × all backend pairs = true cross-platform bit-exactness
for user-facing output.

## Divergence sources 8–14

Numbered continuing the algorithms.md list.

### 8. `(a*b) ± c` across connected statements: per-vendor contraction

The fma principle from algorithms.md #5 says: *"if an expression has the
shape `a*b + c` and matters for bit-exactness, write `fma(a, b, c)`
explicitly."* That rule turned out to need a stricter reading. The
issue isn't just the text of a single expression — it's whether the
compiler's optimizer can reconstruct the shape across statements.

```julia
# Looks like three separate ops, right?
tx = common * n_km       # fmul
dx = tx - qx             # fsub
```

With `tx` dead after the sub, both LLVM NVPTX (CUDA) and LLVM for Apple
Air run with `fp-contract=fast` by default and are *permitted* to fuse
them into a single `fma(common, n_km, -qx)`. But they don't always
agree on whether to fuse at a specific site — heuristics depend on
surrounding code, register pressure, and compiler version.

Empirically, at full 896×512 scale the `tx - qx` / `ty - qy` / `tz - qz`
sites inside the ray-cast stereographic rewrite were being fused on
CUDA but not on Metal. That produced ~11% of pixels drifting 1 UInt8
code on DSN, and up to 100 codes on sun after clamp-boundary
amplification inside the 16-tick integration.

**Fix:** make the fma explicit on every `(a*b) ± c` shape reconstructible
from dataflow, including ones spanning a mul assignment and a
subsequent sub:

```julia
# Deterministic: single-rounded on every IEEE 754 + hardware-FMA platform.
dx = fma(common, n_km, -qx)
dy = fma(common, e_km, -qy)
dz = fma(scale,  u2_m1, -qz)
```

Same treatment for `dn = fma(rho2, INV_4R_KM2_F32, 1.0f0)` (was
`dn = 1 + u2` after `u2 = rho2 * INV_4R_KM2_F32`) and
`alen_sq = fma(-lz_geom, lz_geom, d_sq)` (was
`d_sq - lz_geom*lz_geom`). Also in `_gpu_approx_slope_sq`:
`delta = fma(-hsq, INV_2R_M_F32, elev_m - q_elev_m)` (was
`(elev_m - q_elev_m) - hsq*INV_2R_M_F32`).

**Lesson:** the fma principle isn't about expression text, it's about
dataflow. Anywhere a mul-add or mul-sub can be reconstructed from
connected statements, wrap it in explicit `fma`. The fma value doesn't
change whether the compiler fuses or not — when in doubt, write it.

### 9. `sqrt` and `/` in the hot loop are cross-vendor risks

CUDA's Float32 `/` and `sqrt` can lower to `div.approx.f32` /
`sqrt.approx.f32` (2-ULP max error) rather than IEEE `div.rn.f32` /
`sqrt.rn.f32`, depending on `CUDA.math_mode()` and ptxas flags. Metal
Air is stricter (IEEE-precise by default) on these ops. Rather than
audit each `/` on each backend version, we took the safer position:
**no sqrt in the hot loop, one division per step minimum.**

Old stereographic projection per-step (1 sqrt + 3 divisions):

```julia
rho = sqrt(fma(n_km, n_km, e_km * e_km))
u = rho / (2.0f0 * R_km)
u2 = u * u
dn = 1.0f0 + u2
common = R_total / (R_km * dn)
tz = R_total * (u2 - 1.0f0) / dn
```

New per-step (0 sqrts + 1 division):

```julia
rho2 = fma(n_km, n_km, e_km * e_km)
R_total = fma(telev_m, 0.001f0, R_km)
dn    = fma(rho2, INV_4R_KM2_F32,  1.0f0)
u2_m1 = fma(rho2, INV_4R_KM2_F32, -1.0f0)
inv_dn = 1.0f0 / dn                        # ← only remaining div per step
scale = R_total * inv_dn
common = scale * INV_R_KM_F32
# tx, ty, tz folded directly into dx, dy, dz via fma(common, n_km, -qx) etc.
```

Key moves:

- `rho` was never needed alone; we only used `u² = rho²/(4R²) = rho² · INV_4R_KM2`.
  Dropping `rho` eliminates the sqrt.
- `/ (2·R_km)` and `/ R_km` both divide by runtime-constant values. Replaced
  with multiplies by module-const reciprocals `INV_R_KM_F32`, `INV_4R_KM2_F32`.
- `inv_dn = 1/dn` is one division per step. `dn` varies per step so we
  can't precompute. Empirically CUDA.jl's `DEFAULT_MATH` mode emits IEEE
  `/` for this, and the UInt8 output is bit-exact.

**Lesson:** every runtime division and sqrt is a potential cross-vendor
gotcha. Eliminate what you can via algebraic rewrite; precompute
reciprocals for constants; keep only unavoidable ones.

### 10. Squared-form slope comparisons (design pattern)

The ray cast's central operation is "is this horizon steeper than the
current max?" — a comparison of `lz/sqrt(alen_sq)` to a running max.
The straightforward form is 1 sqrt + 1 div per step:

```julia
slope = lz / sqrt(alen_sq)
if slope > max_slope
    max_slope = slope
end
```

Cross-vendor drift from either op accumulates over hundreds of steps.

Rewrite: track max as `(max_num, max_den_sq)` where
`max_slope = max_num / sqrt(max_den_sq)` is implicit. Compare via cross-
multiplication:

```julia
# slope_a > slope_b, both slopes = num / sqrt(den_sq)
@inline function _gpu_gt_slope_sq(an, ad, bn, bd)
    a_pos = an >= 0f0
    b_pos = bn >= 0f0
    a_pos && !b_pos && return true      # positive beats negative
    !a_pos && b_pos && return false
    lhs = (an * an) * bd                  # cross-multiply, squared
    rhs = (bn * bn) * ad
    return a_pos ? (lhs > rhs) : (lhs < rhs)   # flip if both negative
end
```

Only `*` and `<` — all IEEE-mandated, all deterministic. No sqrt, no
div.

The single sqrt + div is deferred to end-of-ray via the atan2 LUT that
converts `(num, den_sq)` back to degrees: 1 sqrt + 1 div per *ray*
instead of per *step* — hundreds-fold reduction in divergence
opportunities. At full scale this reduced DSN drift from max 9 codes
to max 2 codes (and to 0 after the fma-fusion fix in #8).

**Lesson:** when a cross-platform-sensitive inner-loop comparison can
be expressed in squared form, do it. The sign-split adds a bit of
branching but eliminates sqrt+div from every iteration.

### 11. Orthonormality: alen_sq = |d|² − lz²

The ray cast needs `lz` (vertical component in the query's ENU frame)
and `alen_sq = lx² + ly²` (horizontal distance squared). ENU is a
rotation of MOON_ME, so `M` is orthonormal:

```
|M · d|² = |d|²   ⟹   lx² + ly² + lz² = dx² + dy² + dz²
                 ⟹   alen_sq = d_sq − lz_geom²
```

where `d_sq = dx² + dy² + dz²` is computed directly in the MOON_ME
frame (no matrix multiply needed) and `lz_geom` is `lz` before the
observer-height subtraction (horizontal distance is z-translation
invariant).

This eliminates:

- The per-step `lx`, `ly` computations: two fma-chains per step.
- The M-matrix rows 1 and 2 (`M11..M23`): six scalars per pixel, each
  requiring query-setup work.

Query setup simplifies further. Substituting `qclat = 2u/(1+u²)`,
`qslat = (u²−1)/(1+u²)`, `u = rho/(2R)`, `qclon = qn/rho`,
`qslon = qe/rho` into the row-3 entries:

- `M31 = qclat · qclon = qn_km / (R_km · denom_q) = qn_km · INV_R_KM_F32 · inv_denom_q`
- `M32 = qclat · qslon = qe_km · INV_R_KM_F32 · inv_denom_q`
- `M33 = qslat         = (u² − 1) / denom_q      = u2_q_m1 · inv_denom_q`

No `rho_q = sqrt(rho²_q)`, no `qclon = qn/rho`, no `qslon = qe/rho`.
Three module-const reciprocals, two fma, one division — that's the
whole query setup now. (The CPU `_live_query_setup_f32` in
`src/live_helpers.jl` is kept unchanged as the source of truth for
`_precompute_azel` and the test suite.)

**Numerical caveat:** `d_sq − lz_geom²` loses precision via cancellation
when `lz_geom² ≈ d_sq` (ray near-vertical). At Nobile with horizontal
distances 10–1000 km and elevation differences 0.01–10 km,
`lz_geom² << d_sq`, so cancellation is negligible. Worth reauditing for
different observer geometries.

**Lesson:** when a transformation is known to be orthonormal,
Pythagorean identities often let you skip components. Cheaper, fewer
ops, fewer kernel arguments.

### 12. Module-level reciprocals for JIT-time constants

`x / (2·R_m)` is textually a "division by 2·R_m". Per vendor, that may
compile to hardware `div`, or to a reciprocal-multiply, or to a fused
operation — and the choice can vary between vendors compiling the same
Julia source.

When the divisor is derivable from module `const`s (`R_M_F64`,
`MOON_RADIUS_KM`, etc.), we precompute the Float32 reciprocal at
const-eval time:

```julia
# src/live_helpers.jl
const INV_2R_M_F32   = Float32(1.0 / (2.0 * MOON_RADIUS_M))
const INV_R_KM_F32   = Float32(1.0 / MOON_RADIUS_KM)
const INV_4R_KM2_F32 = Float32(1.0 / (4.0 * MOON_RADIUS_KM * MOON_RADIUS_KM))
const INV_MAX_PHOTONS = Float32(1.0 / (2.0 * sum(HALF_CIRCLE)))
```

Each reduces to a specific Float32 bit pattern at const-eval — the same
bit pattern on every build, every backend. In the kernel,
`x * INV_2R_M_F32` is a plain multiply — IEEE-mandated across all GPU
vendors.

**Lesson:** audit any `x / const` where `const` is derivable from module
constants, and convert to `x * INV_CONST`. Also applies to the sun-disk
integration's `px / max_photons` (→ `px * INV_MAX_PHOTONS`) and
anywhere else "dividing by a constant" is dressed up as a runtime div.

### 13. Metal's 31-slot indirect argument buffer limit

Apple Metal limits kernels to 31 arguments in the indirect argument
buffer. Our kernel was around 28 when the cross-vendor work started;
adding kernel-arg Float32 scalars for `inv_2R_m`, `inv_R_km`, `inv_4R_km2`
pushed it to 33, triggering:

```
NSError: Total number of indirect argument buffer resources exceeded
for buffers (33/31) (AGXMetalG16X, code 3)
```

This limit isn't prominent in the Metal docs and doesn't apply on
NVIDIA/CUDA (which has much higher per-kernel argument limits).

**Fix:** promote these Float32 scalars to module `const` and reference
them directly in the kernel. The Julia GPU compiler inlines module
`const`s as immediates — no arg slot consumed. This collapsed the arg
count back below 31 and was simultaneously better for determinism
(see #12).

**Lesson:** Float32 scalars derived from JIT-time constants belong in
module `const`, not in kernel args. Free constant propagation, no arg
buffer pressure, and the cross-vendor benefit of eliminating one more
potential divergence source.

### 14. 1-ULP Float32 drift in atan2 LUT interpolation (the same bug as #8, found one site later)

After fixes 8–13, UInt8 outputs were bit-exact Metal ↔ CUDA, but the
Float32 diagnostic buffers (`de` in degrees, `d_0..d_7` per-ray
horizons) still differed by up to 1 ULP. We initially documented this
as "acceptable drift in debug-only outputs" — but it turned out to be
exactly the same `(a*b) + c` fma-fusion problem as item #8, just at one
more site we'd missed: the linear interpolation between LUT samples
inside `_gpu_atan2_lut_live`.

```julia
v0 = atan_lut[i0 + 1]; v1 = atan_lut[i0 + 2]
angle = v0 + frac * (v1 - v0)        # ← fma-fusion coin flip
```

Apple Air didn't fuse this site; LLVM NVPTX (with default `fp-contract=fast`)
did. One ULP drift in `angle` per ray, which propagated all the way to
the per-ray Float32 horizon hashes.

The diagnostic chain that confirmed it:
1. **IEEE-math guard** (§ 9, § 12) showed `1/3` and `√2` produce
   identical IEEE-rn bit patterns on Metal and CPU. Same expected on
   CUDA. So `_gpu_slope_to_deg_sq`'s explicit `sqrt(den_sq)` and the
   atan helper's `num/den` are NOT the divergence source — vendors agree
   on those.
2. **CPU ↔ Metal: 210/210 SHAs match** (verified locally). Same Apple
   Silicon machine, different LLVM targets (ARM64 vs Apple Air), but
   both make the same fusion choice at this site, so they agree.
3. By elimination, the Metal ↔ CUDA mismatch was at a site where Apple
   Air and LLVM NVPTX choose differently. There's exactly one such
   `(a*b) + c` left in the inner-most function called once per ray —
   the LUT interp.

**Fix:** explicit fma:

```julia
angle = fma(frac, v1 - v0, v0)
```

Applied to both `_gpu_atan2_lut_live` (GPU helper, `src/gpu_live.jl`)
and the CPU-side `atan2_lut` (`src/deterministic_math.jl`) for symmetry.

**Lesson** (refined from #8): the `(a*b) ± c` rule applies *recursively*
to every helper in the dataflow. After fixing the obvious sites in the
hot path, audit every called function for the same pattern. The
atan-LUT helper is invoked once per ray — its fusion choice rides every
final-degree value out of the kernel.

## Deterministic kernel design patterns (consolidated)

These are the patterns that emerged across all fixes in algorithms.md
§§ 1-7 and this doc §§ 8-14. Apply proactively to new GPU kernels
intended for cross-vendor bit-exactness:

**Explicit `fma` on every `(a*b) ± c`.** Including across statements
(`t = a*b; r = t ± c` → `r = fma(a, b, ±c)`), including mul-then-sub
(`r = c - a*b` → `r = fma(-a, b, c)`). Zero performance cost.

**Squared-form comparisons.** If the inner-loop op is "which of two
ratios is larger," track `(num, den_sq)` and compare via cross-
multiplication. Eliminates per-iteration sqrt + div.

**Orthonormality for `alen_sq` and similar.** Pythagorean identities
on orthogonal projections let you skip components. Fewer ops, fewer
kernel args, fewer dependencies.

**Module-const reciprocals for JIT-time constants.** Any `x / const`
where `const` is derivable from module-level values becomes
`x * INV_CONST`, with the Float32 reciprocal pre-rounded at const-eval.

**LUT-based transcendentals.** `tan`, `atan2`, `cos/sin` via LUT +
linear interpolation + IEEE division. See algorithms.md #2.

**Direct compares for level selection.** Replace `unsafe_trunc(Int,
log2(x))` with a comparison ladder. See algorithms.md #3.

**Keep Float32 everywhere.** No accidental `Float64(...)` promotions
in the hot path. See algorithms.md #4.

**Test at realistic scale.** 128×128 smoke tests miss drift that only
manifests at full 896×512. `bitexact_test.jl` exercises the full code
path with representative timestamps.

**Hash the raw buffers, diff the hashes.** SHA-256 on raw `.bin` output
is a zero-tolerance pass/fail. Pair with per-pixel magnitude stats to
distinguish "isolated pixels drifting" from "widespread 1-ULP drift" —
different fix paths.

## Performance impact

The rewrites are strictly faster. Per-step op count comparison:

| Op class | Before | After |
|---|---|---|
| sqrt | 1 | 0 |
| div | 3 | 1 |
| fma / mul | ~13 | ~11 |
| add / sub | ~8 | ~6 |

Structural simplifications:

- Eliminated `rho`, `u`, `qclon`, `qslon`, `qclat`, `qslat` (intermediate
  stereographic values in both query setup and per-step).
- Eliminated M-matrix rows 1-2: six fewer kernel-local scalars per
  pixel, six fewer arguments to `_gpu_cast_ray`.
- Folded `tx/ty/tz` into `dx/dy/dz` — three fewer locals per step.

Kernel-argument consolidation (the Metal 31-limit issue #13) moved
four Float32 constants into module-level, freeing arg slots.

## Current status: 🎯 fully closed

After algorithms.md §§ 1-7 and this doc §§ 8-14, verified on:
- Apple Silicon CPU (KernelAbstractions CPU backend, ARM64 native)
- Apple Silicon Metal GPU (Metal.jl via KernelAbstractions)
- NVIDIA CUDA GPU on Windows x86_64 (CUDA.jl via KernelAbstractions)

| Output | Bit-exactness verified |
|---|---|
| `sun`, `dsn` UInt8 (kernel raw) | **Apple CPU ↔ Metal ↔ CUDA** |
| `sun_rgb`, `dsn_rgb` (palette-applied 3×H×W RGB — the user-visible content) | **Apple CPU ↔ Metal ↔ CUDA** |
| PNG roundtrip (encode → decode preserves `*_rgb` bytes) | **Verified per-platform on all three backends** |
| `azel` Float32 (CPU precompute of sun/earth directions and slope thresholds) | **Apple CPU ↔ Metal ↔ CUDA** |
| `de`, `d_0..d_7` Float32 (per-ray debug horizons) | **Apple CPU ↔ Metal ↔ CUDA** |

### Final verification ledger

- **`] test` regression**: `test/bitexact.jl` runs 20 representative
  timestamps × 14 intermediates + 2 PNG fixture pixel checks each =
  **320 assertions per run** on the KA CPU backend. CPU ↔ Metal
  baseline empirically bit-exact at this full 20-timestamp scope;
  cross-vendor match to CUDA extends by construction (no kernel math
  changed between the 3-way verification below and the timestamp
  expansion).
- **Initial cross-vendor closure**: 15 timestamps × 14 intermediates
  × 3 pairs (cpu ↔ metal ↔ cuda) = **630 SHAs, 100% match** — the
  snapshot that proved the `(a*b) ± c` fma-fusion fixes of §§ 5, 8,
  and 14 held across Apple and NVIDIA. A single bit of drift anywhere
  would have surfaced as at least one differing SHA.
- **Raw-buffer pixel level** at the same 15-timestamp scope: × (sun,
  dsn, sun_rgb, dsn_rgb) × 3 pairs = **180 buffer comparisons, all
  BIT-EXACT** on both byte count and |Δ|.
- **IEEE-math guard**: PASSED on all three backends before every run.
  Hard-aborts if `fma(1+2⁻¹², 1+2⁻¹², -1)` ≠ `0x3a000400`, or if
  `1/3` ≠ `0x3eaaaaab`, or if `√2` ≠ `0x3fb504f3` — the specific
  bit patterns that distinguish IEEE-rn from fast-math / approx
  implementations.
- **PNG roundtrip lossless**: `save_indexed_png` → `FileIO.load`
  preserves `*_rgb` bytes on every backend; any encoder regression
  fails loudly per-run.
- **Toolchain pinned in each SHAs.txt header**: julia 1.11.5,
  KernelAbstractions 0.9.41, Metal 1.9.3, CUDA 5.9.0; platforms
  `arm64-apple-darwin24.0.0` (Mac) and `x86_64-w64-mingw32` (Windows).
  A future version bump that breaks an invariant is localizable to a
  specific delta in the version strings.

### Scope

By construction the same guarantees extend to any IEEE 754 +
hardware-FMA backend — AMD ROCm, Intel oneAPI, Linux x86 CPU — since
the kernel now uses only IEEE-mandated ops (`* + − / sqrt fma`) plus
LUT-based transcendentals (atan2, cos/sin) built from those same ops.
The `bitexact_test.jl` IEEE-math guard (§ 9, § 12) runs on any
backend before emitting SHAs, so a broken backend fails loudly rather
than silently.

**What this means in practice.** (DEM bytes + SPICE kernels + git SHA)
is a deterministic function of the output directory. SHA-256 of the raw
output buffers — or the palette-applied RGB, or the PNG-decoded pixels
— is a repeatable invariant across vendors, across operating systems,
across ARM and x86. No SSIM tolerance, no "close enough" — byte-exact
on every pixel of every frame.

### A retrospective on the last two fixes

The closing two fixes (items 5 → 8 → 14) all followed the same rule —
`(a*b) ± c` patterns must be wrapped in explicit `fma` — applied at
successively deeper sites in the dataflow:

- **#5** (algorithms.md): single-expression `a*b + c` fusion (CPU ↔ Metal).
- **#8** (this doc): cross-statement `t = a*b; r = t ± c` fusion (Metal ↔ CUDA hot path).
- **#14** (this doc): LUT-interpolation `v0 + frac*(v1 - v0)` inside
  `_gpu_atan2_lut_live` AND `_gpu_cos_sin_lut` / CPU siblings (Metal ↔ CUDA
  Float32 buffers; later found to also shift Mac SHAs because ARM Julia
  was not fusing, only the UInt8-floor absorbed the drift).

Of those, #14 had the subtlest diagnostic chain: the UInt8 outputs had
*already* been bit-exact since the #8 fix, so the residual drift
appeared only in debug Float32 buffers and was initially filed as
"accept and move on." Re-opening it — and separately, re-verifying
rather than assuming the `cos_sin_lut` mirror fix had taken effect —
was what closed the final gap. The lesson generalizes: when the rule
applies recursively through every helper in the dataflow, audit until
you've seen every `(a*b) ± c` literally, not just the "obvious" ones
in the top-level kernel body.

## Cross-reference: the full list of divergence sources

For quick navigation, the consolidated list with "where fixed":

| # | Issue | Doc |
|---|---|---|
| 1 | `4R² + ρ²` catastrophic cancellation in Float32 | algorithms.md |
| 2 | `tan()` transcendental cross-platform drift | algorithms.md |
| 3 | `log2()` boundary flip in mipmap level selection | algorithms.md |
| 4 | Silent Float64 promotion inside Float32 hot loop | algorithms.md |
| 5 | `a*b + c` auto-contraction to fma at single expression | algorithms.md |
| 6 | 16×16 az/el NN subsampling stair-step | algorithms.md |
| 7 | Twilight skip at 0° killing mountain-peak speckle | algorithms.md |
| 8 | `(a*b) ± c` per-vendor contraction across statements | this doc |
| 9 | `div.approx` / `sqrt.approx` in hot loop | this doc |
| 10 | Squared-form slope comparisons (pattern) | this doc |
| 11 | Orthonormality: `alen_sq = \|d\|² − lz²` (pattern) | this doc |
| 12 | Module-const reciprocals for JIT-time constants (pattern) | this doc |
| 13 | Metal's 31-slot indirect argument buffer limit | this doc |
| 14 | 1-ULP Float32 drift in atan2 **and cos_sin** LUT interp (item #8 again, one site deeper) | this doc |
| 15 | `dz = scale·u²−1/dn − qz` catastrophic cancellation when projection origin is *inside* the imaged tile (latent on LDEM, visible at 1m) | this doc |

Items 10-12 are patterns rather than discrete bugs — design disciplines
we adopted to remove whole classes of divergence risk rather than plug
specific sites. Items 5, 8, and 14 are the same rule (`(a*b) ± c` →
explicit `fma`) applied at successively deeper dataflow sites: single
expression, across connected statements, inside every LUT helper.

## Bug 15: dz catastrophic cancellation (1m site DEM)

### Symptom

Rendering the live shadow kernel on a 1m site DEM in its native
locally-tangent stereographic projection produces **prominent concentric
ring artifacts** — even on a synthetic flat DEM (constant elevation
everywhere). The flat DEM should produce one uniform sun value across
the image; instead it produces 138 distinct values arranged in
concentric rings centered on the projection origin. The same kernel run
on the 20 m LDEM is bit-exact correct.

### Diagnostic chain (what the rings ruled out)

The investigation worked through hypotheses by elimination:

1. **Mipmap skip artifacts** — disabled mipmaps (`mipmap_base = 1e9`,
   forcing level 0 throughout). Rings persisted **identically**, same
   sun coverage. *Rules out mipmaps.*
2. **Half-meter Int16 quantization (½ m staircase under grazing sun)** —
   added `load_site_dem_f32` and a Float32 path through the kernel.
   Rings persisted essentially unchanged. *Rules out source quantization.*
3. **Synthetic flat DEM (constant 6000 m everywhere)** — kernel still
   produced 138 distinct sun values arranged in rings. **The kernel itself
   is generating position-dependent variation on uniform terrain.** This
   was the smoking gun: zero terrain features → output should be uniform,
   any variation comes from the kernel math.
4. **Patch sweep** — rendered 256×256 patches at varied (qe, qn) on the
   flat DEM. A patch *centered on the projection origin* (rho = 0.18 km)
   produced a single uniform value (255). Patches farther from the origin
   (rho > 0.5 km) showed strong variation. **The bug magnitude grows with
   distance from the projection origin.**
5. **Per-pixel slope dump on flat terrain** — kernel reported `de` (DSN
   horizon angle) values up to **+5.22°** on truly flat terrain.
   Geometric expectation: ~0° (curvature drop only, sub-degree).
6. **Single-step trace, plain Float32** — at one bad pixel, traced
   `(dx, dy, dz)` and `lz_geom` for a 1 m east step. Found
   `dz = -0.122 mm` where the geometric truth is `~5.7e-7 m`.
   Off by **>10⁵×**.
7. **Float64 reproduction of the same trace** — gave `dz = -0.92 μm`,
   matching the geometric expectation almost exactly. **The bug was
   purely a Float32 precision failure.** The kernel formula is
   mathematically correct.

### Root cause

In the level-0 sample inner loop, `dz` was:

```
dz = fma(scale, u2_m1, -qz)        # = scale·(u²−1)/dn − qz
```

Both operands are ≈ −R_total ≈ −1737.4 km. Float32 ULP at that magnitude
is ~2×10⁻⁴ km ≈ **0.2 m**. The geometric value of `dz` is on the order
of millimeters for nearby samples. So the subtraction is **catastrophic
cancellation**: result is rounding noise, not signal.

The noise is not random — it's a deterministic function of `(qe, qn)`
through the rounding of `1 + ρ²/(4R²)` ≈ `1.000000274`. As `ρ` varies,
this denominator crosses Float32 representable boundaries at discrete
ρ values. Each crossing is one ring.

### Why this is invisible on the LDEM

In south-polar PS, the projection origin is at the south pole — far
from any imaged pixel. Across the LDEM Nobile crop, every query has
`ρ_q ≈ 138 km`. Therefore:
- `qz ≈ −1731.9 km` (well-resolved at this ρ)
- `scale·u²−1/dn ≈ −1729.2 km` for samples 20 m away
- They differ by **~2.7 km** — far above Float32 noise
- The subtraction is benign; precision is preserved

The instant you put the projection origin *inside* the tile (= every
locally-tangent stereographic of a small site), pixels close to that
origin have `qz ≈ −R_total` and samples a few meters away also
`≈ −R_total`. The two operands now differ by *centimeters to meters*,
which is exactly Float32 ULP at lunar radius. Cancellation kills the
signal. The bug had been latent in the kernel since day one; the LDEM's
projection geometry hid it.

### Fix

Reformulate `dz` to avoid the `~−R − ~−R` subtraction. Algebraically:

```
sample_z = R_total_s · (u²_s − 1)/dn_s = −R_total_s + 2·R_total_s·u²_s/dn_s
query_z  = R_total_q · (u²_q − 1)/dn_q = −R_total_q + 2·R_total_q·u²_q/dn_q

dz = sample_z − query_z
   = (R_total_q − R_total_s)              ← elev difference, well-conditioned
   + (sample_pos − qz_pos)                ← curvature term, all O(ρ²/R)
```

where:
- `sample_pos = scale · 2·u²` (= scale_s · 2·u²_s/dn_s in disguise)
- `qz_pos = R_total_q · 2·u²_q/dn_q`

Critically, **`qz_pos` must be computed directly, not as `qz + R_total_q`**.
The latter is *also* a `~−R + R` cancellation with the same Float32
~0.2 m ULP. Build it explicitly from `ρ²_q · (2·INV_4R)` (= `2·u²_q`),
multiplied by `R_total_q · inv_dn_q`. Each factor is O(small) in
isolation; the result is precise to sub-mm at lunar radius.

The new formula reduces algebraically to the old one, so it is
mathematically equivalent — but Float32 ULP differs by ~5 orders of
magnitude in the inputs to the subtraction. On the LDEM, the new formula
produces results within a few ULP of the old (= bit-exact tests need
re-pinning, but the qualitative behavior is unchanged). On the 1 m site
DEM, the rings collapse to noise floor.

### Verification

A flat DEM (synthetic, constant 6000 m) under the fixed kernel produces
**1 unique sun value across the entire 4096×4992 image** (vs 138 with
the bug). Per-pixel `de` reports **−0.0°** for every test pixel (vs
+5.22°). Real 1m terrain renders cleanly with proper crater shadow
geometry and no concentric rings.

### Generalization

Bug 15 is the same template as bug 1 (Float32 catastrophic cancellation,
this time in `dz` instead of in `4R² + ρ²`). Both manifest only when
the operand magnitudes happen to put the cancellation *at* Float32 ULP
for the projection geometry being used. The lesson is the *latency*:
**bit-exact regression on one DEM is not proof of correctness for a
different DEM**, because the per-pixel operand magnitudes change with
projection geometry, and Float32 cancellations can be benign in one
regime and catastrophic in another. Cross-DEM stress tests (synthetic
flat DEMs at varied projection geometries) are now part of the
verification protocol.

### Bug 15b: the second cancellation (regression chasing)

After applying the fix above and verifying flat-DEM uniformity, real-
terrain renders still showed faint contour-following ridges in lit
areas. Tracing one bad pixel revealed a *second* catastrophic
cancellation — this one in the *replacement* formula:

```
dz = (qR_total − R_total) + (sample_pos − qz_pos)
```

`qR_total` and `R_total` are *both ~R_KM in magnitude* (1737.4 km +
elevation/1000). Their Float32 representations have ULP ≈ 0.2 m at
lunar radius. Subtracting them recovers Float32 precision *in metres at
lunar radius* — but the geometric *signal* in this term is the
sample-vs-query elevation difference, which is millimetres-to-metres.
The signal-to-noise ratio sat at ~1, and which Float32 representable
each `R_total` rounded to produced position-dependent jumps that
looked like contour-line ridges following the terrain.

Diagnostic that pinpointed it: per-pixel dump on adjacent pixels along
a smooth-elevation row. Pixel (col=1250, row=500) on `nobile_1m` at
2027-01-22T07 reported `de = −6.94°` (slope ≈ −0.122) while neighbors
on either side reported a smooth ~−10.86°. The value −0.122 is
*exactly* the Float32 ULP-at-lunar-radius / 1 m — a signature of the
bug. All 9 rays from that one pixel returned identical `−6.935°` to
4 decimals, meaning the rays didn't really converge on the same
occluder; they all inherited the same noise floor.

**The fix is trivial once seen.** `qR_total − R_total = (qelev_m − telev_m) · 0.001`
exactly. Compute the elevation-difference term in *metres*, not at
lunar radius:

```
dz = (q_elev_m − telev_m) · 0.001 + (sample_pos − qz_pos)
```

Now the leading term has Float32 precision *in metres of elevation*,
not at lunar radius — sub-millimetre instead of decimetre. The artifact
disappears. After-fix verification at the same pixel: `de = −10.864°`
(continuous with both neighbors). Synthetic flat DEM: 1 unique sun
value across the entire 4096×4992 image. Real 1m terrain: clean crater
shadows, no visible ridges.

`qR_total` itself is no longer needed inside `_gpu_cast_ray` and was
removed from the function signature.

**The general rule both 15 and 15b reinforce:** never form a
quantity-of-interest as the difference of two values at lunar radius
in Float32. Either reformulate to compute the small difference
directly (metres-to-km, not km-to-km), or carry the precision in a
quantity that lives at the *signal's* magnitude rather than the
operand's.

Two cancellations were present, the second hidden behind the first.
The diagnostic chain that found 15b was the same one that found 15
(synthetic flat DEM → patch sweep → per-pixel dump → Float32 trace),
plus the additional step of *also tracing the corrected formula in
Float32* and noticing that intermediate operand magnitudes were still
at lunar radius. **Float64 reproduction of the corrected formula is
necessary but not sufficient — verify each step's operand magnitudes
in Float32 land within Float32's ULP for the signal you're computing.**

### Correctness verification (post-fix)

The pre-fix kernel was **bit-exact across vendors** (Apple CPU ↔ Metal ↔
NVIDIA CUDA all produced identical output). That property guarantees
*reproducibility*, not *correctness*. With the precision floor of the
old `dz` formula sitting at ~0.2 m at lunar radius (one Float32 ULP),
all three vendors made the same Float32 rounding errors and produced
the same wrong answer at boundary pixels.

After fix 15+15b the LDEM 20-timestamp regression had to be re-pinned.
Comparing the old (pre-fix) and new (post-fix) outputs across all 20
LDEM timestamps:

| | Sun | DSN |
|---|---|---|
| Pixels differing | 2.15% | 1.53% |
| Mean deviation per pixel | 0.52 / 255 | 1.31 / 255 |
| Max deviation | 255 (full swing) | 205 |

**The differences cluster along shadow boundaries** — visible as fine
contour curves in the diff heatmaps (`data/outputs/bitexact_diffs/`).
Pixels deep in shadow stay deep in shadow; pixels deep in sunlight
stay deep in sunlight; only pixels close to the lit/shadow threshold
change, because that's where Float32 ULP differences in the slope
computation can flip the kernel's verdict.

#### Numerical proof (one max-deviation pixel)

For LDEM grid pixel (row=8963, col=19206) at 2027-01-22T07-00-00 — old
verdict 0 (full shadow), new verdict 255 (full sun) — traced through
the first sun-ray sample at d=1 against a Float64 reference:

| | dz (μm) | Slope angle (°) | Error vs Float64 |
|---|---:|---:|---:|
| Float64 ground truth | -63.60 | 4.4701 | 0 |
| Float32 NEW | -62.95 | 4.4683 | 0.65 μm dz, 0.002° slope |
| Float32 OLD | **-366.21** | **5.3308** | **302.6 μm dz, 0.861° slope** |

The old formula's `dz` was off by **5.7× from ground truth** due to
the catastrophic cancellation `scale·u²−1 − qz` (both ≈ −R), producing
a 0.86° slope error at one sample. With sun_top_el threshold ≈ 4.88°,
the old kernel computed slope 5.33° (above threshold → "shadow"); the
new kernel computes slope 4.47° (below threshold → "lit"). Same pixel,
inverted conclusions, *because the old formula's Float32 cancellation
floor sat above the true slope's distance from the threshold*.

The new formula matches Float64 to within **1% relative error** on `dz`
(= within Float32 ULP at the metres-of-elevation magnitude where
`(qelev_m − telev_m) · 0.001` evaluates). The 94 max-255 pixels in
this timestamp are pixels where the old cancellation error was large
enough to flip the verdict; the ~10,000 lower-deviation pixels are
boundary pixels where the cancellation produced a smaller but still
incorrect shift.

**Conclusion**: the previous LDEM bit-exact pin was self-consistent
(reproducible across vendors) but systematically wrong at boundary
pixels by up to 0.86° of slope. The new pin captures the correct
geometry, verified against Float64 ground truth.
