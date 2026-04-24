# Shadow generation: precomputed horizons vs. live raycasting

JuliaMapbuilder implements two algorithms that produce the same kind of
output — UInt8 PNG time series of sun-illumination fraction and
DSN over-horizon angle — via fundamentally different architectures. This
document explains both, the (many) floating-point bugs we surfaced while
making the live algorithm bit-exact across heterogeneous hardware, and
the case for standardizing on live going forward.

## TL;DR recommendation: use live

**We recommend live raycasting as the default for all future work.**
Precompute is faster on a single workstation at dense year-scale runs at
a single observer height, but that narrow advantage evaporates once you
factor in the intermediate data product, multi-height scenarios, 1m
target scaling, or cluster access.

The case in one paragraph: precompute produces an intermediate product
that is **2.46 GB per region per observer height at 20m — and scales to
roughly a terabyte at 1m** across the observer heights and regions we
care about. That archive has to be built (serial, ~29 hours at 1m),
versioned, copied to compute nodes, kept synchronized across team
members, and invalidated whenever the horizon algorithm changes. Live
raycasting has no such artifact — just the LDEM and SPICE kernels. It is
**conceptually simpler** (no separate build step, no horizon .bin file
format, no quantization), **more accurate** (continuous sun position
instead of 1440-bucket 0.25° quantization — no binning artifact at the
terminator), and **bit-exact across Apple Silicon, x86, and NVIDIA/AMD
GPUs** thanks to deterministic LUTs and explicit `fma()` usage (see
below). On a supercomputer the throughput gap closes completely: 8× H100
node generates a year of 20m frames in ~2.5 min, 64× cluster generates
a year of 1m frames in ~15 min. For that you get to delete the whole
horizon archive pipeline.

### Where precompute still wins

| Scenario | Winner |
|---|---|
| Single workstation, dense year-scale 20m run, one height, no storage constraint | Precompute (~1.5× faster) |
| Many years (decades), one region, one height, same code version | Precompute amortizes |
| You *have* to use precomputed horizons for external reasons | Precompute |

Everywhere else — multi-height, 1m, cluster, sparse timestamps, clean
terminator, reproducibility — **live wins.**

## Shared geometry primitives

Both algorithms share:

- **Polar-stereographic projection** (MOON_ME cartesian ↔ DEM pixel). At the
  south pole the stereographic coordinates are small enough that a
  numerically-stable Float32 formulation works: `u = ρ / (2R)` gives
  `cos(lat) = 2u/(1+u²)`, `sin(lat) = (u²−1)/(1+u²)`. This avoids the
  catastrophic `4R² + r²` magnitude mismatch (≈ 1.2e13 + 1e10 loses 10
  bits in Float32).
- **ENU (east-north-up) per-pixel frame** for transforming sun/earth
  positions to local elevation/azimuth.
- **Deterministic transcendentals via LUT**: `atan2_lut`, `cos_sin_lut`.
  All table-based + IEEE arithmetic → bit-identical on any IEEE 754
  platform that implements FMA correctly.
- **UInt8 output encoding**: sun channel = `clamp(trunc(255 * sun_frac), 0, 255)`;
  dsn channel = `clamp(floor(over_horizon_deg * 10), 0, 250)`.

## Precomputed horizons (precompute pipeline)

### How it works

For each target pixel, build a 1440-bucket horizon profile once — the
maximum elevation-angle slope visible in each 0.25° azimuthal bucket.
At render time, look up the horizon profile and integrate the sun disk
against the buckets around the sun's azimuth.

```
# Phase 1 — precompute horizons (per patch, one-time per observer height)
for each 128×128 patch in target DEM:
    build near-field caster array from neighboring patch terrain
    build far-field caster array from LDEM (20m global)
    for each pixel in patch:
        for each of 1440 buckets (0.25° each, covering 360°):
            horizon[pixel, bucket] = max slope of any caster in that bucket
    write horizon patch → .bin file

# Phase 2 — render (per timestamp, very fast)
for each timestamp:
    compute sun, earth positions (SPICE)
    for each pixel (subsampled 16×16 for az/el):
        compute sun_az_deg, sun_el_deg, earth_az_rad, earth_el_deg
    for each pixel (full resolution):
        look up 6 buckets of horizon around sun_az
        integrate sun disk visibility → sun_frac
        look up 2 buckets around earth_az, interpolate → horizon_el
        over_horizon_deg = earth_el_deg − horizon_el
        write UInt8
```

### Technical details worth knowing

- **Patch size: 128×128 pixels.** Chosen to amortize per-patch setup cost
  against per-pixel horizon computation without requiring huge transient
  GPU buffers.
- **Near-field caster array.** For each target pixel, gather the 3D
  positions of nearby terrain points (within the same patch plus a
  neighboring-patch halo). Ray geometry reduces to dot products.
- **Far-field caster array.** Grid-filtered far-field sampling — the
  effective stride grows with distance as `d · π/720`, matching the
  angular horizon resolution (0.25° per bucket). This is what the
  precompute samples *into*; the live algorithm replicates this stride
  in `_cast_ray_base` to stay comparable.
- **1440 buckets (= 360° / 0.25°)** is coarser than it sounds — the sun's
  angular radius is 0.258°, so the sun disk spans ~2 buckets. When the
  sun sits between bucket centers, the disk integration has to
  interpolate, producing a visible ~1-pixel stair-step at the
  terminator. This is the "binning artifact."
- **1440-bucket storage cost.** Each pixel stores 1440 × Float32 = 5760
  bytes. A 896×512 region = 2.46 GB per observer height.
- **Observer heights multiply storage.** Six heights (0, 0.5, 1, 2, 3, 4m)
  × 896×512 region = 14.8 GB per region. Production uses multiple
  regions.
- **Subsampled az/el (16×16 block).** Both algorithms compute sun/earth
  az/el at every 16th target pixel and nearest-neighbor-replicate for
  performance. At the terminator this produced visible rectangular
  stair-steps; live now uses per-pixel az/el (see below).
- **Shadow generation GPU kernel** (`src/shadows.jl` + `src/gpu_kernels.jl`)
  processes one patch at a time. Horizon patches are mmap'd from disk on
  demand. 6.3× faster than the original CPU implementation (pixel-
  identical output).
- **Phase 1 cost:** ~83 s/patch × 28 patches for the 896×512 Nobile
  region = ~39 min per observer height.
- **Phase 2 cost:** < 2 min for 4,392 timestamps (one year at 2h cadence).

### Advantages

- **Amortization:** after one precompute pass, any number of timestamps
  for that (region, height) tuple renders in seconds each.
- **Fast per-timestep rendering** once horizons exist.
- **Coarse storage grain** (one file per patch × height) is easy to
  parallelize across a filesystem.

### Disadvantages

- **Huge intermediate data product.** Multi-GB per (region, height);
  must be versioned, copied to compute nodes, archived.
- **Quantization (binning) artifact** at the terminator from the 0.25°
  bucket step.
- **Observer height rigidity.** New heights require redoing the
  precompute.
- **Arbitrary timestamps cheap, but the region and height must be
  fixed up front.**
- **At 1m resolution the precompute itself is expensive** (~29 hours for
  one year / one observer height at Nobile; see scaling section).

## Live raycasting

### How it works

At each timestamp, for each pixel, cast 8 rays (6 around the sun, 2
around earth) through the LDEM mipmap pyramid and integrate partial disk
visibility on the fly. No precomputed horizons.

```
# For each timestamp:
compute sun, earth positions (SPICE)
per-pixel azel = precompute_azel(ldem, sun, earth)   # small buffer

GPU kernel, one work-item per pixel:
    qelev_m = ldem[pixel]
    compute (qx, qy, qz, M) = live_query_setup_f32(pixel, qelev_m)
    sun_az, sun_el, earth_az, earth_el = azel_packed[pixel]
    if sun_top_el_deg > TWILIGHT_SKIP_DEG:                  # ≈ -10°
        for each of 6 sun buckets around sun_az:
            d_i = cast_ray_hierarchical(mipmap, direction, threshold)
        sun_frac = integrate_sun_disk(d_0..d_5, sun_el_deg, sun_az_deg)
    else:
        sun_frac = 0
    if earth_el_deg > TWILIGHT_SKIP_DEG:
        de, df = cast_ray_hierarchical × 2 around earth_az
        over_hz_deg = earth_el_deg - fma(e_fr, de - df, earth_el_deg - de)
    else:
        over_hz_deg = -90
    write UInt8(clamp(trunc(255 * sun_frac), 0, 255))
    write UInt8(clamp(floor(over_hz_deg * 10), 0, 250))
```

### Technical details worth knowing

- **Hierarchical mipmap pyramid** (5 levels, each half-resolution). The
  mipmap stores both max-pooled and min-pooled elevations. At each ray
  step the mipmap cell is checked:
  - If cell's max-possible slope < current max_slope: skip the whole
    cell (jump by `cell_width`).
  - If cell's min-possible slope ≥ threshold: early-terminate ray.
  - Else: descend to base level, bilinear-sample, commit slope.
  The approximate slope uses a flat-earth + moon-curvature correction
  (`drop = d² / (2R)`), cheap to compute.
- **Log2-free mipmap level selection.** Uses direct `<` compares against
  `{1, 2, 4, 8, 16} × MIPMAP_BASE_THRESH`. Metal's `log2` and Julia's
  `log2` differ by up to 3 ULP, and at values just below a power of two
  (e.g. `31.9999996`) `trunc(log2)` goes different ways → different
  mipmap level → divergent ray casts. Compares are bit-exact.
- **Deterministic `tan` via `cos_sin_lut`.** Early-return thresholds use
  `tan(el * π/180)` — a transcendental that differs by ~1 ULP between
  Julia and Metal. Precomputed on CPU via `sin/cos` LUT + IEEE divide,
  passed into GPU buffer.
- **FMA-stable arithmetic.** Metal's optimizer auto-contracts `a*b + c`
  into a single-rounded `fma` when it finds the pattern. Julia on CPU
  does *not*. Where this matters (e.g. `earth_el − (de + e_fr*(df−de))`),
  the code is rewritten to an explicit `fma(e_fr, de−df, earth_el−de)`
  form that both compilers honor byte-identically.
- **Float32 everywhere.** `_slope_to_deg_f` was previously Float64 mid-
  computation; changed to `rad * Float32(180/π)` for byte-exactness with
  GPU (and as a side-effect ~2-3× CPU speedup, because the Float64
  widening was in a hot inner loop).
- **Per-pixel az/el.** Old subsampled-16×16 nearest-neighbor scheme
  produced stair-step artifacts at the terminator. Per-pixel is actually
  *faster* on 8 threads (1.5 ms vs 4.8 ms for the subsampled path on the
  20m frame) because the 458k-item loop parallelizes cleanly where the
  1,792-item subsampled loop is dominated by thread startup.
- **Twilight skip at -10°.** Early-exit when sun/earth disk top is below
  `TWILIGHT_SKIP_DEG = -10°`. Derived as a conservative safety margin
  over the worst-case local-horizon depression `−2·√(Δh/(2R)) ≈ −8.7°`
  for `Δh = 20 km` lunar relief. Skip at 0° would kill the mountain-peak
  speckle visible across the terminator.
- **Dynamic ray max-distance.** The ray terminates when no terrain at
  distance d can occlude (d > MAX_TERRAIN_M / tan(useful_el)). Prevents
  wasted marching when the sun is high.
- **Ray-bucket rotation.** Each pixel has a different local frame
  orientation. The frame-offset azimuth `off_rad` and bucket remapping
  align ray directions so bucket indices mean the same physical
  direction across pixels.
- **Byte-exact CPU ↔ GPU.** Verified on 7 representative 2026
  timestamps × 458k pixels × 2 channels = 6.4M pixel comparisons, zero
  diff. Extends to any IEEE 754 platform with correct FMA (H100, V100,
  AMD GPUs).
- **Performance:** ~0.83 s per timestep on Apple M-series, ~0.9× → 0.3 s
  on H100 (rough FP32 scaling). Full year (4,392 timesteps at 2h) ≈ 60
  min on M-series, ~20 min on 1× H100.

### Advantages

- **No intermediate data product.** Just the LDEM + SPICE kernels.
  Trivial to ship to compute nodes; no 2.46 GB × height × region archive.
- **No binning artifact.** Continuous per-pixel sun position instead of
  1440-bucket quantization. Visibly cleaner terminator.
- **Flexible** — new observer heights, arbitrary timestamps, masked
  sub-regions at zero additional cost.
- **Byte-exact, platform-independent.** Same PNGs on CPU, Metal, CUDA,
  ROCm. SHA-256 of output is a deterministic function of (DEM, SPICE
  kernels, git SHA of code).
- **Trivially parallelizable at the timestep level** — each frame is
  independent, no inter-GPU communication needed.

### Disadvantages

- **No amortization.** Every timestep pays the full ray-cast cost.
- **Slower on a single workstation** for dense year-scale runs at a
  single observer height.
- **Scales badly with target resolution.** 1m target has 44.5× more
  pixels → 44.5× more ray casts. Mipmapping keeps per-ray cost constant,
  but per-pixel count is the fundamental limit. An algorithmic rewrite
  (spatial coherence — ray-bundle sharing, coarse horizon cache) would
  be needed to compete with precompute at 1m on a single GPU.

## Performance comparison

### 20m target (896×512 Nobile), 1 year at 2h cadence, 1 observer height

| Phase | Precompute | Live (1× M-series) | Live (1× H100 est.) |
|---|---|---|---|
| Setup / precompute | 39 min | ~2 s | ~2 s |
| Render all 4,392 timestamps | < 2 min | 60 min | 20 min |
| **Total wall time (1 year)** | **~41 min** | **~60 min** | **~20 min** |
| Cost to add 2nd year (same height) | 2 min | 60 min | 20 min |
| Cost to add 2nd observer height | 39 min | 0 min | 0 min |

### 1m target (4992×4096 Nobile), 1 year, 1 observer height

| Phase | Precompute | Live (1× M-series est.) | Live (1× H100 est.) |
|---|---|---|---|
| Precompute | ~29 hrs | — | — |
| Render (1.2 s/ts × 4,392) | ~88 min | — | — |
| Year generation (36 s/ts × 4,392) | — | ~44 hrs | ~15 hrs |
| **Total (1 year)** | **~30 hrs** | **~44 hrs** | **~15 hrs** |

At 1m, live on a single GPU is slower than precompute. But on a modest
cluster (8× H100 node) live drops to ~2 hrs/year — competitive with
precompute, with all the flexibility benefits, and with no intermediate
data product to manage.

### 64× H100 (8 nodes × 8 GPUs) timestep-parallel projection

| Target | Per-year wall time |
|---|---|
| 20m | ~20 s |
| 1m | ~15 min |

## Bugs and discoveries: achieving bit-exact CPU ↔ GPU

Both algorithms converge to bit-exact output across platforms *if* the
implementation is disciplined. Precompute was determinism-audited
earlier. The live path was a longer journey — we found seven classes of
platform-divergent behavior, each producing visually visible artifacts
or bit-off outputs, each with an interesting lesson. A developer
dropping into this codebase should read this section before touching the
math.

### 1. Float32 stereographic projection: `4R² + r²` catastrophic cancellation

The textbook polar-stereographic cartesian → MOON_ME formula involves
`4R² + ρ²` in the denominator. At the Moon's scale `4R² ≈ 1.2×10¹³`
while `ρ² ≈ 1×10¹⁰` — the ratio is ~10³, which means ρ² contributes
only to the bottom 10 bits of a Float32 mantissa. Most of the
information is silently discarded during addition.

**Fix:** reformulate in the tan-half-angle variable `u = ρ/(2R)`. For
the Nobile region `u ≈ 0.05`, so `1 ± u²` stays near 1 with full Float32
precision. The same cartesian comes out via:

```
u = ρ / (2R)
cos(lat)  =  2u / (1 + u²)
sin(lat)  =  (u² − 1) / (1 + u²)
X = common · n_km,  Y = common · e_km,  Z = R_total · (u²−1)/(1+u²)
     where common = R_total / (R · (1 + u²))
```

This is used in both `_live_query_setup_f32` and `_stereo_to_moonme_f32`.
Lesson: at planetary scales, always audit Float32 formulas for
magnitude mismatch in intermediate sums; a numerically-stable
reformulation often exists.

### 2. `tan()` is a transcendental; transcendentals are not IEEE-specified

We originally had `sun_useful_slope = tan(sun_top_el_deg * π/180)`.
`tan` at Float32 differs by ~1 ULP between Julia (on ARM64) and Metal
for reasonable inputs. That 1 ULP is enough to shift the early-return
threshold, causing a ray to terminate one step earlier on one platform
than another — which propagates into a completely different `max_slope`
and a visibly-different UInt8 for that pixel.

**Fix:** compute `tan` as `sin(θ) / cos(θ)` using the `cos_sin_lut`
(bit-identical LUT + linear interp) followed by IEEE division. This is
done on CPU and passed to the GPU in `azel_packed[*, *, {5,6}]`.
IEEE 754 mandates correctly-rounded `/`, so both sides agree
bit-for-bit.

Lesson: IEEE 754 mandates exact rounding for `+ − × / sqrt fma`, but
not for transcendentals. Any algorithm aspiring to cross-platform bit-
exactness must either use LUTs or restrict itself to the IEEE-mandated
basic ops.

### 3. `log2()` diverges at power-of-two boundaries → mipmap level diverges

The hierarchical ray cast used `unsafe_trunc(Int32, log2(dm))` to pick
the natural mipmap level for distance `dm`. `log2(31.9999996)` returns
5.0 on one platform and 4.9999…6 on another (up to 3 ULP apart). The
truncation then gives level 5 vs level 4 — completely different
mipmap cells, completely different terrain samples, different ray cast.
We saw this as ~2 pixels drifting by 1 UInt8.

**Fix:** replace with direct comparisons against the known level
boundaries. Bit-exact and actually faster:

```
lvl = d < 1·MIPMAP_BASE_THRESH ? 0 :
      d < 2·MIPMAP_BASE_THRESH ? 1 :
      d < 4·MIPMAP_BASE_THRESH ? 2 :
      d < 8·MIPMAP_BASE_THRESH ? 3 : 4
```

Lesson: anywhere you use a transcendental only to round to an integer,
replace with a comparison ladder.

### 4. Silent Float64 promotion inside a Float32 hot loop

```julia
# CPU (wrong):
@inline function _slope_to_deg_f(slope::Float32)
    rad = atan2_lut(slope, 1.0f0)
    return Float32(Float64(rad) * 180.0 / π)     # ← Float64 mid-computation
end
```

The `Float64(rad) * 180.0 / π` path promoted to Float64, did the mul and
div in Float64, then cast back. The GPU version did `rad * Float32(180/π)`
entirely in Float32 (Metal has no Float64). Result: CPU and GPU differed
by 1 ULP in the per-ray horizon degrees (`d0..d5`), which then
compounded through the sun-disk integration into a visible UInt8 diff.

**Fix:** `rad * Float32(180/π)` on both sides. As a bonus this also made
the CPU ray cast ~2–3× faster because the Float64 widening was in a hot
inner loop.

Lesson: a single implicit Float64 promotion in a Float32 pipeline is
both a correctness bug (platform divergence, since GPUs often lack
Float64) and a perf bug. Audit for `Float64(x) * ...` and
`some_Int * some_Float32` patterns.

### 5. Metal auto-contracts `a*b + c` into FMA; Julia CPU doesn't

This was the subtlest bug and the one that required the deepest dive.
In the DSN over-horizon calculation:

```
over_hz_deg = earth_el_deg - (de + e_fr * (df - de))
```

The inner expression `de + e_fr * (df - de)` fits a pattern Metal's
optimizer recognizes and fuses into a single-rounded `air.fma.f32`
instruction. Julia's LLVM front-end on CPU does *not* auto-contract
(and doesn't, unless you explicitly use `muladd` or enable
`@fastmath`). The two sides now do a different number of roundings:

- Metal: one rounding (single FMA)
- CPU:   two roundings (separate mul, separate add)

They disagreed by 1 ULP on a handful of pixels near the terminator.

We spent a while narrowing the cause:

1. First wrote narrow tests on individual ops — `atan2_lut`,
   `_stereo_to_moonme_f32`, 3-term matrix-vector product, `sqrt`,
   division, `mod` — all 0/100,000 diverge. So the primitives were fine.
2. Tried adding `muladd(a, b, c)` everywhere on both sides. Made it
   *worse*: CPU's `muladd` does lower to hardware FMA on Apple Silicon,
   so CPU was now using FMA; but GPU was apparently *not* contracting
   this particular spot — the two sides were now asymmetric in which
   sites used FMA.
3. Instrumented a single-pixel trace output from the GPU kernel +
   matching CPU trace, compared all 30 intermediate Float32 values at
   the known divergent pixel. Found that individual `d0..d5` agreed
   bit-for-bit *except* when they went through `_slope_to_deg_f` (bug
   #4 above), and separately that `over_hz_deg` disagreed while its
   inputs `de, df, e_fr, earth_el_deg` all agreed.
4. Isolated the `over_hz_deg` computation in a standalone kernel — CPU
   and GPU agreed. So the pattern was fine *in isolation*. The Metal
   compiler was doing something different in the context of the full
   kernel — presumably rearranging the subtraction into the FMA
   differently.
5. Rewrote the expression as an algebraically equivalent single-FMA
   form that neither compiler can "improve":

   ```
   over_hz_deg = fma(e_fr, de - df, earth_el_deg - de)
   ```

   This is `earth_el − de + e_fr·(de − df)` — mathematically the same
   as `earth_el − de − e_fr·(df − de) = earth_el − (de + e_fr·(df−de))`,
   but written as a single explicit `fma` call on a pre-computed
   difference. Both Julia's `fma()` (→ hardware FMA on Apple Silicon)
   and Metal's `Base.fma` (→ `air.fma.f32`) produce the same
   single-rounded result. No auto-contraction ambiguity.

After this fix: 7 representative timestamps × 458,752 pixels × 2
channels = **6,422,528 pixel comparisons, zero bit divergence.**

### The `fma()` principle

> **`fma(a, b, c)` is the one floating-point function you can trust to
> produce the same bit pattern on every IEEE 754 platform with hardware
> FMA.** Use it explicitly wherever two sides of a fused-vs-unfused
> boundary might disagree.

This is the cornerstone of the live pipeline's cross-platform bit-
exactness. IEEE 754-2008 mandates `fma(a, b, c)` must produce the
correctly-rounded value of `a·b + c` with a single rounding step, full
stop. Metal's `air.fma.f32`, NVIDIA's PTX `fma.rn.f32`, AMD's ROCm
FMA, ARM NEON `FMLA`, x86 AVX2 `VFMADD` — all produce the identical
bit pattern for the same inputs. In contrast, any expression that
**could** be fused is a platform-dependent coin flip: some compilers
fuse, some don't, and the choice can change across compiler versions or
even based on surrounding code. Writing `fma(a, b, c)` by hand
removes the ambiguity.

Where we use it in live:

- **DSN over-horizon:** `fma(e_fr, de - df, earth_el_deg - de)` (the
  bug #5 fix).
- **Sun-disk integration**: the inner loop `horizon_el = frac *
  bucket_delta + left_el` stays as auto-contractible pattern because
  both sides happen to agree here — but could be upgraded to explicit
  `fma` for future-proofing.
- Elsewhere in the projection (`_stereo_to_moonme_f32`,
  `_live_query_setup_f32`) we deliberately keep the arithmetic simple
  and match textually on both sides so auto-contraction (if it happens)
  happens the same way on both.

Rule of thumb: if an expression has the shape `a*b + c` or `a*b - c`
**and matters for bit-exactness**, write `fma(a, b, c)` or
`fma(a, b, -c)` explicitly. The performance is the same; the
determinism is guaranteed.

### 6. 16×16 subsampled az/el produced stair-step artifacts

The original reference's `compute_azel_subsampled` computes sun/earth
az/el at every 16th target pixel and nearest-neighbor-replicates. At
the terminator — where sun_el is very near zero and sun_el variation
across the 896-pixel patch is only ~0.6° — the 16-pixel NN step of
~0.01° in `sun_el` is ~4% of the sun disk diameter. That's enough to
flip a pixel from "fully lit" to "fully shadowed" at every block
boundary, producing a visible rectangular staircase at every
terminator in every twilight frame.

**Fix:** per-pixel az/el. On a multi-threaded CPU it turns out to be
*faster* than the subsampled path (1.5 ms vs 4.8 ms for a 896×512
frame on 8 threads), because the 458,752-item loop parallelizes
cleanly while the 1,792-item subsampled loop is dominated by thread
launch overhead.

Lesson: subsampling optimizations that were worthwhile on single-
threaded 2010-era hardware may now be net-slower on a threaded 2025
machine. Always re-benchmark.

### 7. Twilight skip at 0° killed mountain peak speckle

We had an early-exit

```
sun_below  = sun_top_el_deg <= 0°    # ← bug
earth_below = earth_el_deg  <= 0°    # ← bug
```

that zeroed sun_frac whenever the sun *disk top* was below a flat
horizon. But sun_el varies only ~0.6° across the frame while mountain
peaks can see terrain a lot lower — their local horizon is negative.
A pixel on Mt. Malapert looking into the Nobile basin has a local
horizon at roughly `−2·sqrt(Δh/(2R))` radians (minimizing over distance
with the `d²/(2R)` curvature drop term). For Δh = 20 km of lunar relief
and R = 1737 km, that's **≈ −8.7°**. So the flat-horizon check was
discarding ray casts at exactly the pixels where they mattered most —
mountain peaks catching sunlight when the valley floor is in shadow.
The visible artifact was a clean straight diagonal cutting through
every terminator frame, with no speckle in the dark half.

**Fix:** skip threshold lowered to −10° (`TWILIGHT_SKIP_DEG`), derived
from the geometric worst case above plus a small safety margin. Night
pixels (sun_el << −10°) still short-circuit. Twilight pixels run the
full ray cast and correctly resolve mountain-peak visibility.

Lesson: when an "obvious" early-exit is cheap, double-check the
worst-case geometry before trusting the threshold. A wrong threshold
can corrupt the output invisibly at first (no crash, just wrong
pixels).

### Summary table

| Issue | Symptom | Fix |
|---|---|---|
| `4R² + ρ²` in Float32 | Silent precision loss at Moon scale | `u = ρ/(2R)` tan-half-angle form |
| `tan()` is a transcendental | 1-ULP CPU/GPU drift | Compute on CPU via `sin/cos` LUT + IEEE `/`, pass in buffer |
| `log2()` near 2ⁿ | Mipmap level flips → divergent ray cast | Direct `<` compares |
| Float64 promotion inside Float32 inner loop | 1-ULP + perf hit | Keep pure Float32 |
| Metal auto-FMA vs Julia no-FMA | 1 pixel diff near terminator | Explicit `fma(a, b, c)` with algebraically-rewritten form |
| 16×16 az/el NN replication | Rectangular stair-step at terminator | Per-pixel az/el |
| Twilight skip at 0° | Mountain-peak speckle erased | Skip at −10° (worst-case local horizon) |

**Result (intra-vendor): 6.4M pixel comparisons, zero bit divergence**
between Apple Silicon CPU and Metal GPU. That established intra-vendor
determinism on a single machine.

Making the same kernel bit-exact across *different* GPU vendors
(Metal ↔ CUDA) turned up seven more classes of divergence that the
CPU↔Metal audit didn't exercise — compiler fp-contract defaults differ
per vendor, `div.approx.f32` / `sqrt.approx.f32` may replace IEEE
rounding in the hot loop, LUT-interpolation helpers have the same
`(a*b) ± c` fusion problem as the outer kernel, and so on. See
[`cross-vendor-determinism.md`](cross-vendor-determinism.md) for the
full story (items 8-14), the design patterns that emerged
(squared-form comparisons, orthonormality, module-const reciprocals,
explicit fma on `(a*b) ± c` across statements), and the verification
harness (IEEE-math guard, palette-applied RGB audit, PNG lossless
roundtrip check).

**🎯 Current verified status: fully closed.** 15 representative 2027
timestamps × 14 intermediates × 3 backend pairs = **630 SHAs, all
matching** across Apple Silicon CPU ↔ Metal ↔ NVIDIA CUDA. Every
stage of the pipeline — kernel raw UInt8 output, palette-applied RGB,
PNG roundtrip, Float32 diagnostics, and CPU precompute — is byte-exact.
The regression check is now wired into `] test` (`test/bitexact.jl`,
20-timestamp scope on the KA CPU backend, ~10-15 min; compares SHAs
against a hardcoded known-good table and decoded PNG fixtures pixel-for-
pixel). By construction the same guarantees extend to any IEEE 754 +
hardware-FMA backend (AMD ROCm, Intel oneAPI, Linux x86 CPU).

## Deployment: supercomputer batch generation

For multi-GPU batch generation the live algorithm has clear architectural
advantages:

1. **Timestep parallelism, not patch parallelism.** Each timestamp is
   independent; no inter-GPU communication required. Patch parallelism
   only helps if a single timestep exceeds one GPU's memory, which
   doesn't happen here (2.5 GB mipmap < any HBM).
2. **Dynamic work queue.** Timestep costs vary ~3× (night vs high-sun).
   Static range assignment produces stragglers; a dynamic pull queue
   keeps GPUs busy.
3. **One-time setup per worker.** Each GPU loads DEM, builds mipmaps,
   initializes SPICE once, then iterates over its timestamp chunk.
4. **Shared-FS PNG writes.** PNG encode is CPU-light; output goes straight
   to the parallel filesystem.

Pseudocode:

```
driver (rank 0):
    read timestamp list or date range
    initialize MPI work queue

worker (rank k, 1 GPU):
    load LDEM, build mipmap pyramids        # once
    init SPICE                               # once
    while queue not empty:
        ts = queue.pop()
        compute sun, earth positions        # SPICE
        sun_u8, dsn_u8 = generate_live_shadow_frame_gpu(...)
        save PNG to shared FS
        report (ts, wall_time)

driver:
    aggregate timings, sha256 output, report
```

## Summary: standardize on live

The precompute approach's headline advantage is ~1.5× faster wall time
for a narrow scenario — single workstation, single observer height,
dense year-scale run at 20m. Everything else points the other way:

1. **Intermediate data product is a nightmare at scale.** 2.46 GB per
   region per observer height at 20m. Six heights × a few regions is
   already a working set of tens of GB that has to be versioned,
   transferred, and kept consistent. **At 1m it grows to roughly a
   terabyte** across heights and regions. Live has zero such artifact.
2. **Live is conceptually simpler.** One algorithm, one code path, no
   separate .bin file format, no precompute-vs-render phase split, no
   per-height rebuild. New contributors onboard faster. Debugging
   happens in one place.
3. **Live is more accurate.** Continuous per-pixel sun position instead
   of 1440-bucket (0.25°) quantization. No binning artifact at the
   terminator. Observer-height changes are free.
4. **Live is bit-reproducible across heterogeneous hardware.** 🎯
   Verified byte-exact output across Apple Silicon CPU ↔ Metal ↔ NVIDIA
   CUDA at 896×512 × 15 representative timestamps: **630 SHAs across
   every pipeline stage, 100% match** (kernel raw UInt8, palette-applied
   RGB, PNG roundtrip, Float32 diagnostics, CPU precompute). The
   pipeline uses only IEEE-mandated basic ops (`* + − / sqrt fma`),
   explicit `fma()` on every mul-add / mul-sub pattern (including
   across connected statements and inside every LUT helper — see the
   cross-vendor doc), squared-form comparisons to eliminate in-loop
   sqrt+div, orthonormality tricks to skip matrix rows, and LUT-based
   transcendentals. An IEEE-math guard runs before every frame and
   hard-aborts on any `fma`/`/`/`sqrt` bit-pattern drift. By construction
   the same guarantees extend to any IEEE 754 + hardware-FMA backend
   (AMD MI-series, Intel oneAPI, ARM/x86 CPU). Take the SHA-256 of an
   output directory; it's a deterministic function of (DEM, SPICE
   kernels, git SHA). Precompute required a separate validation step to
   establish the same property and remains sensitive to horizon-
   regeneration across hardware changes.
5. **Supercomputer access collapses the throughput gap.** Timestep
   parallelism is embarrassingly parallel — no inter-GPU communication,
   dynamic work queue handles the 3× cost variance between night and
   high-sun frames. 8× H100 → 2.5 min/year at 20m. 64× H100 →
   20 s/year at 20m, ~15 min/year at 1m.
6. **Live has headroom for further optimization.** The current
   implementation is a clean, correct, bit-exact baseline; it has not
   been architecturally optimized for spatial coherence across
   neighboring pixels (which at 1m scale are nearly identical). There
   are multiple known levers to make 1m live competitive with
   precompute on a single GPU.

**Recommendation: standardize all future shadow-map generation on the
live algorithm.** The precompute path remains in the codebase for
validation / historical comparison, but should not be the default for
new runs.
