# Hyperion — live shadow generation (GPU, cross-platform)

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
cd Hyperion.jl
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

Both paths call the same `Hyperion.ensure_ldem!()` /
`ensure_test_data!()` functions, so they're safe to mix.

### Farfield LDEM versions

There are several south-polar 20 m/px LDEM artefacts that any far-field
shadow code should be aware of. They share the same projection (polar
stereographic, lat origin −90, lunar sphere R = 1737400 m) and 20 m
pixel spacing, but their pixel grids, elevation values, and physical
extent differ.

| Tag | File | SHA-256 | Format | Source |
|---|---|---|---|---|
| **Shirley** | (`/maps/lola_pds/LDEM_80S_20M-2017-06-15-processed.img`) | `caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b` | raw 30400×30400 int16 LE, scale 0.5 m, no header (1.85 GB) | Mark Shirley's pipeline (he maintained the upstream C# Mapbuilder until his retirement in 2025). Origin is unclear — believed to be a WUSTL 2017 download with one or more manual elevation patches. **This is the project's actual baseline** (despite the URL in `src/test_data.jl` pointing at the WUSTL Geosciences Node — see "Known download mismatch" below). |
| **WUSTL 2017** (live) | (downloaded by `Hyperion.ensure_ldem!()`) | unknown — different from Shirley | raw 30400×30400 int16 LE, scale 0.5 m, no header (1.85 GB) | [PDS Geosciences Node @ WUSTL — LOLA GDR](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/LDEM_80S_20M.IMG). The official, ongoing distribution. SHA differs from Shirley; a fresh `ensure_ldem!()` would fail SHA-verify against the pinned Shirley SHA. |
| **2023 Barker et al** | `LDEM_80S_20MPP_ADJ.TIF` | `09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649` | Cloud-Optimised GeoTIFF, Float32, DEFLATE (2.70 GB) | [GSFC PGDA product 90](https://pgda.gsfc.nasa.gov/products/90), Barker et al. 2023, *Planet. Sci. J.*, 4, 183. **Long-term migration target** (replaces Shirley as the canonical farfield once we re-pin the test suite). |
| **Barker on 2017 grid** | `ldem_80s_20m_2023.img` | `898a803d8f980731f01634af90a0eb659423c738fef787c4ce4f882a7c3f2398` | raw int16 LE, scale 0.5 m, same .img layout as Shirley (1.85 GB) | The full 2023 Barker product re-formatted across the entire 30400×30400: resampled onto the Shirley/WUSTL pixel grid AND re-referenced to the same sphere. Mean diff vs Shirley at the Nobile window: 0.12 m, max: 42.5 m. Not a patch — a wholesale replacement of all elevations. |
| **LNSI** ("Large Nobile Site of Interest") | `large_nobile.tif` (external, at `/Volumes/WD_BLACK/large_nobile.tif`) | (Float32 GeoTIFF) | 896×896 polar-stereographic 20 m crop, Float32 elevations in metres | **Byte-identical to Shirley at the Nobile window** (origin (63520, 130240), pixel offset (row 8688, col 18376)) — verified by direct comparison. So LNSI is a Shirley crop, not a Barker-derived product, despite the surrounding folder name. The C# Mapbuilder reference renders in `mapbuilder_large_nobile_sun/` rendered against this, hence against Shirley elevations. |

**This project currently uses Shirley** (SHA `caaf017f…`) — that's the
file at `data/inputs/ldem_80s_20m.img`, what `Hyperion.load_ldem`
returns, what the bit-exact test pin in `test/bitexact.jl` was
generated against, and (via LNSI) what the upstream C# Mapbuilder
reference renders in `mapbuilder_large_nobile_sun/` were produced
from. So Hyperion sim output and the Mapbuilder reference sim are
pinned to the same far-field DEM, and any sim-vs-sim drift isolates to
the kernel rather than the input data.

**Known download mismatch.** `src/test_data.jl` advertises the WUSTL
Geosciences Node URL as the LDEM source, but pins SHA `caaf017f…`
(Shirley). A fresh download from that URL would fail SHA verification
because WUSTL serves a different (live) version of the same nominal
product. We accept this as a known failure mode — the legacy Shirley
artefact is preserved as our test baseline so we can compare against
the C# Mapbuilder pipeline (which used the same Shirley file)
byte-for-byte. **Migration plan:** repin against either current WUSTL
2017 or 2023 Barker, regenerate the bit-exact tables, drop the legacy
Shirley dependency.

**Grid offset gotcha:** the native 2023 Barker grid is shifted 10 m W
and 10 m N from Shirley's grid. Shirley's .img has UL ≈ (−303980,
+303980); the native Barker TIF has UL = (−304000, +304000). The
"Barker on 2017 grid" file resamples Barker onto Shirley's grid so
it's a drop-in replacement; native-Barker crops are not. If you swap
LDEM versions in any pipeline, expect a half-pixel realignment to be
necessary downstream.

See `/Volumes/WD_BLACK/mapbuilder/test_inputs/dem_grid_notes.md`
(external to this repo, in the upstream Mapbuilder tree) for the
full provenance + Nobile-crop alignment details.

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
HYP_BACKEND=metal julia --project scripts/bitexact_test.jl   # or cuda / cpu
```

### Tier 0 correctness regression (opt-in, GPU required)

Beyond bit-exact regression, Hyperion has a **correctness regression**
test that scores its sun-fraction output against real LROC NAC
observations. The test requires a GPU backend (Metal or CUDA must be
loaded in your active environment) since each LNSI render takes ~200 s
on CPU; with Metal it's ~5 s/render (~5 min total for the full
25-NAC sweep).

Because `Pkg.test()` runs in a sub-environment that doesn't see
globally-installed Metal/CUDA, run the test directly via the
convenience runner:

```
HYP_RUN_CORRECTNESS=1 julia --project scripts/correctness/run_test.jl
```

This renders Hyperion at 25 NAC capture times (the bundled "Tier 0"
subset, ~1 MB at `test/fixtures/correctness/`), scores each render
against the Canny+Otsu-derived ground truth, and asserts the per-NAC
and dataset-aggregate metrics match the pinned baseline at
`test/fixtures/correctness/baseline_tier0.csv` to bit-exact (Hyperion
is cross-vendor bit-exact by design, and every downstream stage of
the comparison is IEEE 754 deterministic).

When you've intentionally changed kernel behaviour and want to
update the baseline:

```
julia --project scripts/correctness/refresh_baseline.jl
git diff test/fixtures/correctness/baseline_tier0.csv
git commit  # with a justification for the metric change
```

There are also two larger correctness data tiers, both external:

- **Tier 1** (~22 MB): full 599-NAC ground truth. Set
  `HYP_CORRECTNESS_TIER1_DIR` to a directory containing
  `shadow_20m/`, `sun_frac_20m/`, `timestamps.csv` to enable the
  statistically-robust evaluation. There is no default fallback —
  the env var must be set explicitly.
- **Tier 2** (~30 GB): raw LROC NAC `*.map.tif` orthoproducts. Used
  only for re-deriving Tier 1 from scratch (verification audits).

## Library API

```julia
using Hyperion
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
