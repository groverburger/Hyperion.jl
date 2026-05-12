# Hyperion: live lunar illumination and shadow generation

Hyperion answers a practical lunar-planning question: for a patch of
Moon at a particular time, which parts can see the Sun, and which parts
can see Earth?

That question matters most near the lunar poles. The Sun stays low on
the horizon there, so small ridges and crater rims can decide whether a
site is lit, shadowed, power-positive, thermally stable, or able to talk
to Earth. Hyperion takes a terrain model, a time, and SPICE ephemerides,
then renders illumination and Earth-visibility maps directly on the GPU.

The main output is a sun-fraction image. A value of `0.0` means the
solar disk is fully blocked by terrain, `1.0` means it is fully visible,
and values in between mean partial illumination. Hyperion can also emit
binary lit/shadow masks, Earth visibility maps, DSN-style communication
maps, and diagnostic images used by the test suite.

The current branch (`1m-shadows`) contains the live
KernelAbstractions GPU pipeline, 20 m south-polar LDEM support, and
native high-resolution site DEM support. The older precomputed-horizons
pipeline and the older hand-written CPU live path have been removed.
The KA CPU backend still exists as a slow reference backend.

Hyperion is not tied to one output resolution. The same live kernel is
parameterized by DEM origin, pixel size, projection frame, and elevation
scale, so it can render the 20 m LOLA south-polar LDEM, native 1 m
site DEMs, and other local products such as 5 m DEMs when they are
stereographic GeoTIFFs with dimensions compatible with the mipmap
pyramid. The bundled tests pin both the 20 m LDEM path and the 1 m
site-DEM path.

Some terms used below:

- A **DEM** is a digital elevation model: a raster image where each
  pixel stores terrain height instead of color.
- The **LDEM** is the LOLA south-polar lunar DEM used for far-field
  horizon and shadow casting.
- **LROC NACs** are high-resolution Lunar Reconnaissance Orbiter camera
  images. Hyperion uses derived NAC shadow maps as observational checks.
- **SPICE** kernels provide the Sun, Earth, Moon, and spacecraft
  geometry used to render a particular time.

## Bit-exactness across hardware

The GPU kernel uses only IEEE-754-mandated ops (+, -, *, /, sqrt, fma)
plus LUT-based transcendentals. **The output is bit-identical on any
backend KernelAbstractions supports**: Apple Metal, NVIDIA CUDA, AMD
ROCm, Intel oneAPI, and the CPU fallback. The SHA-256 of an output
directory is a deterministic function of the DEM, SPICE kernels,
backend-independent code, and git SHA.

## Install

Library deps are backend-agnostic (no Metal / CUDA in `Project.toml`).
Fresh clone to running tests is three commands:

```
git clone <repo-url>
cd Hyperion.jl
git checkout 1m-shadows
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite expects the Shirley LDEM to already be present at
`data/inputs/ldem_80s_20m.img`. It must be the raw 30400x30400 int16 LE
artifact with SHA:

```
caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b
```

If it is missing or has the wrong SHA, the LDEM-dependent tests are
skipped or scripts fail with placement instructions. The test harness
does not download DEM data. To validate the local file and derive the
Nobile GeoTIFF used by diagnostic scripts:

```
julia --project scripts/fetch_test_data.jl
```

This script does not download data.

The Tier 0 correctness runner can also be pointed at another local LDEM
without replacing the Shirley baseline file:

```
HYP_BACKEND=metal HYP_LDEM_PATH=data/inputs/LDEM_80S_20MPP_ADJ.TIF \
    julia --project scripts/correctness/run_test.jl
```

### Farfield LDEM versions

There are several south-polar 20 m/px LDEM artefacts that any far-field
shadow code should be aware of. They share the same projection (polar
stereographic, latitude origin -90, lunar sphere R = 1737400 m) and 20 m
pixel spacing, but their pixel grids, elevation values, and physical
extent differ.

| Tag | File | SHA-256 | Format | Source |
|---|---|---|---|---|
| **Shirley** | (`/maps/lola_pds/LDEM_80S_20M-2017-06-15-processed.img`) | `caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b` | raw 30400x30400 int16 LE, scale 0.5 m, no header (1.85 GB) | Mark Shirley's pipeline (he maintained the upstream C# Mapbuilder until his retirement in 2025). Origin is unclear, but it is believed to be a WUSTL 2017 download with one or more manual elevation patches. **This is the project's actual baseline.** |
| **WUSTL 2017** (live) | `LDEM_80S_20M.IMG` | unknown, different from Shirley | raw 30400x30400 int16 LE, scale 0.5 m, no header (1.85 GB) | [PDS Geosciences Node @ WUSTL, LOLA GDR](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/LDEM_80S_20M.IMG). The official, ongoing distribution. SHA differs from Shirley; do not use it with the current pins. |
| **2023 Barker et al** | `LDEM_80S_20MPP_ADJ.TIF` | `09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649` | Cloud-Optimised GeoTIFF, Float32 elevations in metres, DEFLATE (2.70 GB) | [GSFC PGDA product 90](https://pgda.gsfc.nasa.gov/products/90), Barker et al. 2023, *Planet. Sci. J.*, 4, 183. Supported directly by `load_ldem`; currently the strongest known far-field DEM for correctness statistics, but not yet the bit-exact baseline. |
| **Barker on 2017 grid** | `ldem_80s_20m_2023.img` | `898a803d8f980731f01634af90a0eb659423c738fef787c4ce4f882a7c3f2398` | raw int16 LE, scale 0.5 m, same .img layout as Shirley (1.85 GB) | The full 2023 Barker product re-formatted across the entire 30400x30400: resampled onto the Shirley/WUSTL pixel grid and re-referenced to the same sphere. Mean diff vs Shirley at the Nobile window: 0.12 m, max: 42.5 m. Not a patch, but a wholesale replacement of all elevations. |
| **LNSI** ("Large Nobile Site of Interest") | `large_nobile.tif` (external, at `/Volumes/WD_BLACK/large_nobile.tif`) | (Float32 GeoTIFF) | 896x896 polar-stereographic 20 m crop, Float32 elevations in metres | **Byte-identical to Shirley at the Nobile window** (origin (63520, 130240), pixel offset (row 8688, col 18376)), verified by direct comparison. So LNSI is a Shirley crop, not a Barker-derived product, despite the surrounding folder name. The C# Mapbuilder reference renders in `mapbuilder_large_nobile_sun/` used this crop, so they used Shirley elevations. |

**This project still uses Shirley for bit-exact pins** (SHA
`caaf017f...`). That is the file expected at
`data/inputs/ldem_80s_20m.img`, what the 20 m bit-exact fixtures in
`test/bitexact.jl` were generated against, and (via LNSI) what the
upstream C# Mapbuilder reference renders in
`mapbuilder_large_nobile_sun/` were produced from. So Hyperion sim
output and the Mapbuilder reference sim can still be compared on the
same far-field DEM when isolating kernel differences.

`Hyperion.load_ldem` now supports both the Shirley/WUSTL raw `.img`
layout and the native Barker GeoTIFF layout:

- raw `.img`: memory-mapped `Int16`, elevation scale `0.5 m/count`
- `.tif` / `.tiff`: read with ArchGDAL as `Float32`, elevation scale
  `1.0 m/unit`

**No auto-download.** WUSTL serves a different live version of the same
nominal product, so this repo no longer attempts to download an LDEM.
Place the Shirley artifact manually at `data/inputs/ldem_80s_20m.img`.
The legacy Shirley artifact is preserved as our test baseline so we can
compare against the C# Mapbuilder pipeline (which used the same Shirley
file) byte-for-byte. **Migration plan:** repin against either current
WUSTL 2017 or 2023 Barker, regenerate the bit-exact tables and Tier 0
baseline, then drop the legacy Shirley dependency.

**Grid offset gotcha:** the native 2023 Barker grid is shifted 10 m W
and 10 m N from Shirley's grid. Shirley's .img has UL approximately
(-303980, +303980); the native Barker TIF has UL = (-304000, +304000). The
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
KA CPU backend at 20 representative timestamps (full 896x512 each) and
verifies:
- SHA-256 of every intermediate (sun, dsn, palette-applied RGB,
  Float32 diagnostics, azel precompute) against a hardcoded known-good
  table, and
- pixel equality of the decoded PNG against 40 committed reference
  fixtures in `test/fixtures/bitexact/`.

Runtime: ~10-15 minutes. The same invariants hold bit-for-bit on
Apple Metal and NVIDIA CUDA. See
[`docs/cross-vendor-determinism.md`](docs/cross-vendor-determinism.md).

For a cross-vendor forensic audit (actual Metal or CUDA hardware +
raw .bin buffers), use `scripts/bitexact_test.jl`:

```
HYP_BACKEND=metal julia --project scripts/bitexact_test.jl   # or cuda / cpu
```

### Correctness evaluation against LROC NACs

Beyond bit-exact regression, Hyperion has a **correctness evaluation**
that scores sun-fraction output against real LROC NAC observations.
The test requires a GPU backend (Metal or CUDA must be loaded in your
active environment) since each LNSI render takes ~200 s on CPU; with
Metal it is usable interactively.

The default test suite runs the Tier 0 correctness check unless
`HYP_SKIP_CORRECTNESS=1` is set. You can also run it directly via the
convenience runner:

```
HYP_BACKEND=metal julia --project scripts/correctness/run_test.jl
```

This renders Hyperion at 25 NAC capture times (the bundled "Tier 0"
subset, ~1 MB at `test/fixtures/correctness/`), scores each render
against the Canny+Otsu-derived ground truth, and compares the result
with `test/fixtures/correctness/baseline_tier0.csv`.

Each run writes:

- `input_shas.csv`: SHA-256 of the input DEM
- `current.csv`: per-NAC metrics for this run
- `delta.csv`: per-NAC metric deltas versus the pinned baseline
- `summary.csv`: median, mean, and quality-score rows

The pass/fail gate is now aggregate and direction-aware: every median
metric must be at least as good as the pinned baseline. Per-NAC
regressions are still reported in `delta.csv`, but they do not fail the
test if the corresponding median metrics pass.

The current quality score row is
`core_mean_relative_improvement`, the mean relative improvement across
the core median metrics:

- `ber`
- `iou_shadow`
- `ssim_continuous`
- `mae`
- `p99_abs_err`

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
  statistically-robust evaluation. There is no default fallback;
  the env var must be set explicitly.
- **Tier 2** (~30 GB): raw LROC NAC `*.map.tif` orthoproducts. Used
  only for re-deriving Tier 1 from scratch (verification audits).

Current broad-set Barker numbers, measured on the 599-NAC Tier 1 set
at `/Volumes/WD_BLACK/lroc-nac-maps/derived/`, are:

- median continuous illumination-fraction agreement:
  `1 - median(MAE) = 97.18%`
- median binary lit/shadow pixel agreement: `97.68%`
- median continuous MAE: `2.82 percentage points`
- 87.6% of NACs have continuous MAE <= 5 percentage points
- 98.8% of NACs have binary pixel agreement >= 90%

Compared to `/Volumes/WD_BLACK/mapbuilder_large_nobile_sun`, the
Barker/Hyperion exact-time run improves median MAE by about 10.8% and
raises median binary pixel agreement from 97.38% to 97.68%. That
comparison is not perfectly apples-to-apples because the Mapbuilder
folder is a 2-hour bucketed time series (median NAC time offset
~50 minutes), while Hyperion's Tier 1 run renders exact NAC timestamps.

## Library API

### 20 m far-field LDEM

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

To use the native Barker GeoTIFF instead of the Shirley raw IMG:

```julia
ldem = load_ldem("data/inputs/LDEM_80S_20MPP_ADJ.TIF")
max_mm, min_mm = build_ldem_mipmaps_minmax(ldem.data)

sun, dsn = generate_live_shadow_frame_gpu(
    ldem.data, 8688, 18376, 896, 896,
    sun_pos, earth_pos, 0.0;
    max_mipmaps = max_mm, min_mipmaps = min_mm,
    backend = Metal.MetalBackend(),
    DeviceArray = Metal.MtlArray,
    elev_scale_to_m = ldem.elev_scale_to_m,
)
```

### High-resolution site DEMs

Use `load_site_dem` or `load_site_dem_f32` for local stereographic
GeoTIFF site DEMs. The pixel size is read from the GeoTIFF transform,
so the same path works for 1 m, 5 m, or other local resolutions. The
DEM is kept in its native projection and pixel grid; Hyperion does not
force it onto the 20 m LDEM grid.

```julia
using Hyperion
using Metal

site = load_site_dem_f32("/path/to/site_5m_or_1m.tif")
max_mm, min_mm = build_site_mipmaps_minmax(site)

sun, dsn, _, _ = generate_live_shadow_frame_site_gpu(
    site, sun_pos, earth_pos, 0.0;
    max_mipmaps = max_mm,
    min_mipmaps = min_mm,
    backend = Metal.MetalBackend(),
    DeviceArray = Metal.MtlArray,
    origin_r = 0,
    origin_c = 0,
    H = site.H,
    W = site.W,
)
```

`load_site_dem` quantizes elevations to Int16 half-metres, matching the
20 m LDEM path and minimizing memory. `load_site_dem_f32` preserves
Float32 metre elevations for DEMs where sub-half-metre precision is
worth the extra memory. Site DEM dimensions must be divisible by 16 so
the 5-level min/max mipmap pyramid can be built.

## Scripts

- `scripts/bitexact_test.jl`: cross-vendor forensic harness: 20
  timestamps x full 896x512 x backend-of-your-choice, SHAs + raw .bin
  buffers + PNGs. Pair with `scripts/diff_bitexact_{shas,pixels}.jl`
  for N-way pairwise comparison across `data/outputs/bitexact/`.
- `scripts/bench_workgroup.jl`: sweep workgroup sizes {128, 256, 512}
- `scripts/generate_live_year.jl`: full year (2h cadence) with per-ts
  timing and SSIM vs an optional precomputed reference directory
- `scripts/render_1m_demo.jl`: render a native high-resolution site
  DEM crop or full tile
- `scripts/fetch_test_data.jl`: validate local Shirley and derive the
  Nobile GTiff used by diagnostic scripts
- `scripts/azimuth_range.jl`, `scripts/select_test_timestamps.jl`:
  diagnostic helpers

## Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```

Covers deterministic-math LUTs, Float32 stereographic projection,
mipmap pyramid shape, and the canonical cross-platform regression:
20-timestamp bit-exactness check on the KA CPU backend, including
SHA-256 of all intermediates + pixel-exact comparison against 40
committed PNG fixtures. See
[`docs/cross-vendor-determinism.md`](docs/cross-vendor-determinism.md)
for the theory and verification protocol.
