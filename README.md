# Hyperion.jl

Live lunar illumination and Earth-visibility map generation.

Hyperion takes lunar DEMs, SPICE ephemerides, and a time selection, then
renders Sun and DSN visibility maps. The main product is a **mapset**: a
reproducible directory of timestamped PNG frames plus metadata.

## What It Does

- Renders 20 m Shirley/LOLA south-polar mapsets.
- Renders VIPER 8.0 1 m site mapsets with Shirley 20 m farfield continuation.
- Supports time ranges and explicit timestamp lists.
- Preserves deterministic, bit-exact GPU behavior across supported backends.
- Tests correctness against a committed LROC NAC Tier 0 fixture.

## Install

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

GPU packages are intentionally not in `Project.toml`. Install the backend you
use in your global Julia environment:

```bash
julia -e 'using Pkg; Pkg.add("Metal")'  # Apple Silicon
julia -e 'using Pkg; Pkg.add("CUDA")'   # NVIDIA
```

## Required Data

Hyperion does not download DEMs during tests or generation. Place inputs under
`data/inputs/`.

| Product | Path | SHA-256 |
|---|---|---|
| Shirley 20 m LDEM | `data/inputs/ldem_80s_20m.img` | `caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b` |
| VIPER 8.0 Nobile crop | `data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif` | `85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19` |
| Nobile 1 m fixture | `data/inputs/nobile_1m.tif` | `e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4` |

Mapset specs include SHA checks so accidental input swaps fail early.
Detailed provenance lives in the docs.

## Generate Mapsets

20 m Shirley Nobile extent, 896 x 512 pixels:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml \
  --backend=auto --overwrite
```

VIPER 8.0 1 m site DEM with Shirley 20 m farfield:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --backend=auto
```

Explicit timestamps:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=2027-01-22T07:00:00,2027-02-12T12:00:00
```

Or use a timestamp file:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=scripts/test_timestamps.txt
```

Outputs are written to `data/outputs/<mapset-name>/`:

```text
sun/sun.<timestamp>.png
dsn/dsn.<timestamp>.png
other/hillshade.tif
other/slope.tif
other/azimuths_elevations.csv
other/manifest.csv
```

Pass `--dataset-description` to also write
`other/dataset_description.json`.

## Define New Mapsets

Create a TOML file with a name, time selection, and layer list:

```toml
name = "my_site_shirley"
start = "2027-01-22T07:00:00"
stop = "2027-01-23T07:00:00"
step_hours = 1

[[layers]]
kind = "site"
path = "data/inputs/my_site.tif"
sha256 = "<expected hash>"

[[layers]]
kind = "farfield"
path = "data/inputs/ldem_80s_20m.img"
sha256 = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
height = 30400
width = 30400
pixel_size_m = 20.0
data_type = "int16"
elevation_scale_m = 0.5
byte_order = "little"
```

Then run:

```bash
julia --project scripts/generate_mapset.jl --spec=path/to/spec.toml
```

The library API is the same idea: build a `MapsetSpec` from
`SiteDEMLayer(...)` and `PolarDEMLayer(...)`, then call `generate_mapset`.

## Test

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The normal test suite covers deterministic math, projection helpers, mipmaps,
bit-exact pinned frames, and the Tier 0 correctness gate when required data
and a GPU backend are available.

Skip the expensive correctness sweep:

```bash
HYP_SKIP_CORRECTNESS=1 julia --project -e 'using Pkg; Pkg.test()'
```

## Maintenance Tools

Fixture and audit tooling is in `tools/`.

```bash
julia --project tools/fixtures/build_tier0_fixture.jl
julia --project tools/fixtures/refresh_correctness_baseline.jl
HYP_BACKEND=cuda julia --project tools/bitexact/bitexact_test.jl
julia --project tools/bitexact/diff_bitexact_shas.jl
```

These tools record or validate hashes where external data enters the workflow.

## Documentation

Documenter.jl docs live under `docs/`:

```bash
julia --project=docs docs/make.jl
```

Start with:

- `docs/src/mapsets.md`
- `docs/src/correctness.md`
- `docs/src/bitexact.md`
- `docs/src/data.md`
- `docs/src/raycasting.md`

Older research notes are retained under `docs/src/reference/`.
