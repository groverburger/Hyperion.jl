# Mapsets

Mapsets are the public workflow for Hyperion products. A mapset defines:

- output name and time selection,
- one or more terrain layers,
- output cadence,
- backend and tiling options.

The first layer defines the output grid. Additional layers are farfields used
after rays leave the nearfield tile.

## CLI

```bash
julia --project scripts/generate_mapset.jl --list-presets
```

Built-in presets:

```bash
julia --project scripts/generate_mapset.jl \
  --preset=nobile20m \
  --start=2027-01-22T07:00:00 \
  --stop=2027-01-22T07:00:00 \
  --backend=auto

julia --project scripts/generate_mapset.jl \
  --preset=viper8-shirley \
  --start=2027-01-22T07:00:00 \
  --stop=2027-01-22T07:00:00 \
  --backend=auto
```

Explicit timestamp lists are supported inline or from a text file:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=2027-01-22T07:00:00,2027-02-12T12:00:00

julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=scripts/test_timestamps.txt
```

Pass `--dataset-description` to write `other/dataset_description.json`.

## TOML Spec

```toml
name = "viper8_1m_shirley"
start = "2027-01-22T07:00:00"
stop = "2027-01-22T07:00:00"
step_hours = 1
tile_height = 1024
tile_width = 1024

[[layers]]
kind = "site"
path = "data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif"
name = "VIPER 8.0 Nobile 1m crop"
sha256 = "85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19"

[[layers]]
kind = "farfield"
path = "data/inputs/ldem_80s_20m.img"
name = "Shirley LDEM 80S 20m"
sha256 = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
height = 30400
width = 30400
pixel_size_m = 20.0
data_type = "int16"
elevation_scale_m = 0.5
byte_order = "little"
```

Layer SHA fields are optional, but supported specs should include them.
Site GeoTIFF sample type is detected at runtime. Headerless `.img`
farfields should declare their dimensions, pixel size, sample type, and
elevation scale because those cannot be inferred from the file.

## Library API

```julia
using Dates, Hyperion

spec = MapsetSpec(
    "viper8_1m_shirley",
    [
        SiteDEMLayer("data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif"),
        PolarDEMLayer("data/inputs/ldem_80s_20m.img"),
    ],
    DateTime("2027-01-22T07:00:00"),
    DateTime("2027-01-22T07:00:00");
    step = Hour(1),
)

generate_mapset(spec; backend = :auto)
```

Set `dataset_description = true` on `MapsetSpec` to also write
`other/dataset_description.json`.
