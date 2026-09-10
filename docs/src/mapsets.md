# Mapsets

A mapset contains Sun images, DSN images, and metadata for selected times.
The first terrain layer defines the output grid.
A ray continues into the next layer after it leaves the loaded terrain raster.
An output tile boundary does not end the terrain raster.

## Make a mapset

1. Put the necessary terrain files in `data/inputs/`.
2. Select a TOML file from `data/inputs/mapsets/`.
3. Do a check of the specification:

   ```bash
   julia --project scripts/generate_mapset.jl \
     --spec=data/inputs/mapsets/nobile_20m_shirley.toml --dry-run
   ```

4. Make the images:

   ```bash
   julia --project scripts/generate_mapset.jl \
     --spec=data/inputs/mapsets/nobile_20m_shirley.toml --backend=auto
   ```

`--dry-run` checks file paths and optional hashes, but it does not test the renderer.
For map generation, `gdaldem` must be on `PATH` to make slope and hillshade files.
The command accepts `auto`, `metal`, `cuda`, and `cpu` backends.
An unavailable GPU backend can cause a CPU fallback.
You must supply `--spec` to the current command.
The previous preset options are no longer available.

## Time selection

The range includes the stop time if that time falls on the selected interval.
All timestamps refer to UTC.

Override the range with these options:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml \
  --start=2028-01-01T00:00:00 --stop=2028-01-02T00:00:00
```

Select explicit timestamps with either command:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=2027-01-22T07:00:00,2027-02-12T12:00:00

julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --times=scripts/test_timestamps.txt
```

A timestamp file contains one time per line.
The parser ignores empty lines and text after `#`.
`--times` takes precedence over the TOML time selection.
A TOML `times` list takes precedence over a range.

The TOML values `step_hours` and `azel_step_hours` take precedence over the corresponding command options.
Change those TOML values to change an existing specification's intervals.
The mapset command uses whole-hour intervals.
The library API also accepts other positive `Dates.Period` values.

## TOML configuration

```toml
name = "viper8_1m_shirley"
start = "2027-01-22T07:00:00"
stop = "2027-01-22T07:00:00"
step_hours = 1
azel_step_hours = 1
observer_height_m = 0.0
tile_height = 1024
tile_width = 1024

[[layers]]
kind = "site"
path = "data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif"
sha256 = "85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19"
cutoff = false

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

Relative layer paths start at the repository root, not at the TOML directory.
A layer can specify `window = [row, column, height, width]`.
Row and column indices start at zero.
The first layer can be a site DEM or a south-polar DEM.
A mapset with multiple layers must start with a site DEM.
It can then have one or two south-polar layers.
Only the first layer can specify an output window.

For a site layer, `cutoff = false` keeps the full site terrain for shadow calculations.
`cutoff = true` loads only the selected window.
Thus, a cutoff changes the available terrain, not only the output size.
Tiling limits output buffers while it keeps the loaded terrain.

The `sha256` field is optional.
The supplied specifications include hashes to identify their external inputs.
The [data guide](data.md) explains file formats and height scales.

## Output and continuation

The default output path is `data/outputs/<name>/`.
`--out=<path>` changes the output root.

| File | Content |
|---|---|
| `sun/sun.<timestamp>.png` | Solar disk visibility |
| `dsn/dsn.<timestamp>.png` | Earth elevation above the terrain horizon |
| `other/hillshade.tif` | Terrain hillshade |
| `other/slope.tif` | Terrain slope |
| `other/azimuths_elevations.csv` | Sun and Earth geometry at the output window center |
| `other/manifest.csv` | Configuration and run metadata |
| `other/timestamps.txt` | Explicit timestamps; absent in range mode |

A normal run keeps full image pairs.
A run with one missing image calculates the frame and saves the missing image.
`--overwrite` replaces existing images.
The command does not compare an existing image with a changed specification.
After an input or configuration change, use a new output name or `--overwrite`.

`--dataset-description` adds `other/dataset_description.json`.
An explicit timestamp list must have regular intervals for that file.

## Supplied specifications

`nobile_20m_shirley.toml` and `viper8_shirley_range.toml` each select one timestamp by default.
`v8_medium_barker_sep.toml` currently selects two frames on 2027-09-14.
Its output name is `v8_medium_barker`.
The file name does not define the active time range.
External terrain files are also necessary for the NAC-time and mock-hill specifications.

## Library API

```julia
using Dates, Hyperion

spec = MapsetSpec(
    "nobile_window",
    [PolarDEMLayer("data/inputs/ldem_80s_20m.img";
        window = (8960, 18432, 512, 896))],
    DateTime(2027, 1, 22, 7),
    DateTime(2027, 1, 22, 7);
    step = Hour(1),
)
generate_mapset(spec; backend = :auto)
```

The exported API names are `MapsetSpec`, `SiteDEMLayer`, `PolarDEMLayer`, and `generate_mapset`.
Other module functions are internal interfaces.

## VIPER window command

The dedicated command uses the full VIPER 8.0 DEM and Shirley terrain.
Its default output window is 500 × 500 pixels.

```bash
julia --project scripts/generate_viper8_1m_radius_shirley_mapset.jl \
  --from-lat-lon --lat=-85.467 --lon=32.015 --radius-m=250 \
  --no-cutoff --dry-run
```

`--from-lat-lon` enables the latitude and longitude options.
Without that option, the command uses its fixed pixel origin.
The command defaults to cutoff operation.
`--no-cutoff` keeps the full site terrain for shadow calculations.
