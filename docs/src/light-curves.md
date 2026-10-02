# Light curves

The light-curve command exports terrain-shadowed solar visibility for one location over time.
It uses the same ray calculation as the Sun maps.
The output is a CSV and a TOML metadata file.
A spreadsheet or plot program can display `sun_fraction` against `time_utc`.

## Make a curve

Run this example from the repository root:

```bash
julia --project scripts/generate_light_curve.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --lat=-85.47574 --lon=31.45666 \
  --start=2028-01-01T00:00:00 --stop=2028-04-01T00:00:00 \
  --step=1h --observer-height-m=0.5 \
  --out=data/outputs/site_2028.csv
```

These dates illustrate a time range; they do not define the lunar summer season.
Use separate output paths for studies in 2028, 2029, and 2030.
Select the study dates for the intended site and season.

The example uses two external DEM files from the selected specification.
Refer to [input data](data.md) for file requirements.
This command does not use `gdaldem` or make full map images.

## Select coordinates

Supply exactly one pair of coordinate options:

| Options | Meaning |
|---|---|
| `--lat=<degrees> --lon=<degrees>` | Lunar latitude and east-positive longitude |
| `--x=<meters> --y=<meters>` | Projected easting and northing in the first DEM grid |
| `--col=<integer> --row=<integer>` | Column and row indices in the full first DEM, starting at zero |

For example, replace the latitude and longitude options with `--col=2285 --row=4663` for a pixel selection.
The pixel must be inside the selected DEM.
Longitude can range from −180° to 360°.

Latitude/longitude and X/Y inputs select the nearest pixel center.
At an exact cell boundary, the command selects the larger pixel index.
The command reports the selected coordinates before it starts the calculation.
The CSV and metadata file also record the selected coordinates.
The command does not interpolate a light curve between pixels.

Site coordinates use the supported stereographic projection in the GeoTIFF.
Site grids must have square pixels, no rotation, and north at the top.
Polar coordinates use Hyperion's south-polar grid, with the zero-easting and zero-northing indices at 15199.5.
The pixel size comes from the layer specification.
The mapset loader does not derive a different polar grid from a GeoTIFF transform.

## Select times and height

`--start` and `--stop` are mandatory UTC timestamps.
The time range includes the stop time when it falls on the selected interval.
`--step` defaults to one hour.
It accepts a positive integer with `s`, `m`, `h`, or `d`, or an `HH:MM:SS` interval.
For example, `--step=15m` selects 15-minute samples.

The command ignores the mapset's timestamp list, start time, stop time, and sampling intervals.
This rule prevents a short example mapset from changing the requested study period.

`--observer-height-m` specifies height above the terrain at the selected pixel.
Its default comes from `observer_height_m` in the specification, or zero if that field is absent.
The command gets terrain elevation from the DEM sample.

## Terrain extent and memory

The command keeps the full first DEM for shadow calculations.
It overrides that layer's output window and sets its site cutoff to false.
A one-pixel output does not remove nearby terrain that can block sunlight.
The remaining layers supply distant terrain after a ray leaves the inner DEM.

A layered specification must start with a site DEM and can have up to two outer polar layers.
Only the first layer can specify an output window.
A single site or polar layer is also permitted.
The existing loader constraints still apply, including the site-only mipmap dimensions.

The command loads terrain and builds mipmaps once.
It also reuses the device terrain arrays across timestamps.
Large terrain files can still use a large amount of memory and take time to prepare.
A one-pixel output does not make the terrain files smaller.

## Backend and checks

The default backend is CPU.
Use `--backend=metal`, `--backend=cuda`, or `--backend=auto` to select another backend.
An explicit GPU request fails if that backend cannot load.
Automatic selection can fall back to CPU and reports the actual backend.

Install Metal or CUDA in the active Julia environment or the default environment for that Julia version.
The dynamic loader supports Julia 1.12 global bindings.

1. Add `--dry-run` to resolve the coordinates and time range.
2. Examine the reported pixel and sample count.
3. Remove `--dry-run` to calculate the curve.

The dry run checks the terrain paths and supplied hashes.
It does not load full terrain arrays, test the backend, or establish SPICE coverage for the requested dates.
`--kernels=<directory>` selects another kernel directory.
`--help` shows all command options.
Options accept `--key=value` and `--key value` forms.

## Output

| CSV field | Meaning |
|---|---|
| `time_utc` | Sample time in UTC |
| `row`, `col` | Selected pixel indices |
| `latitude_deg`, `longitude_deg` | Selected pixel center on the reference sphere |
| `x_m`, `y_m` | Selected pixel center in projected meters |
| `terrain_elevation_m` | DEM elevation above the 1,737.4 km reference sphere |
| `observer_height_m` | Observer height above that DEM sample |
| `sun_u8` | The Sun-map value from 0 to 255 |
| `sun_fraction` | `sun_u8 / 255`; visible solar-disk fraction with map quantization |
| `sun_azimuth_deg` | Geometric Sun azimuth, counterclockwise from east |
| `sun_elevation_deg` | Geometric Sun elevation above the local horizontal plane |

Zero solar fraction means full shadow in the model; one means full solar-disk visibility.
Intermediate values describe partial solar-disk visibility.
The calculation uses eight Sun rays and the existing fixed solar half-angle of 0.27°.
The fraction does not include irradiance, surface-incidence losses, panel orientation, or electrical power.
Geometric Sun elevation does not include terrain obstruction.

The metadata path is the CSV path with `.toml` appended.
It records the input hashes, effective layer settings, selected pixel, time range, observer height, Julia version, and actual backend.
Existing outputs cause an error unless you supply `--overwrite`.
The command writes temporary files first and installs the outputs after a successful calculation.
It does not continue a partial run.

## Small tests

```bash
julia --project test/light_curve.jl
```

These tests create small synthetic terrain files and use the bundled SPICE kernels.
They compare coordinate modes, cached output, map pixels, CSV values, and metadata.
They do not use external DEMs.
The standard test suite also includes them.
