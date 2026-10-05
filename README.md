# Hyperion.jl

Hyperion makes lunar illumination maps from digital elevation models (DEMs) and SPICE ephemerides.
A mapset contains Sun maps, Earth visibility maps, and metadata for selected times.
The current package version is 0.1.0.

## Functions

- Make maps from a south-polar DEM or a site DEM with distant terrain layers.
- Select a time range or a list of timestamps.
- Continue a mapset run after an interruption.
- Export Sun and Earth azimuths and elevations for a specified location.
- Compare output with stored images and lunar observations.

The Sun output gives the visible fraction of the solar disk, with values from 0 to 255.
The DSN output gives the Earth elevation above the terrain horizon.
These outputs do not give solar-panel power or a full communication link assessment.
The [light-curve command](docs/src/light-curves.md) exports these solar fractions over time for one selected location.

## Installation

Julia 1.11.5 and 1.12.7 passed the targeted output comparisons.
GPU packages must be available in the environment for the selected Julia version.

1. Install the project dependencies:

   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

2. Install the GPU package for your hardware in the default Julia environment:

   ```bash
   julia -e 'using Pkg; Pkg.add("Metal")'  # Apple Silicon
   julia -e 'using Pkg; Pkg.add("CUDA")'   # NVIDIA
   ```

3. For mapsets, install the GDAL command-line tools.

4. Make sure that `gdaldem` is on `PATH`.

A GPU package is not necessary for CPU operation.
The mapset command can use the CPU if the requested GPU backend is unavailable.
CPU map generation can take a long time.
Git does not store `Manifest.toml`, so a new installation can select different dependency versions.

## Input data

Git stores the SPICE kernels, test reference images, and mapset specifications.
Git does not store the terrain files in `data/inputs/`.
Hyperion does not download terrain files.
The tests find their terrain files by SHA-256, so those files can have any name in `data/inputs/`.
Mapset specifications use the paths in the TOML file.

| Operation | Necessary terrain files in `data/inputs/` |
|---|---|
| Shirley 20 m maps and regressions | `ldem_80s_20m.img` |
| VIPER 8.0 crop with Shirley terrain | `nobile_area_viper_sfs_dem_8_0_native_crop.tif`, `ldem_80s_20m.img` |
| Full 1 m regression | `nobile_1m.tif`, `ldem_80s_20m.img` |
| Other mapsets | The files specified in the selected TOML file |

The [data guide](docs/src/data.md) explains file formats and data checks.
The [input hash table](docs/src/reference/input-data-hashes.md) gives the expected SHA-256 values.

## Map generation

Run this command from the repository root:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml \
  --backend=auto
```

The example makes one 896 × 512 pixel frame for 2027-01-22T07:00:00.
Output goes to `data/outputs/nobile_20m_shirley/`.
Existing full frames remain unchanged.

Use `--dry-run` for a check of the specification and input hashes without map generation.
Use `--overwrite` to replace existing frames.
Use `--backend=cuda --gpus=6` to divide timestamps across six visible NVIDIA GPUs on one node.
The command writes one mapset and requires no manual merge.
Use `--help` to show the command options.

The [mapset guide](docs/src/mapsets.md) explains layers, timestamps, tiles, and output files.

## Light curves

```bash
julia --project scripts/generate_light_curve.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml \
  --row=9216 --col=18880 \
  --start=2028-01-01T00:00:00 --stop=2028-04-01T00:00:00 \
  --step=1h --out=data/outputs/site_2028.csv
```

Use `--lat` and `--lon`, or projected `--x` and `--y`, instead of pixel indices.
The command keeps the full terrain and exports solar visibility, with input and location metadata.
These example dates do not define a lunar summer season.
Refer to the [light-curve guide](docs/src/light-curves.md) for coordinates, terrain selection, and output fields.

## Azimuth and elevation CSV

Run this example to export geometry for one location:

```bash
julia --project scripts/generate_azel_csv.jl \
  --lat -85.47574 --lon 31.45666 \
  --start 2028-01-01T00:00:00 --stop 2028-01-02T00:00:00 \
  --step 1h --out /tmp/hyperion_azel.csv
```

This example does not define a lunar summer interval.
The CSV does not include terrain shadows.
The [CSV guide](docs/src/azel.md) explains the coordinate and time options.

## Tests

Run the small terrain tests without external data:

```bash
HYP_BACKEND=cpu julia --project test/terrain_stack.jl
```

Run the full suite only when the necessary data and compute resources are available:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The full suite can take a long time.
Some tests do not run if data or a GPU backend is unavailable.
`HYP_SKIP_CORRECTNESS=1` disables the observation comparison, but the large frame regressions remain enabled.

The [test guide](docs/src/testing.md) gives test scope and requirements.

## Documentation

- [Workflow commands](docs/src/tooling.md)
- [Mapset configuration](docs/src/mapsets.md)
- [Data requirements](docs/src/data.md)
- [Tests](docs/src/testing.md)
- [Ray casting](docs/src/raycasting.md)
- [Documentation terms and style](docs/src/terms.md)

Build the HTML documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The output directory is `docs/build/`.
The reference pages include dated investigation results.
Those results apply to the recorded code and test conditions.
