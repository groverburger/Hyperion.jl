# Azimuth and elevation CSV

The CSV command calculates Sun and Earth geometry at one lunar location.
It uses the SPICE kernels in `kernels/`.
A DEM, a GPU, and `gdaldem` are not necessary for this command.
It does not calculate terrain shadows or solar-panel power.

## Export a time range

Run this example from the repository root:

```bash
julia --project scripts/generate_azel_csv.jl \
  --lat -85.47574 --lon 31.45666 \
  --start 2028-01-01T00:00:00 --stop 2028-01-02T00:00:00 \
  --step 1h --out /tmp/hyperion_azel.csv
```

Latitude and longitude use degrees.
Positive longitude is east.
Azimuth in the output increases counterclockwise from east.
Elevation is relative to the local horizontal plane.
`--elev <meters>` specifies the location elevation above the 1737.4 km reference sphere.
The default elevation is zero; it is not an automatic DEM sample.

The time range includes the stop time when it falls on the selected interval.
`--step` accepts `HH:MM:SS` or an integer with `s`, `m`, `h`, or `d`.
The timestamps refer to UTC.
The example dates do not define a lunar summer interval.

## Export a timestamp list

```bash
julia --project scripts/generate_azel_csv.jl \
  --lat -85.47574 --lon 31.45666 \
  --list test/fixtures/correctness/timestamps.csv \
  --out /tmp/hyperion_nac_azel.csv
```

The input CSV must have a `time` column, a `Start Time` column, or timestamps in its first column.
The supplied NAC fixture uses the `Start Time` column.
`--pixel <row> <column>` selects a Shirley-grid location instead of latitude and longitude.
Those pixel indices start at zero.
The pixel option uses the grid geometry without a DEM read.

## Output

The output contains UTC timestamps and these quantities:

- Sun and Earth azimuths and elevations, in degrees.
- Distance to the Sun, in kilometers and astronomical units.
- Distance to the Earth, in kilometers.
- Sun and Earth angular diameters, in degrees.

Without `--out`, the command writes the CSV to standard output.
`--kernels <directory>` selects another SPICE kernel directory.
`--help` shows the command options.

## Point illumination studies

A solar elevation curve does not include terrain shadows.
The renderer can calculate solar disk visibility at a selected DEM pixel.
But, the current commands do not export that result as a point light-curve CSV.
A dedicated point workflow must include coordinate conversion, terrain selection, a time loop, and a CSV writer.
The shadow calculation must keep the surrounding terrain.
