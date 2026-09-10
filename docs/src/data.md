# Input data

The terrain files are external inputs under `data/inputs/`.
Git stores the TOML specifications in `data/inputs/mapsets/`, but it does not store their DEM files.
Hyperion does not download DEM files during tests or map generation.

## Terrain products

| Product | File | Use |
|---|---|---|
| Shirley 20 m | `ldem_80s_20m.img` | Current 20 m and 1 m regression farfield |
| Nobile 1 m | `nobile_1m.tif` | Full 1 m regression and baseline maps |
| VIPER 8.0 Nobile crop | `nobile_area_viper_sfs_dem_8_0_native_crop.tif` | VIPER crop mapsets |
| Barker 2023 20 m | `LDEM_80S_20MPP_ADJ.TIF` | Barker mapsets |
| Full VIPER 8.0 | `viper_sfs_dem_8_0.tif` | Dedicated radius/window command |
| VIPER medium extent | `viper_sfs_dem_8_0_v71_2027_medium_extent.tif` | Medium Barker mapsets |
| Mock 20 m hill | `mock_lunar_south_pole_20m_hill_128.tif` | Small synthetic mapset specification |

The mock hill raster is also external; its specification does not create it.
A specification gives the necessary path and, when present, its expected hash.
The [hash table](reference/input-data-hashes.md) lists the principal inputs.

## Formats and geometry

The Shirley file contains 30400 × 30400 signed 16-bit samples without a header.
Its byte order is little-endian.
Each stored count represents 0.5 m.
The file size is 1,848,320,000 bytes.

Site layers use stereographic GeoTIFF geometry.
The site loaders read the projection origin and pixel scale from the file.
A site layer does not accept every possible GeoTIFF projection.
Additional terrain layers use the south-polar stereographic grid.

The mapset loader keeps floating-point site data as Float32 meters.
For integer site sources, `load_site_dem` produces Int16 half-meter counts.
The internal function `load_site_dem_f32` produces Float32 meters.
A source elevation scale converts stored values to meters before geometry calculations.

For raw polar files, the TOML specification supplies dimensions, pixel size, sample type, elevation scale, and byte order.
For polar GeoTIFFs, the layer specification still supplies the polar grid configuration.
The file format alone does not establish a correct grid match.

## Data checks

1. Get the exact DEM files for the selected workflow.
2. Put each file at the path in its specification.
3. Compare each file hash with the expected value.
4. Run the mapset command with `--dry-run`.

On macOS or Linux:

```bash
shasum -a 256 data/inputs/ldem_80s_20m.img
```

On Windows PowerShell:

```powershell
Get-FileHash data\inputs\ldem_80s_20m.img -Algorithm SHA256
```

A current download of a similar LDEM product can have a different hash from the Shirley baseline.
The specified baseline bytes are necessary for the regression tests.
Examine a replacement file and update its references separately.

## Bundled data

Git stores the SPICE kernels and their file list under `kernels/`.
Git also stores the test PNGs, NAC masks, timestamps, and metric baseline.
The external LROC pipeline data is necessary only to rebuild the observation fixture.

The kernel list uses DE440s for Sun and Earth positions.
A local check got finite positions at 36 monthly dates across 2028–2030.
That check did not establish lunar summer intervals or validate landing-site illumination.
