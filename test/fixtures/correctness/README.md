# Tier 0 correctness fixtures

This directory contains reference masks for 25 Lunar Reconnaissance Orbiter Camera (LROC) Narrow Angle Camera (NAC) observations.
Git stores these files.
The external Shirley DEM and a supported GPU backend are also necessary for a correctness run.
The bundled masks alone are not sufficient to run that test.

## Files

| Path | Contents |
|---|---|
| `shadow_20m/<id>.tif` | UInt8 masks: 0 shadow, 255 illuminated, 128 NoData |
| `sun_frac_20m/<id>.tif` | Float32 solar fractions from 0 to 1; NaN means NoData |
| `timestamps.csv` | Observation times in UTC |
| `selection.csv` | Selection reasons and comparison metrics |
| `baseline_tier0.csv` | Stored correctness metrics |

The fixture contains 50 mask images and four supporting files, including this README.
`Hyperion.Correctness` uses only this bundled Tier 0 fixture.

## Selection

The external LROC pipeline selected observations across these categories:

| Category | Count |
|---|---|
| Lower balanced error rate than Mapbuilder | 5 |
| Improvements with exact observation times | 3 |
| Higher balanced error rate than Mapbuilder | 3 |
| Sun below the horizon | 2 |
| Sun above 5° elevation | 3 |
| Close agreement between kernels | 3 |
| High baseline error | 2 |
| Low baseline error | 2 |
| Large directional differences | 2 |

Refer to `selection.csv` for observation identifiers and metric values.
The categories describe the selection process; they do not guarantee future results.

## External source data

The fixture was copied from a derived dataset with 599 observations.
The upstream pipeline also uses raw LROC products of approximately 30 GB.
These external products are necessary to rebuild the masks, but not to use the committed masks.

The copy tool is `tools/fixtures/build_tier0_fixture.jl`.
Its `LROC_PIPELINE_DIR` variable must identify the derived pipeline output directory.
That directory must contain `smoke_subset_ids.txt`, `smoke_subset.csv`, `timestamps.csv`, `shadow_20m/`, and `sun_frac_20m/`.

## Fixture changes

1. Rebuild the derived masks if the upstream method changes.
2. Run the upstream observation-selection tool.
3. Run `tools/fixtures/build_tier0_fixture.jl` with the correct source directory.
4. Make new the affected Hyperion maps.
5. Run `tools/fixtures/refresh_correctness_baseline.jl` with the selected backend.
6. Examine the mask and metric changes.
7. Explain the method change in the commit description.

The regression checks aggregate median metrics.
An individual observation can become worse while the aggregate check still passes.
Refer to `docs/src/correctness.md` for the full procedure and output files.
