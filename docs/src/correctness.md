# Correctness

The correctness test compares simulated sunlight with 25 LROC Narrow Angle Camera (NAC) observations.
Git stores this Tier 0 fixture in `test/fixtures/correctness/`.
The test uses the Shirley DEM and a Metal or CUDA backend.
The full external LROC data collection is not necessary for this test.

## Run the test

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The standard suite includes the observation comparison.
The test does not run if the Shirley DEM or a GPU backend is unavailable.
`HYP_SKIP_CORRECTNESS=1` disables this comparison.
Other large regressions remain enabled.

## Pass conditions

The test calculates an image for each distinct observation time.
Observations with the same timestamp share that calculation.
It then compares the images with the stored masks.

Each aggregate median metric must be at least as good as its baseline value.
A worse result for an individual observation does not fail the test if the aggregate medians pass.
The test still records those individual differences.

The result directory is `data/outputs/correctness/<run-time>/`.

| File | Content |
|---|---|
| `<product-id>.tif` | Simulated image |
| `input_shas.csv` | Input DEM hash |
| `current.csv` | Metrics for each observation |
| `delta.csv` | Differences from baseline metrics |
| `summary.csv` | Aggregate differences |

A matching backend hash proves repeatability for that output.
It does not prove agreement with lunar observations.

## Replace the observation fixture

1. Make the full external LROC pipeline output.
2. Set `LROC_PIPELINE_DIR` to that output directory.
3. Run the fixture tool:

   ```bash
   LROC_PIPELINE_DIR=/path/to/lroc-nac-maps/derived \
     julia --project tools/fixtures/build_tier0_fixture.jl
   ```

The tool copies the selected masks and timestamps.
It also writes `input_shas.csv` in the fixture directory.
The fixture README at `test/fixtures/correctness/README.md` describes the necessary source files.

## Replace baseline metrics

Use this procedure after an intentional algorithm or input change.

1. Make new metrics:

   ```bash
   HYP_BACKEND=metal julia --project tools/fixtures/refresh_correctness_baseline.jl
   ```

2. Examine the baseline changes:

   ```bash
   git diff test/fixtures/correctness/baseline_tier0.csv
   ```

3. Record the reason for the change in the commit message.

The tool replaces `baseline_tier0.csv` and writes `baseline_tier0_input_shas.csv`.
Its default backend is CPU, which can take a long time.
Explain each baseline change.
A baseline change is not a general repair for a failed test.
