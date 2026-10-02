# Tests

The repository contains the standard test code and reference outputs.
The large terrain inputs are external files.
A successful test command does not mean that every test ran.
The test output identifies unavailable data and backends.

## Small terrain tests

Run the synthetic terrain tests on the CPU:

```bash
HYP_BACKEND=cpu julia --project test/terrain_stack.jl
```

These tests make their own terrain arrays.
The checks include layer transfers, terrain shadows, projection geometry, and three-layer operation.
External DEMs and SPICE kernels are not necessary for these tests.

Select Metal or CUDA with `HYP_BACKEND=metal` or `HYP_BACKEND=cuda`.

## Small mapset worker tests

```bash
julia --project test/mapset_workers.jl
```

These tests compare one-process and two-process output on synthetic terrain.
They also check worker failures, missing-image continuation, explicit timestamps, and shared metadata.
They use the CPU and the stored SPICE kernels.
The render checks require `gdaldem` on `PATH`; they do not run if it is unavailable.
External terrain files are not necessary.

On a node with at least two visible NVIDIA GPUs, also test the GPU launcher:

```bash
HYP_MAPSET_TEST_GPUS=2 julia --project test/mapset_workers.jl
```

The additional check compares the GPU images with the CPU images.

## Standard suite

Run the suite from the repository root:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

| Test | Input requirements |
|---|---|
| Lookup tables and projection geometry | No external data |
| Synthetic terrain stacks | No external data |
| Mapset workers | Stored SPICE kernels and `gdaldem`; optional NVIDIA GPUs |
| Mipmap pyramid | SHA-verified Shirley DEM |
| 20 m output regression; 20 timestamps | Shirley DEM and a render backend |
| Full 1 m output regression; 20 timestamps | Shirley DEM, `nobile_1m.tif`, and a render backend |
| Tier 0 observation comparison | Shirley DEM and Metal or CUDA |

The full 1 m frames contain 4992 × 4096 pixels.
The full suite can take a long time, including CPU preparation before GPU work.

The backend loader first tries a GPU package.
Large frame regressions do not run if no backend is available.
`HYP_BACKEND=cpu` or `HYP_ALLOW_CPU_TEST_BACKEND=1` permits CPU frame regressions.
The Tier 0 observation test does not permit CPU operation.
The synthetic terrain tests can use the CPU without those overrides.

The targeted checks passed on Julia 1.11.5 and 1.12.7.
The backend loader handles the global-binding rules in Julia 1.12.

## Disable the observation comparison

```bash
HYP_SKIP_CORRECTNESS=1 julia --project -e 'using Pkg; Pkg.test()'
```

This option does not disable the large frame regressions.
The [output comparison guide](bitexact.md) and [correctness guide](correctness.md) give more details.

## Stored reference data

Git stores these reference files:

- Forty 20 m PNGs and their expected hashes.
- Forty full 1 m PNGs and their expected hashes.
- Twenty-five shadow masks and twenty-five solar-fraction masks from LROC NAC observations.
- Observation timestamps, selection records, and baseline metrics.
- SPICE kernels and DSN horizon-mask files.

Git ignores terrain files in `data/inputs/` and run results in `data/outputs/`.
The tests do not download terrain data.
A missing or incorrect Shirley hash disables Shirley-dependent tests.
An existing 1 m file with an incorrect hash causes an error when its regression runs.

## Separate checks

The scripts in `tools/bitexact/` produce and compare backend audit results.
The scripts in `tools/fixtures/` make or replace reference data.
They are separate from the standard suite.

The [2026-09-10 comparison](reference/working-tree-validation-2026-09-10.md) records targeted checks of the current cleanup.
Its 30-render comparison was a separate local script, not a new standard test.
