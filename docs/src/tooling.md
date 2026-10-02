# Workflow commands

Run commands from the repository root.
The four scripts below have `--help` options.

| Entry script | Function |
|---|---|
| `scripts/generate_light_curve.jl` | Export terrain-shadowed solar visibility for one location |
| `scripts/generate_mapset.jl` | Make a mapset from a TOML specification |
| `scripts/generate_azel_csv.jl` | Export Sun and Earth geometry for one location |
| `scripts/generate_viper8_1m_radius_shirley_mapset.jl` | Make a VIPER window mapset with Shirley terrain |

The [mapset guide](mapsets.md) and [CSV guide](azel.md) give examples.
The previous `scripts/fetch_test_data.jl` entry script is no longer available.
The mapset command no longer accepts preset options.

## Tests and audits

| Entry point | Function |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | Run the standard suite |
| `HYP_BACKEND=cpu julia --project test/terrain_stack.jl` | Run small synthetic terrain tests |
| `julia --project test/mapset_workers.jl` | Check mapset processes and output on small synthetic terrain |
| `tools/bitexact/bitexact_test.jl` | Produce hashes and raw buffers for one backend |
| `tools/bitexact/diff_bitexact_shas.jl` | Compare stored backend hashes |
| `tools/bitexact/diff_bitexact_pixels.jl` | Compare stored image buffers |
| `tools/bitexact/cross_vendor_test.ps1` | Run Windows CPU/CUDA checks |
| `tools/bitexact/windows_cuda_unified_stack.ps1` | Run Windows terrain and CUDA checks |

Julia tools use the command form `julia --project <script>`.
PowerShell tools use the command form `.\tools\bitexact\<script>.ps1`.
The [test guide](testing.md) identifies necessary data and automatic test exclusions.

## Reference maintenance

These tools can write large outputs or replace reference data.

| Tool in `tools/fixtures/` | Function |
|---|---|
| `generate_tier0_baseline_maps.jl` | Make 20 m or full 1 m maps at NAC times |
| `build_tier0_fixture.jl` | Copy the 25-observation fixture from external LROC pipeline output |
| `refresh_correctness_baseline.jl` | Replace the Tier 0 metric baseline |
| `regenerate_bitexact_pins.jl` | Make replacement PNG fixtures and hash tables |

The baseline-map tool accepts `--only=20m`, `--only=1m`, or `--only=both`.
`--limit=<count>` limits the selected observations.
`--out=<path>` selects the output root.
The [correctness guide](correctness.md) gives fixture procedures.

## Pixel diagnosis

```bash
julia --project tools/debug/probe_pixel.jl \
  --spec=data/inputs/mapsets/v8_medium_barker_sep.toml \
  --time=2027-10-27T06:00:00 --col=2285 --row=4663
```

The tool checks one pixel in a site DEM.
It uses the CPU and reports ray geometry, terrain blockers, and an independent Float64 comparison.
Pixel indices start at zero.
Optional arguments are `--observer=<meters>`, `--patch=<radius>`, and `--no-farfield`.
The tool does not compare layer hashes with the specified values.
Its file header contains the usage instructions; it has no `--help` option.

## Documentation and benchmark

`julia --project=docs docs/make.jl` builds the HTML documentation after installation of the docs dependencies.

`tools/bench/bench_workgroup.jl` is not ready for normal use.
Its warm-up call omits necessary backend arguments.
It also selects Metal in its source code.
A developer must repair that call before a benchmark run.
