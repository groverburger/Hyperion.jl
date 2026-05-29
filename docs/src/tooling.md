# Tooling

Supported command-line entry points are intentionally small.

## Scripts

- `scripts/generate_mapset.jl` - main mapset generator.
- `scripts/generate_azel_csv.jl` - standalone azimuth/elevation CSV export.
- `scripts/fetch_test_data.jl` - validate local baseline inputs and derive
  small local fixtures.

## Tools

- `tools/fixtures/build_tier0_fixture.jl`
- `tools/fixtures/refresh_correctness_baseline.jl`
- `tools/fixtures/regenerate_bitexact_pins.jl`
- `tools/fixtures/generate_tier0_baseline_maps.jl`
- `tools/bitexact/bitexact_test.jl`
- `tools/bitexact/diff_bitexact_shas.jl`
- `tools/bitexact/diff_bitexact_pixels.jl`
- `tools/bitexact/cross_vendor_test.ps1`
- `tools/bitexact/windows_cuda_unified_stack.ps1`
- `tools/bench/bench_workgroup.jl`

Everything else should justify its existence in docs or live outside git.
