# Correctness

Correctness is tested in `test/correctness.jl` against the committed Tier 0
LROC NAC fixture in `test/fixtures/correctness/`.

Run the normal suite:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The correctness sweep is GPU-backed and may be skipped automatically if Metal
or CUDA is unavailable. To skip it explicitly:

```bash
HYP_SKIP_CORRECTNESS=1 julia --project -e 'using Pkg; Pkg.test()'
```

## Fixture Maintenance

These tools are not normal user entry points. They exist so correctness data
can be regenerated deliberately and with input hashes recorded.

Refresh the Tier 0 fixture from an external Tier 1 LROC pipeline output:

```bash
LROC_PIPELINE_DIR=/path/to/lroc-nac-maps/derived \
  julia --project tools/fixtures/build_tier0_fixture.jl
```

The tool writes `test/fixtures/correctness/input_shas.csv` with hashes for
the copied source files.

Refresh the pinned baseline after an intentional behavior change:

```bash
julia --project tools/fixtures/refresh_correctness_baseline.jl
git diff test/fixtures/correctness/baseline_tier0.csv
```

The refresh tool writes `baseline_tier0_input_shas.csv` beside the baseline.
Commit baseline changes only with a short explanation of the algorithm or data
change that justified them.
