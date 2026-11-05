# Permanently shadowed regions

The PSR workflow tests whether any modeled direct sunlight reaches each pixel
at any sample in a study interval. A strictly positive floating-point visible
solar-disk fraction retires the pixel permanently. Classification happens before
conversion to the 0–255 image scale; even exposures that round to image value
zero retire a pixel. This is a sampled-time geometric definition, not a thermal
criterion or a guarantee between samples.

## Run

From the repository root:

```sh
julia +1.12.7 --project=. --threads=auto scripts/generate_psr.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml --backend=metal
```

Use `--backend=cuda` on NVIDIA or `--backend=cpu` for small checks. An explicitly
requested GPU must be available. The default interval is 1970-01-01 through
2044-01-01 inclusive, with a six-hour cadence and observer height zero. The
mapset specification supplies the terrain layers and output window. Its dates,
timestamp list, and rover observer height do not override the PSR defaults.

Override study settings explicitly:

```sh
julia --project=. scripts/generate_psr.jl --spec=<mapset.toml> \
  --start=1970-01-01T00:00:00 --stop=2044-01-01T00:00:00 \
  --step-hours=6 --observer-height-m=0 --out=<output-directory>
```

The 74-year interval reproduces mapbuilder's study dates. It is not an exact
four-cycle astronomical period. SPICE coverage is checked for all sample times
before ray tracing starts. All required kernels must be present.

## Retirement and execution order

Both CPU geometry preparation and GPU ray tracing receive only the compacted
surviving pixel IDs. A pixel removed after one sample is never computed again.
The renderer reuses device terrain buffers and processes survivors in batches
(default 65,536). Earth-visibility rays and diagnostic frames are disabled.

Samples are visited in bit-reversed index order, spreading the earliest tests
across the interval rather than spending the beginning of a run on consecutive
winter samples. Every regular-grid timestamp is still visited exactly once
unless all pixels have already been eliminated. The output is independent of
sample order. Persistently shadowed pixels must still be tested for the whole
interval; elimination rates do not imply a constant runtime speedup.

## Outputs and resume

The default destination is `data/outputs/<mapset name>/psr/`:

- `psr.tif`: final georeferenced mask; **1 = sampled PSR**, **0 = ever sunlit**,
  **255 = NoData**. Written only when the classification is complete.
- `checkpoint.bin`: atomic checkpoint of surviving candidates, the completed
  sample count, and cumulative pixel evaluations. Candidates are not yet PSRs.
- `metadata.toml`: definition, dates, cadence, observer height, window, backend,
  completion state, and hashes of input terrain, SPICE kernels, and relevant code.
- `progress.csv`: sample times, pixels evaluated, survivors, and elapsed seconds
  for the current process. A resumed run can repeat rows after its last checkpoint;
  the checkpoint and metadata are authoritative.

While running, a progress line is printed every 10 seconds (`--progress-seconds=N`):

```text
PSR 40/108113 samples (0.04%) | 28457 of 458752 pixels unlit (6.2%) | 18s elapsed | ~1h 28m left
```

The time left uses the rate since the previous line. Early estimates are much
too high, because samples get faster as pixels are retired.

Rerun the same command to resume. Changed settings, terrain, kernels, Julia
version, backend, or relevant code invalidate the checkpoint. Use a new output
directory for a different study. Checkpoints are saved approximately every
60 seconds and at a clean stop; a single sample may take longer. `--max-samples=N`
limits additional samples for a test run. `--no-resume` refuses an existing
checkpoint. No minimum-area or connectivity filter is applied.

For the Julia API, construct a `MapsetSpec` with the desired study interval and
call `generate_psr(spec; backend=:metal)`. Unlike the CLI, the API uses the
specification's dates, cadence, and observer height directly.

## Tests

```sh
julia +1.12.7 --project=. test/psr.jl
HYP_PSR_TEST_GPU=1 julia +1.12.7 --project=. test/psr.jl
```

Tests compare sparse classifications against full-frame ray diagnostics for
single-source and layered terrain, including sub-byte sunlight, sparse index
mapping, permanent retirement, time ordering, and checkpoint integrity.

The optional real-data regression compares every pixel of the 20 m Nobile
window with the full-frame renderer's unquantized ray diagnostics:

```sh
julia +1.12.7 --project=. --threads=8 test/psr_nobile_validation.jl
```

On 2026-10-21, the Metal comparison at 2026-01-27T00:00:00 matched all
458,752 pixels, including 312 positive exposures below the byte-image cutoff.
The 256-sample trial over the 1970–2044 grid retained 23,742 candidates (5.18%)
and performed 8,234,460 pixel evaluations instead of 117,440,512 without
retirement. These survivors are candidates from a partial trial, not a final
74-year PSR result. Validation logs are in
`data/outputs/nobile_20m_shirley/psr-validation/`.

## Small-region filtering

The CLI preserves raw `psr.tif` and, after completion, runs a separate
postprocessing step to write `psr-filtered.tif`. It retains regions with at least
five PSR pixels connected by shared edges (four-connectivity). Diagonal contact
does not connect regions. NoData, projection, and pixel transform are preserved.
This is a size filter, not a change to the physical shadow classification.

Filter existing outputs without recomputing illumination:

```sh
julia +1.12.7 --project=. scripts/filter_psr.jl path/to/psr.tif [another/psr.tif]
```

The Julia API is `filter_psr(input; output=..., min_pixels=5)`. The lower-level
`generate_psr` API continues to produce the raw classification; call `filter_psr`
after it to obtain the separate filtered product.

## Mapbuilder-compatible sunlight cutoff

Pass `--mapbuilder-cutoff` to `scripts/generate_psr.jl` (API:
`mapbuilder_cutoff=true`) to classify each sample using
`trunc(Int, 255f0 * sun_frac) > 0`. Exposures below approximately 1/255
(0.392% of the solar disk) truncate to zero. The default remains strict
`sun_frac > 0`. This matches mapbuilder's quantization rule, not all of its
terrain or illumination modeling. The mode is recorded in metadata and the
checkpoint fingerprint; different modes cannot share a checkpoint.
Use a separate `--out` directory for a comparison run.

New raw and filtered TIFFs use red (255,0,0) for PSR and black (0,0,0)
for non-PSR; classification values remain 1 and 0, with 255 reserved for NoData.
