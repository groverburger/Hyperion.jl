# Exact output comparison

For a bit-exact comparison to pass, the output bits must be identical.
The standard regression compares hashes and decoded PNG pixels with stored reference values.
It uses the selected backend; it does not run all backends in one command.
The [test guide](testing.md) gives data requirements and backend selection.

## Run a backend audit

Run the audit on each available backend:

```bash
HYP_BACKEND=metal julia --project tools/bitexact/bitexact_test.jl
HYP_BACKEND=cuda julia --project tools/bitexact/bitexact_test.jl
HYP_BACKEND=cpu julia --project tools/bitexact/bitexact_test.jl
```

The audit uses the Shirley DEM and 20 timestamps.
Before the frame calculations, a small kernel checks selected arithmetic results.
The audit then writes hashes, raw buffers, and PNG files under `data/outputs/bitexact/<backend>/`.
It also checks that PNG decoding keeps the image pixels.
The standard regression and this audit have different entry scripts.

## Compare backend results

1. Put results from at least two backends under `data/outputs/bitexact/`.
2. Compare the hashes:

   ```bash
   julia --project tools/bitexact/diff_bitexact_shas.jl
   ```

3. Compare the image buffers:

   ```bash
   julia --project tools/bitexact/diff_bitexact_pixels.jl
   ```

Raw PNG file bytes can differ between encoders while the decoded pixels remain identical.
The backend comparison uses output buffers and decoded image content.

## Windows tools

```powershell
.\tools\bitexact\cross_vendor_test.ps1
.\tools\bitexact\windows_cuda_unified_stack.ps1
```

These scripts run Windows CPU/CUDA checks and record results.
The second script also has options for full package tests and Tier 0 map generation.

## Reference changes

`tools/fixtures/regenerate_bitexact_pins.jl` makes new PNG fixtures and hash tables.
This tool changes reference data.
A developer must examine the changed images and hashes before a commit.

The [kernel arithmetic notes](reference/cross-vendor-determinism.md) explain previous precision fixes.
The [May 2026 audit](reference/cross-vendor-verification-2026-05-01.md) records one historical result.
Neither record proves compatibility with every GPU or later package version.
