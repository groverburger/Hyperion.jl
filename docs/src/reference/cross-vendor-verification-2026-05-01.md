# Backend verification: May 1, 2026

This historical audit used commit `300f557` before the current terrain-stack interface.
It compared Apple CPU, Metal, Windows CPU, and CUDA results for the tested configurations.
It does not establish agreement for all current mapsets or devices.

## Results

| Check | Recorded result |
|---|---|
| Stored product hashes | 280 of 280 matched per audited output set |
| Hash coverage | 20 timestamps × 14 products |
| Raw-buffer comparisons | 228 matched: 19 timestamps × four channels × three backend pairs |
| Detailed site case | 256 × 256 pixel, 1 m site-only test on CPU |

The site test did not include CUDA or the current full 1 m terrain stack.
The raw-buffer comparison used 19 timestamps and must not be described as a 20-timestamp check.

## Related changes

Commits `bfda7f6`, `533f9ed`, and `facfe85` contain numerical and backend corrections associated with this work.
Commits `b048496` and `3a57940` contain pool, step, and minimum-bound corrections.
Commit `122e31d` updated stored bit-exact references.
Commit `9ecfb43` updated the correctness baseline.

## Current audit commands

These commands invoke the current tools; they do not recreate the previous commit automatically.

```bash
HYP_BACKEND=cpu julia --project tools/bitexact/bitexact_test.jl
HYP_BACKEND=metal julia --project tools/bitexact/bitexact_test.jl
```

For Windows PowerShell:

```powershell
.\tools\bitexact\cross_vendor_test.ps1
```

The external terrain files in the [data guide](../data.md) are necessary for the audit.
Use the [bit-exact guide](../bitexact.md) to compare audit outputs.
The [September comparison](working-tree-validation-2026-09-10.md) records the recent remote-versus-working checks.
