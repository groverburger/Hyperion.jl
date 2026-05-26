# Bit-exactness

`test/bitexact.jl` is the default regression check. The maintenance harnesses
under `tools/bitexact/` are for cross-vendor audits and fixture refreshes.

Run the forensic harness on a backend:

```bash
HYP_BACKEND=metal julia --project tools/bitexact/bitexact_test.jl
HYP_BACKEND=cuda  julia --project tools/bitexact/bitexact_test.jl
HYP_BACKEND=cpu   julia --project tools/bitexact/bitexact_test.jl
```

Compare backend outputs:

```bash
julia --project tools/bitexact/diff_bitexact_shas.jl
julia --project tools/bitexact/diff_bitexact_pixels.jl
```

On Windows CUDA machines, use:

```powershell
.\tools\bitexact\cross_vendor_test.ps1
```

`tools/bitexact/windows_cuda_unified_stack.ps1` is the broader Windows
validation driver: package tests, CUDA bit-exact harness, and optional Tier 0
baseline product generation.
