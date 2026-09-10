# Numerical precision and backend agreement

Hyperion uses shared Float32 kernel code for CPU, Metal, and CUDA.
Identical source can still produce different values because compilers and devices can evaluate expressions differently.
Agreement must be checked for the actual inputs, dependencies, and hardware.
Agreement also does not prove physical accuracy.

This note records the numerical design and earlier corrections.
The [May audit](cross-vendor-verification-2026-05-01.md) and [September comparison](working-tree-validation-2026-09-10.md) have separate test scopes.

## Tested configuration

The earlier multi-backend work used Julia 1.11.5, KernelAbstractions 0.9.41, Metal 1.9.3, and CUDA 5.9.0.
One recorded comparison matched 630 hashes: 15 timestamps, 14 products, and three backend pairs.
A separate raw-buffer check matched 180 comparisons: 15 timestamps, four channels, and three backend pairs.
These are historical results, not checks automatically performed by each mapset run.

## Device arithmetic guard

The audit script `tools/bitexact/bitexact_test.jl` checks selected host and device operations before it renders images.
The standard test suite does not run this guard.

| Expression | Expected Float32 bits |
|---|---|
| `fma(a, a, -1f0)`, where `a = 1f0 + 2f0^-12` | `0x3a000400` |
| `1f0 / 3f0` | `0x3eaaaaab` |
| `sqrt(2f0)` | `0x3fb504f3` |

These checks detect specific arithmetic differences.
They do not cover every compiler transformation or floating-point input.

## Explicit fused operations

A fused multiply-add calculates `a*b+c` with one final rounding step.
Separate multiplication and addition can give a different result.
The renderer uses explicit `fma` calls where this distinction affects ray geometry or lookup-table interpolation.

For example, a local displacement uses this form:

```julia
dx = fma(common, n_km, -qx)
dy = fma(common, e_km, -qy)
```

A lookup-table interpolation uses `fma(frac, v1 - v0, v0)`.
This rule applies to the angle and sine/cosine tables.
Explicit fused operations make the necessary evaluation clear to the compiler.

## Stable slope comparisons

The renderer delays conversion from slope to angle until an output angle is necessary.
For slopes with numerators `an`, `bn` and positive squared horizontal distances `ad`, `bd`, it compares squared products.
The products are `an^2 * bd` and `bn^2 * ad`.
The comparison reverses when the two numerators are negative.
Different numerator signs are handled separately.

This method avoids a square root and division for each candidate slope.
The signed comparison is necessary; squares alone cannot distinguish positive and negative slopes.
Angle conversion uses the shared lookup table at the end.

Other expressions use squared polar radius directly.
Selected constant reciprocals are computed once, including `INV_2R_M_F32`, `INV_R_KM_F32`, `INV_4R_KM2_F32`, and `INV_MAX_PHOTONS`.
For variable denominators, the applicable division is still necessary.

## Horizontal distance and observer height

For an orthonormal local basis, horizontal distance satisfies:

```math
h^2 = d^2 - z_{\mathrm{geom}}^2.
```

The geometric vertical component defines this horizontal distance.
Observer height changes the vertical slope component afterward.
This order prevents observer height from changing the horizontal distance.
Subtraction near a vertical ray can still lose relative precision.
Examine this condition when you change the formula.

## Local vertical displacement correction

An earlier formula subtracted Moon-scale Cartesian coordinates to get a small local height difference.
Float32 spacing at the lunar radius is approximately a decimeter.
This subtraction could therefore lose important local terrain detail.
The resulting ring artifacts affected 20 m output as well as detailed site output.

For south-polar geometry, write the vertical coordinate as:

```math
z = -R_t + \frac{2R_tu^2}{1+u^2},
```

where `R_t` includes terrain elevation.
Calculate the small positive term directly for the sample and the query.
Then calculate the vertical difference from the terrain-height difference and the two positive terms:

```math
\Delta z = (e_q-e_s)\,0.001 + (z_{s,+}-z_{q,+}).
```

Elevations `e_q` and `e_s` are in meters; the result is in kilometers.
The implementation uses explicit fused operations:

```julia
dz = fma(q_elev_m - telev_m, 0.001f0,
         fma(scale, two_u2, -qz_pos))
```

Do not recover `qz_pos` by addition to a rounded Moon-scale coordinate.
Do not calculate the terrain-height difference by subtraction of two rounded total radii.
The two operations would cause the loss of precision again.

### Recorded precision checks

A synthetic constant-elevation DEM contained 4,096 × 4,992 pixels at 6,000 m elevation.
The earlier output had 138 distinct Sun values.
The corrected output had one Sun value.
The Earth debug result changed from a spurious +5.22° to approximately zero.

A 20 m case at row 8963, column 19206 used a 2027-01-22 timestamp.
Its Sun output changed from 0 to 255.
The independent calculation gave these local values:

| Calculation | Vertical displacement | Slope angle | Displacement error |
|---|---|---|---|
| Float64 reference | −63.60 µm | 4.4701° | Reference |
| Corrected Float32 | −62.95 µm | 4.4683° | 0.65 µm |
| Earlier Float32 | −366.21 µm | 5.3308° | 302.6 µm |

In one earlier 20 m image comparison, 2.15% of Sun pixels and 1.53% of DSN pixels changed.
Mean absolute encoded differences were 0.52 and 1.31, respectively.
Maximum differences were 255 and 205.
These measurements describe that comparison only.

## Other recorded corrections

| Area | Correction |
|---|---|
| Connected arithmetic expressions | Make necessary fused operations explicit |
| Radius and distance calculations | Avoid unnecessary square roots and unstable subtraction |
| Signed slopes | Compare signed squared quantities before angle conversion |
| Constant division | Use shared precomputed reciprocals |
| Metal resource limits | Reduce kernel resource use in the affected configuration |
| Lookup tables | Use explicit fused interpolation |
| Local vertical geometry | Preserve terrain differences before Moon-scale coordinate operations |

The historical Metal failure reported 33 resources against a limit of 31.
That observation describes the affected kernel configuration, not a universal device limit.

## Regression procedure

1. Run the small arithmetic and terrain checks.
2. Run the affected stored-image cases with the necessary terrain files.
3. Compare raw output buffers across the available backends.
4. Do an independent check of physical geometry when a numerical formula changes.
5. Replace reference images only after you explain the changed output.

The standard bit-exact test covers 20 timestamps for each supported reference configuration.
Backend selection and external data determine which cases actually run.
Refer to [bit-exact tests](../bitexact.md) for commands and limitations.
