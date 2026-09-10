# Shadowed hilltop investigation: July 21, 2026

The checked hilltop output was consistent with the terrain and Sun geometry.
The investigation found no renderer defect in this case.
At the reported time, the Sun was approximately 3° below the local horizontal plane.
A slope can receive light from that direction while nearby terrain blocks the light at the hilltop.

## Inputs

The investigation used the medium VIPER site DEM and the Barker 2023 outer DEM.
The corresponding specification is `data/inputs/mapsets/v8_medium_barker_sep.toml`.
The probe command below overrides its current time range.

| Quantity | Value |
|---|---|
| Site file | `viper_sfs_dem_8_0_v71_2027_medium_extent.tif` |
| Outer file | `LDEM_80S_20MPP_ADJ.TIF` |
| Site dimensions | 4,992 columns × 5,248 rows, 1 m pixels |
| Site elevations | Float32 meters above the 1,737.4 km sphere |
| Projection origin | −85.42088° latitude, 31.6218° longitude |
| Hilltop pixel | Column 2285, row 4663; indices start at zero |
| Hilltop position | −85.47574° latitude, 31.45666° longitude |
| Hilltop elevation | 6,309.395 m |
| Flank pixel | Column 2285, row 4729; elevation approximately 6,306.4 m |
| Reported time | 2027-10-27T06:00:00 UTC |

The crest is nearly flat.
Maximum elevations within 5, 25, and 100 pixels were 6,309.43 m, 6,310.28 m, and 6,314.10 m, respectively.
The probe pixel is not the highest pixel in each of these neighborhoods.

## Sun geometry

| Quantity | Recorded value |
|---|---|
| Sun azimuth | 268.16°, counterclockwise from east |
| Sun center elevation | −3.008° |
| Solar angular diameter | 0.537° |
| Solar disk top | Approximately −2.74° |
| Earth elevation | −1.72° |
| Kernel Sun elevation | −3.0076° |

The independent Float64 SPICE calculation agreed with the kernel geometry to four decimal places for Sun elevation.
The Earth was also below the local horizontal plane.

A 12-hour time sweep gave the following approximate events.
These events use zero geometric Sun elevation, without a terrain-horizon correction.

| Event | Approximate date |
|---|---|
| Sunset | September 21, 2027, 18:00 UTC |
| Minimum Sun elevation, −3.26° | September 27–28 |
| Sunrise | October 3, 18:00 UTC |
| Maximum Sun elevation, +6.00° | October 12–13 |
| Sunset | October 21, 18:00 UTC |
| Minimum Sun elevation, −3.01° | October 27, 06:00–12:00 UTC |
| Sunrise | November 2, 06:00 UTC |

## Terrain explanation

At the hilltop, nearby terrain approximately 13 m away sets a horizon near −0.30°.
This terrain horizon is above the solar disk top of −2.74°.
It therefore blocks the Sun at ground level.

Parts of the flank have terrain horizons from approximately −3.1° to −5.6°.
The Sun is partly or fully above these horizons.
Small changes in slope produce alternate illuminated and shadowed bands.
The absolute elevation of the hilltop alone cannot determine its illumination.

The following strip uses column 2285 at the reported time.
Horizon values came from an independent Float64 calculation.
PNG values are encoded solar fractions from 0 to 255.

| row | elev (m) | horizon @0 m (blocker dist) | @0.5 m | @1.0 m | PNG 0 m | PNG 0.5 m | PNG 1 m |
|---|---|---|---|---|---|---|---|
| 4660 | 6309.4 | +0.25° (3 m) | −1.61° | −2.65° | 0 | 0 | 0 |
| 4664 | 6309.4 | −0.22° (12 m) | −1.88° | −3.08° | 0 | 0 | 169 |
| 4676 | 6309.3 | −1.14° (11 m) | −3.19° | −3.53° | 0 | 230 | 255 |
| 4692 | 6308.8 | −3.06° (68 m) | −3.47° | −3.89° | 157 | 255 | 255 |
| 4704 | 6307.9 | −2.80° (55 m) | −3.31° | −3.81° | 16 | 255 | 255 |
| 4728 | 6306.4 | −2.25° (29 m) | −3.18° | −4.04° | 0 | 222 | 255 |
| 4760 | 6305.1 | −3.51° (1 m) | −5.56° | −5.77° | 255 | 255 | 255 |

The independent geometry reproduced the observed illuminated and shadowed bands at the three checked observer heights.
The comparison includes partially illuminated pixels near the edge of the solar disk.

## Observer height

| Observer height | Hilltop, row 4663 | Flank, row 4729 |
|---|---|---|
| 0 m | 0 | 0 |
| 0.25 m | 0 | 0 |
| 0.5 m | 0 | 215 |
| 1 m | 127 | 255 |

A height change has a large angular effect when the blocking terrain is close.
For a fixed blocker 13 m away, `atan(1/13)` is approximately 4.4°.
The controlling blocker can change as the observer height changes.
The full terrain calculation is therefore necessary for the final solar fraction.

## Verification

Three checks supported the result:

1. The production kernel rendered a 1 × 1 window with the full loaded site terrain.
2. An independent Float64 terrain calculation reproduced the illumination pattern.
3. The local latitude and solar declination predicted a minimum Sun elevation near −3°.

The production probe used the CPU backend.
The result is evidence for this case, not a new guarantee of agreement across all backends.

A positive control used the same pixel at 2027-10-12T00:00:00 UTC.
The Sun elevation was +5.96°, and the Sun map value was 255.
The eight ray horizons ranged from +3.36° to +3.42°.
The independent horizon was approximately +3.40°.
A site ridge approximately 2.23 km away supplied the controlling terrain.
The DSN result was 4.2° above the terrain horizon.

| Possible cause | Recorded check |
|---|---|
| Incorrect outer-layer obstruction | The controlling terrain was on the site layer in the two probes |
| Large elevation mismatch at the layer boundary | The checked outer elevation differed by approximately −0.51 m |
| Incorrect Sun coordinate conversion | Kernel and independent SPICE values agreed |
| Mipmap traversal error | Mipmap and finest-level traversal agreed for the checked rays |
| Incorrect observer-height effect | The height variants agreed with independent terrain geometry |

## Debug output limits

A shadowed Sun ray stops after it finds sufficient terrain to block the solar disk.
Its reported debug horizon can therefore be a lower bound, rather than the maximum terrain horizon.
At the hilltop, the first sample gave approximately −1.39°, already above the disk top.
The ray stopped before it reached the true horizon near −0.30°.
This early stop did not change the shadow result.

The −10° twilight limit did not apply to this case.
The 1 × 1 render window kept the full loaded site extent.
Its output agreed with the corresponding archived full-frame pixel.

## Reproduction

Run the shadowed case from the repository root:

```bash
julia --project tools/debug/probe_pixel.jl \
  --spec=data/inputs/mapsets/v8_medium_barker_sep.toml \
  --time=2027-10-27T06:00:00 --col=2285 --row=4663
```

Run the illuminated control:

```bash
julia --project tools/debug/probe_pixel.jl \
  --spec=data/inputs/mapsets/v8_medium_barker_sep.toml \
  --time=2027-10-12T00:00:00 --col=2285 --row=4663
```

The tool reports kernel results, terrain blockers, and an independent Float64 site-horizon calculation.
Optional arguments are `--observer=<meters>`, `--patch=<radius>`, and `--no-farfield`.
The tool does not compare input SHA-256 values and has no `--help` option.
The external DEMs are necessary.
The archived mapsets are not stored in Git.

The seven-slide presentation `docs/hilltop_shadow_explained.pptx` illustrates the Sun cycle, terrain profile, and observer-height results.
