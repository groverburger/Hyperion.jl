# VIPER and Shirley investigation: May 22, 2026

This note records the earlier site-DEM window and tile investigation.
The measurements apply to that run and its hardware.
They are not memory or speed guarantees for other mapsets.

## Input checks

| DEM | Width × height |
|---|---|
| VIPER 8.0 crop | 5,040 × 4,144 pixels |
| Full VIPER 8.0 | 14,336 × 11,008 pixels |
| Nobile 1 m | 4,992 × 4,096 pixels |

The checked VIPER crop used Float32 elevations.
The inspection found no NaN, infinite, or NoData samples.
The two crop dimensions are divisible by 16.
An earlier note incorrectly described these dimensions as incompatible with a 16-pixel workgroup.

## Window reads

The earlier site loader read the full raster before it applied a crop.
Windowed raster reads reduced memory use for cutoff windows.
The mapset dry run reads metadata without construction of the device terrain stack.
A checked 16 × 16 window matched the corresponding full-raster values.

A 128 × 128 output with the full inner DEM and Shirley terrain used approximately 4.4 GB in the recorded run.
Output size alone does not determine memory use.
Loaded terrain and geometry buffers can be much larger than the output.

## Tile correction

An early tile implementation cut the inner terrain at each tile edge.
A 512 × 512 comparison then had 92 different DSN pixels, although its Sun pixels matched.
Rays lost nearby terrain beyond the tile boundary.

The correction kept the full loaded inner DEM for every tile.
The repeated comparison had no Sun or DSN differences.
This behavior is now separate from the explicit site `cutoff` option.

## Recorded validation

The full 5,040 × 4,144 VIPER crop completed with the default 1,024-pixel tile size.
The recorded peak resident memory was approximately 5.75 GiB.
The synthetic Metal terrain tests passed 47 assertions.
The investigation also recorded a full-suite pass at that time.
The full suite was not run again for this documentation update.

Fourteen packed Float32 geometry planes for this crop use approximately 1.17 GB per full buffer.
Host copies, device copies, terrain data, and other arrays add to this amount.
Refer to [terrain stack behavior](terrain-stack-kernel.md) for the current window and cutoff rules.
