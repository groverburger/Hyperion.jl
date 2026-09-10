# Ray casting

Hyperion calculates shadows for each requested timestamp.
It does not use a stored azimuth-horizon archive.
A GPU work item calculates one output pixel.
KernelAbstractions also permits CPU execution of the renderer.

## Calculation sequence

1. Convert the DEM sample to meters.
2. Calculate the query position and local frame.
3. Calculate Sun and Earth directions from the SPICE positions.
4. Cast eight rays across the solar disk and one ray toward the Earth.
5. Sample terrain along each ray with bilinear interpolation.
6. Keep the greatest terrain slope along each ray.
7. Calculate solar disk visibility and Earth elevation above the terrain horizon.
8. Encode the two results as UInt8 values.

## Terrain layers

The first layer defines the query grid.
The renderer can continue rays through one or two polar farfield layers.
The [terrain-stack notes](reference/terrain-stack-kernel.md) describe the coordinate transfer.
An output tile changes the calculation window, not the available nearfield terrain.
A site cutoff changes the available terrain.

## Numerical methods

The kernel uses Float32 arithmetic and lookup tables for trigonometric functions.
Explicit `fma` operations control selected rounding steps.
Slope comparisons use a numerator and squared horizontal distance.
This representation avoids a square root and division at each comparison.

Polar ray segments use maximum mipmaps to omit terrain that cannot increase the current horizon bound.
A mipmap cell includes a border for the bilinear sample footprint.
The renderer keeps minimum pyramids, but it no longer uses the previous minimum-cell termination shortcut.
Site segments use the level-zero terrain samples in the layered kernel.

## Output meaning

The Sun value ranges from 0 to 255.
Zero means that the solar disk is fully obscured in the model.
A value of 255 means full disk visibility.
Intermediate values represent partial disk visibility.
The model uses eight ray directions and sixteen integration samples with a fixed solar half-angle of 0.27°.
It does not calculate irradiance or solar-panel power.

The DSN value encodes Earth elevation above the terrain horizon in 0.1° increments, with a maximum value of 250.
It is not a full radio link calculation.

A shadowed ray can terminate when its slope reaches the necessary threshold.
Its diagnostic horizon value can therefore be a lower bound, rather than the final terrain horizon.
The [hilltop investigation](reference/shadowed-hilltop-investigation-2026-07-21.md) gives an example.
