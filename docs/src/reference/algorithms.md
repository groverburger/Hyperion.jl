# Renderer algorithms

## Current method

Hyperion casts rays from each output pixel through a terrain stack.
The stack contains one inner layer and up to two outer polar layers.
The renderer uses eight Sun rays and one Earth ray.
A fixed solar half-angle of 0.27° defines the solar disk.

Each ray maintains the maximum terrain slope found so far.
Max-elevation mipmaps let the ray pass over terrain that cannot increase this slope.
The renderer samples the finest level where the bound is insufficient.
A Sun ray can stop when terrain blocks the full solar disk.
The Earth ray determines the Earth elevation above the terrain horizon.

The implementation uses Float32 device arithmetic and selected explicit fused operations.
CPU, Metal, and CUDA use the same kernel source through KernelAbstractions.
Shared source does not by itself prove identical output on every device.
Refer to [numerical precision](cross-vendor-determinism.md) for the known failure modes and recorded checks.

## Projection geometry

For polar stereographic radius `rho` and reference sphere radius `R`, define `u = rho / (2R)`.
For the south-polar projection:

```math
\sin\phi = \frac{u^2-1}{1+u^2}, \qquad
\cos\phi = \frac{2u}{1+u^2}.
```

This form avoids the subtraction of large squared distances.
Site layers use their supported stereographic projection metadata.
The ray geometry includes terrain elevation and lunar curvature.
The [terrain stack note](terrain-stack-kernel.md) explains layer transitions.

## Historical horizon-table method

An earlier design stored a horizon angle for each pixel and azimuth bin.
A 0.25° azimuth interval gives 1,440 bins.
At four bytes per angle, storage is 5,760 bytes per pixel.
An 896 × 512 pixel window therefore uses approximately 2.46 GiB for the horizon table alone.

A stored table can reduce the work per timestamp after the initial terrain calculation.
It also uses a large amount of storage and introduces azimuth interpolation choices.
The current mapset workflow casts rays for each timestamp.
It does not expose a horizon-table archive or a cluster scheduler.

## Earlier numerical corrections

| Problem | Correction |
|---|---|
| Loss of precision in polar coordinates | Use the dimensionless `u` formulation |
| Device differences in transcendental functions | Use host-built lookup tables where necessary |
| Different `log2` results at traversal boundaries | Use explicit bounds for level selection |
| Accidental Float64 device expressions | Use Float32 constants and inputs |
| Different automatic fused operations | Use explicit fused operations at sensitive expressions |
| Visible steps from a coarse azimuth/elevation grid | Calculate geometry per output pixel |
| Missing illumination below zero elevation | Use the −10° twilight limit for the solar disk top |

Later corrections are described in the numerical precision note.
The earlier seven-frame comparison covered 6,422,528 output values across Sun and Earth maps.
That result applies to the recorded inputs and software versions.
It is not a guarantee for every terrain file, backend, or future change.

## Output interpretation

Sun maps encode visible solar-disk fraction as UInt8 values from 0 to 255.
DSN maps encode Earth elevation above terrain in 0.1° increments, with a maximum encoded value of 250.
These maps do not include a solar-panel model or individual ground-station link constraints.
Refer to [raycasting](../raycasting.md) for the current calculation limits.
