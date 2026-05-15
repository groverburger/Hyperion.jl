# Terrain-Stack Kernel Direction

This note records the intended shape of the unified terrain-stack renderer.
It should be read alongside `docs/algorithms.md` and
`docs/cross-vendor-determinism.md`; all kernel changes must preserve the
same bit-exactness rules: explicit `fma` for reconstructible multiply-add
or multiply-subtract dataflow, no `log2` for mipmap level selection, no
new hot-loop transcendental calls, and test coverage against the pinned
20 m byte fixtures.

## Current State

The one-source 20 m renderer and the 1 m site -> 20 m farfield renderer now
share the hot ray-marching core:

- level-0 bilinear terrain sampling and slope accumulation,
- hierarchical mipmap skipping for polar-stereographic farfield layers,
- sun-disk integration,
- UInt8 output encoding.

The two cases still have different outer GPU kernels because their per-pixel
setup differs:

- a single polar-stereographic layer starts and ends in one grid;
- a site layer starts in a local stereographic grid, exits that raster, then
  hands the ray to a polar-stereographic farfield grid.

## Handoff Rules

Layer handoff has two separate pieces of geometry:

- **Ray geometry uses the true elevated query position.** This is the
  position used for slope tests against terrain samples.
- **Map reprojection uses the datum position.** Row/column coordinates in
  the next layer are planimetric and must not shift with terrain elevation.
  The datum point is the surface point at lunar radius along the query
  normal. Projecting the elevated terrain point caused multi-cell farfield
  offsets at Nobile elevations.

When a ray exits a layer, the next layer starts at the same physical
distance along the ray:

```
next_start_pixels = max(1, exit_distance_pixels * from_pixel_size_m / to_pixel_size_m)
```

Each segment is bounded by both a dynamic terrain-height limit and the
distance to leave that layer's raster. There is no fixed hard cap in the
stacked path.

## Projection Model

The current implementation supports two device-side projection formulas:

- **Local stereographic site grid.** Used by `SiteDEM`; this is a
  stereographic grid centered on `(lat0, lon0)` and represented in a local
  frame before entering the kernel.
- **South polar stereographic grid.** Used by global or cropped LDEM
  farfield layers.

Supporting arbitrary innermost DEM projections without resampling requires a
different representation. The GPU cannot call GDAL/PROJ. A non-stereographic
inner layer needs precomputed, device-resident geometry rasters, for example:

- datum MOON_ME position per grid cell, or enough values to reconstruct it;
- local surface normal / ENU frame per grid cell, or enough values to
  reconstruct it;
- pixel-to-metre scale information for distance stepping;
- elevation scale and min/max mipmaps for terrain bounds.

The ray marcher can then bilinear-sample both elevation and geometry fields
instead of assuming stereographic `s0/l0/pixel_size` math. That is the right
path for arbitrary first-layer projections. Polar-stereographic outer layers
can keep the current analytic formula because it is compact, deterministic,
and already byte-pinned.

## Target Layer Combinations

The unified renderer should handle these as specializations of one layer
model:

- `20 m polar` only,
- `1 m local stereographic -> 20 m polar`,
- `1 m custom geometry grid -> 20 m polar`,
- `1 m local/custom -> 5 m polar -> 20 m polar`,
- more polar farfield layers, as long as the kernel launch specializes on
  the layer count and concrete buffer layout.

The implementation should grow in small, pinned steps:

1. Keep the one-source 20 m byte fixtures unchanged.
2. Move shared device math into small helpers that Metal/CUDA inline cleanly.
3. Add identity tests where an LDEM crop is used as the inner layer and the
   same LDEM is used as farfield.
4. Introduce a fixed layout for layer metadata and projection kind. The
   current two-layer stack already passes dimensions, projection kind,
   pixel size, elevation scale, and mipmap base through layer metadata.
5. Move handoff transforms into edge metadata. The current site -> polar
   handoff already passes the 3x3 source-local-to-MOON_ME datum transform
   through edge metadata.
6. Add geometry-grid inner layers only after the stereographic stack is stable.
