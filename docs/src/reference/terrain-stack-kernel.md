# Terrain-Stack Kernel Direction

This note records the intended shape of the unified terrain-stack renderer.
It should be read alongside `docs/src/reference/algorithms.md` and
`docs/src/reference/cross-vendor-determinism.md`; all kernel changes must preserve the
same bit-exactness rules: explicit `fma` for reconstructible multiply-add
or multiply-subtract dataflow, no `log2` for mipmap level selection, no
new hot-loop transcendental calls, and test coverage against the pinned
20 m byte fixtures.

## Current State

The one-source 20 m renderer, the site -> polar farfield renderer, and the
geometry-grid -> polar farfield renderer now share the hot ray-marching core:

- level-0 bilinear terrain sampling and slope accumulation,
- hierarchical mipmap skipping for polar-stereographic farfield layers,
- sun-disk integration,
- UInt8 output encoding.

The one-source polar case still has a separate outer GPU kernel because its
per-pixel setup starts and ends in one polar grid. The layered stack kernel
handles the mixed inner/farfield cases:

- a single polar-stereographic layer starts and ends in one grid;
- a site layer starts in a local stereographic grid, exits that raster, then
  hands the ray to one or two polar-stereographic farfield grids;
- a geometry-grid layer starts in caller-supplied local coordinates, exits
  that raster, then hands the ray to one or two polar-stereographic farfield
  grids.

## Handoff Rules

Layer handoff has two separate pieces of geometry:

- **Ray geometry uses the true elevated query position.** This is the
  position used for slope tests against terrain samples.
- **Map reprojection uses the datum position.** Row/column coordinates in
  the next layer are planimetric and must not shift with terrain elevation.
  The datum point is the surface point at lunar radius along the query
  normal. Projecting the elevated terrain point caused multi-cell farfield
  offsets at Nobile elevations.
- **Inner arbitrary grids use local ray coordinates.** The geometry-grid
  path does not subtract two MOON_ME positions at lunar-radius magnitude in
  the device hot loop. Each cell supplies small local datum coordinates for
  slope tests and separate MOON_ME datum coordinates for handoff projection.
  This preserves the same precision rule as the stereographic path.

When a ray exits a layer, the next layer starts at the same physical
distance along the ray:

```
next_start_pixels = max(1, exit_distance_pixels * from_pixel_size_m / to_pixel_size_m)
```

Each segment is bounded by both a dynamic terrain-height limit and the
distance to leave that layer's raster. There is no fixed hard cap in the
stacked path.

## Projection Model

The current implementation supports three innermost/farfield forms:

- **Local stereographic site grid.** Used by `SiteDEM`; this is a
  stereographic grid centered on `(lat0, lon0)` and represented in a local
  frame before entering the kernel.
- **Geometry-grid inner layer.** Used by `GeometryGridTerrain`; the GPU
  bilinear-samples precomputed local datum coordinates and up vectors instead
  of evaluating a map projection. CPU precompute uses the supplied local
  frame to compute Sun/DSN ray directions and uses separate MOON_ME datum
  rasters to project the handoff point.
- **South polar stereographic grid.** Used by global or cropped LDEM
  farfield layers.

The GPU cannot call GDAL/PROJ. Arbitrary innermost DEM projections are
therefore represented by precomputed geometry rasters, not by device-side
projection callbacks. Polar-stereographic outer layers keep the analytic
formula because it is compact, deterministic, and already byte-pinned.

## Target Layer Combinations

The unified renderer should handle these as specializations of one layer
model:

- `20 m polar` only,
- `1 m local stereographic -> 20 m polar`,
- `1 m local/custom -> 5 m polar -> 20 m polar`,
- `1 m custom geometry grid -> 20 m polar`,
- `1 m custom geometry grid -> 5 m polar -> 20 m polar`,
- more polar farfield layers, as long as the kernel launch specializes on
  the layer count and concrete buffer layout.

The implementation should grow in small, pinned steps:

1. Keep the one-source 20 m byte fixtures unchanged.
2. Move shared device math into small helpers that Metal/CUDA inline cleanly.
3. Add identity tests where an LDEM crop is used as the inner layer and the
   same LDEM is used as farfield.
4. Introduce a fixed layout for layer metadata and projection kind. The
   current two-layer stack already passes dimensions, projection kind,
   pixel size, elevation scale, and mipmap base through fixed-capacity
   layer metadata buffers sized for three layers.
5. Move handoff transforms into edge metadata. The current site -> polar
   handoff already passes the 3x3 source-local-to-MOON_ME datum transform
   through fixed-capacity edge metadata buffers sized for two handoff edges.
6. Support three-layer local-stereo -> polar -> polar stacks. The current
   GPU stack path supports a high-resolution local stereographic source
   followed by two polar-stereographic farfield sources, including differing
   pixel sizes such as `1 m -> 5 m -> 20 m`.
7. Support geometry-grid inner layers. The current GPU stack path supports
   geometry-grid -> polar and geometry-grid -> polar -> polar stacks. Tests
   cover geometry-grid handoff, flat-field equivalence, farfield blocking,
   and three-layer continuation.
