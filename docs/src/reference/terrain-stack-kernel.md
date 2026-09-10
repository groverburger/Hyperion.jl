# Terrain stack kernel

A terrain stack joins a detailed inner DEM to lower-resolution distant terrain.
The current kernel supports one inner layer and at most two outer polar layers.
The public mapset API uses `SiteDEMLayer` and `PolarDEMLayer` descriptors.
A mapset with multiple layers must start with a site DEM.

## Layer types

| Layer | Geometry | Typical use |
|---|---|---|
| Site DEM | Supported stereographic projection from a GeoTIFF | Detailed local terrain |
| Polar DEM | South-polar stereographic grid | Local or distant lunar terrain |
| `GeometryGrid` | Explicit coordinate and basis arrays | Low-level synthetic tests and experiments |

`GeometryGrid` is a low-level interface.
The mapset command does not provide a general file loader for arbitrary coordinate systems.
A GeoTIFF input does not imply support for every GDAL projection.

## Ray traversal

The inner layer supplies the output pixel elevation and local coordinate basis.
Each ray traverses the inner terrain before it enters an outer layer.
The layer transition uses the physical exit point and converts the distance to the next layer's pixel units.
This conversion prevents gaps or duplicate distances at different pixel scales.

The projection reference sphere defines the grid coordinates.
Elevated terrain defines the physical ray geometry.
These quantities have different roles and must remain separate in coordinate calculations.

All layers use the shared ray-cast helpers.
The selected kernel supplies the number and types of terrain layers.
Max-elevation mipmaps bound terrain between detailed samples.
The current traversal has no min-elevation shortcut.

## Windows, tiles, and terrain extent

A render window selects output pixels.
It does not limit the terrain available to rays when the full inner DEM is loaded.
A 1 × 1 output window can therefore use the same terrain as a full image.

Tiles divide output work and device buffers.
They keep the loaded inner terrain extent for ray traversal.
This behavior prevents tile edges from removing nearby shadow sources.

A site layer with `cutoff=true` loads only the requested DEM window.
This option changes the available terrain and can change the output.
`SiteDEMLayer` defaults to `cutoff=false`.
The radius command defaults to a cutoff; use `--no-cutoff` to keep the full site DEM.

## Checks

`test/terrain_stack.jl` contains small synthetic layer and window tests.
The same file can run with a CPU or GPU backend.
The larger bit-exact tests compare output with stored images and hashes.
Refer to [tests](../testing.md) for commands and data requirements.
