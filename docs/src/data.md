# Data I/O

Hyperion does not download DEM products during tests. Required inputs live
under `data/inputs/` and are validated by SHA where the workflow depends on a
specific product.

## Shirley LDEM

The current baseline farfield is:

```text
data/inputs/ldem_80s_20m.img
sha256 caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b
format raw 30400 x 30400 Int16 little-endian
scale  0.5 m/count
```

## VIPER 8.0 Nobile Crop

```text
data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif
sha256 85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19
format GeoTIFF, Float32 meters
```

## Site DEM Height Encoding

`load_site_dem` converts source meters to Int16 half-meter counts.
`load_site_dem_f32` keeps Float32 meters. Mapset site layers inspect the
GeoTIFF band type at runtime and preserve floating-point sources as
Float32 meters.

## Hash Policy

- Supported mapset specs include `sha256` for every external layer.
- Fixture-building tools write input SHA manifests.
- If a workflow uses external data but cannot validate a hash, document why.
