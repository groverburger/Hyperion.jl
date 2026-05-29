# Raycasting

Hyperion renders one work item per output pixel. For each pixel it computes
Sun/Earth direction, marches rays through terrain, and keeps the maximum
terrain slope above the local horizon.

## Algorithm Sketch

```text
for each output pixel:
  compute query point position on R_moon + elevation
  compute Sun/Earth azimuth and elevation
  cast 8 Sun-disk rays and 1 Earth ray
  bilinear-sample terrain along each ray
  convert each sample to 3D Moon coordinates
  keep the maximum horizon slope
  encode Sun fraction and DSN over-horizon angle
```

## Important Details

- DEM samples are converted to meters with each layer's `elev_scale_to_m`.
- Raw Shirley `.img` uses Int16 half-meter counts.
- GeoTIFF DEMs use Float32 meters.
- Bilinear interpolation uses an explicit `fma` chain for deterministic
  rounding.
- Horizon slopes are compared as `(num, den_sq)` to avoid per-sample `atan`,
  `sqrt`, and division.
- Farfield ray marching uses max-mipmap conservative skipping.
- Trigonometry in the hot path uses deterministic LUTs.

See the reference algorithm notes for the full derivation.
