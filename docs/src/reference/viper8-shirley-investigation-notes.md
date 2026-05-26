# VIPER 8.0 1m + Shirley 20m Map Generation Investigation

Started: 2026-05-22 02:25:49 PDT

## Context

- Problem combination: VIPER 8.0 1m crop site DEM + Shirley 20m farfield DEM.
- Other combinations reportedly work: Shirley-only, and `nobile_1m.tif` + Shirley.
- Observed symptoms: sometimes the run stalls without generating output; sometimes it produces output; one recent run caused extreme memory pressure and crashed the computer.
- Prior suspicion from lost context: VIPER 8.0 1m crop may contain `NaN` values.

## Guardrails

- Avoid loading whole DEM rasters into memory unless the dimensions and element type show it is safe.
- Prefer metadata inspection and windowed/block reads.
- Keep notes in this file as the investigation proceeds.

## Running Notes

- 2026-05-22 02:25 PDT: Recreated context from the repo. Worktree already has modifications in `src/live_helpers.jl`, `src/mapsets.jl`, `src/terrain_stack.jl`, `test/fixtures/bitexact/known_good_1m_viper8_barker2023.jl`, and `test/terrain_stack.jl`; treat them as pre-existing user/session work.
- 2026-05-22 02:27 PDT: `generate_viper8_1m_shirley_mapset.jl` uses the pre-cropped VIPER 8.0 file via `require_viper8_nobile_crop_tif!()`. `generate_viper8_1m_radius_shirley_mapset.jl` defaults to the full `data/inputs/viper_sfs_dem_8_0.tif` but then asks for a 500 x 500 render window with `cutoff=true`.
- 2026-05-22 02:30 PDT: `src/mapsets.jl` currently applies `cutoff` after `load_site_dem_f32(spec.path)`, so the full VIPER 8.0 DEM is read into memory before cropping. This is a likely memory-pressure issue for radius/full-DEM runs.
- 2026-05-22 02:32 PDT: `gdalinfo -mm` metadata:
  - VIPER 8.0 native crop: 5040 x 4144 Float32, row-strip blocks of 5040 x 1, NoData -1e6, computed min/max 5980.641/6464.644, valid percent 100.
  - Full VIPER 8.0: 14336 x 11008 Float32, tiled 256 x 256, NoData -1e6, computed min/max 5064.355/6464.644, valid percent 100.
  - `nobile_1m.tif`: 4992 x 4096 Float32, row-strip blocks of 4992 x 1, NoData -3.4028235e38, computed min/max 5989.398/6464.754, valid percent 100.
- 2026-05-22 02:34 PDT: Windowed/block scan with rasterio found no NaNs, infinities, or NoData pixels in VIPER crop, full VIPER, or `nobile_1m.tif`.
- 2026-05-22 02:35 PDT: VIPER crop dimensions 4144 x 5040 are not divisible by 16. `nobile_1m.tif` dimensions 4096 x 4992 are divisible by 16. In layered mapsets, `_site_mapset_mipmaps` returns `(nothing, nothing)` if non-divisible, so no site mipmap pyramid is built for the VIPER crop in layered mode.
- 2026-05-22 02:39 PDT: Existing output directories show several full VIPER-crop runs reached `manifest.csv`/hillshade/azimuth generation but produced zero frames. Small radius/full-VIPER cutoff runs produced frames; a 500 x 500 full-VIPER cutoff run produced 214 Sun/DSN frames.
- 2026-05-22 02:43 PDT: Current worktree already included a memory refactor before this investigation: reusable layered GPU context, no per-frame debug outputs for mapsets, and reusable packed precompute buffer. This should reduce repeated allocation, but the full crop still has a large persistent packed buffer.
- 2026-05-22 02:52 PDT: Added windowed site DEM loading. `load_site_dem` and `load_site_dem_f32` now accept a zero-based `(origin_r, origin_c, H, W)` window and adjust `s0`/`l0`; `SiteDEMLayer(..., cutoff=true, window=...)` now reads only that window instead of reading the full DEM and then copying the crop.
- 2026-05-22 02:54 PDT: Changed `generate_viper8_1m_radius_shirley_mapset.jl` to resolve its window from `read_site_dem_info`, avoiding a full DEM load before dry-run or cutoff rendering.
- 2026-05-22 02:56 PDT: Verified windowed loader on full VIPER 8.0 DEM: a 16 x 16 window matched full-load pixels and adjusted `s0`/`l0`.
- 2026-05-22 03:00 PDT: Ran one end-to-end 128 x 128 VIPER full DEM + Shirley frame with the windowed loader: `viper8_radius_windowed_loader_probe` completed and wrote frames. Process RSS during setup was about 4.4 GB, dominated by Shirley farfield/mipmap/context work rather than the VIPER window.
- 2026-05-22 03:03 PDT: Focused terrain-stack tests passed: `julia --project test/terrain_stack.jl` reported 47/47 passing on Metal.
- 2026-05-22 later: Added tiled layered mapset rendering. For layered site + polar mapsets larger than `tile_height x tile_width` (defaults 1024 x 1024), `generate_mapset` now renders site tiles and stitches the full Sun/DSN frames. Farfield device mipmaps are prepared once and reused across tiles.
- 2026-05-22 later: Verified tiled rendering against the original full-context path on a small synthetic CPU case; Sun and DSN outputs matched exactly.
- 2026-05-22 later: Focused terrain-stack tests still pass after tiling changes: `julia --project test/terrain_stack.jl` reported 47/47 passing on Metal.
- 2026-05-22 later: Ran bounded real-data probe `viper8_crop_tiled_512_probe`: VIPER crop window `(0, 0, 512, 512)` + Shirley, one timestamp, 256 x 256 tiles. It completed and wrote Sun/DSN frames. Observed process RSS during run was about 3.25 GiB at ~40 s and about 4.2 GiB at ~112 s.
- 2026-05-22 later: Ran bounded real-data probe `viper8_crop_tiled_1024_probe`: VIPER crop window `(0, 0, 1024, 1024)` + Shirley, one timestamp, 512 x 512 tiles. It completed and wrote Sun/DSN frames. Observed process RSS was about 3.32 GiB at ~40 s and about 4.21 GiB at ~111 s.
- 2026-05-22 later: Initial tiled implementation cropped the site DEM per tile. A 512 x 512 tiled-vs-untiled comparison showed Sun matched exactly, but DSN differed in 92 pixels. Cause: rays handed off to Shirley at tile boundaries instead of continuing through the full 1 m site nearfield.
- 2026-05-22 later: Fixed tiled rendering to use tiles only as output/precompute windows while retaining the full loaded site DEM as nearfield. Re-ran 512 x 512 real-data tiled-vs-untiled comparison: Sun and DSN PNG SHA-256 hashes matched exactly; pixel comparison had 0 differences.
- 2026-05-22 later: Re-ran focused terrain-stack tests after the tiling fix: `julia --project test/terrain_stack.jl` reported 47/47 passing on Metal.
- 2026-05-22 later: Ran corrected full-frame real-data probe `viper8_crop_full_frame_tiled_fixed_probe`: full `nobile_area_viper_sfs_dem_8_0_native_crop.tif` 5040 x 4144 site DEM + Shirley, one timestamp, default 1024 x 1024 tiles. It completed and wrote full-size 5040 x 4144 Sun/DSN PNGs. Observed RSS: ~3.95 GiB at 44 s, ~4.69 GiB at 2:23, ~5.75 GiB during tile rendering. No runaway memory pressure observed.
- 2026-05-22 later: Ran full repo test suite: `julia --project test/runtests.jl`. It completed successfully on Metal. Reported sections included deterministic math, projection, mipmap pyramid, all cross-platform bit-exactness timestamps, all 1m full + 20m farfield bit-exactness timestamps, VIPER 8.0 crop + Barker 2023 dimensions, terrain-stack farfield continuation, and Tier 0 correctness aggregate baseline with 0 regressions.

## Current Assessment

- The VIPER crop does not appear to contain NaNs or NoData holes.
- The full-VIPER radius script had a definite avoidable memory problem: it loaded the full 14336 x 11008 Float32 DEM before cropping. This is now fixed in the worktree.
- The pre-cropped VIPER 8.0 + Shirley full-window mapset remains intrinsically heavy: 4144 x 5040 output pixels require a 14-plane Float32 packed context of about 1.17 GiB on CPU, plus a device copy, DEM/device arrays, output buffers, and Shirley farfield mipmaps.
- Full-window zero-frame runs likely failed between metadata/manifest writing and first frame save, i.e. during layered GPU context setup, precompute, device copy, or the first kernel.
