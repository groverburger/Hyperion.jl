# Input Data Hashes

Hyperion keeps canonical local data products under `data/inputs/`.
The directory is intentionally gitignored because these files are large,
but code and scripts should treat these paths as the local source of truth.

Use SHA-256 to verify a local checkout:

```sh
shasum -a 256 data/inputs/ldem_80s_20m.img
shasum -a 256 data/inputs/nobile_1m.tif
```

On Windows PowerShell:

```powershell
Get-FileHash data\inputs\ldem_80s_20m.img -Algorithm SHA256
Get-FileHash data\inputs\nobile_1m.tif -Algorithm SHA256
```

| Logical name | Canonical repo-local path | SHA-256 | Notes |
|---|---|---|---|
| Shirley south-polar LDEM | `data/inputs/ldem_80s_20m.img` | `caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b` | Current bit-exact 20 m farfield baseline. Raw 30400x30400 Int16 little-endian, scale 0.5 m/count. |
| Nobile 1 m site DEM | `data/inputs/nobile_1m.tif` | `e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4` | Current high-resolution site DEM for 1 m stack tests and Tier 0 1 m + 20 m farfield generation. |
| WUSTL 2017 south-polar LDEM | `data/inputs/ldem_80s_20m_wustl.img` | `db44c3b2444acff1af528301c8a6988934b490249aceff0ac55668bdf3eab365` | Official WUSTL-style raw LDEM present locally. Not the current bit-exact baseline. |
| Barker et al. 2023 south-polar LDEM | `data/inputs/LDEM_80S_20MPP_ADJ.TIF` | `09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649` | Native GeoTIFF from GSFC PGDA product 90. Supported by `load_ldem`, but not the current bit-exact baseline. |

When an input artifact is changed, regenerate any affected pinned SHAs,
Tier 0 baselines, and cross-vendor verification outputs in the same change.
