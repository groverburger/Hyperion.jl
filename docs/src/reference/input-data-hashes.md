# Input data hashes

The terrain files below are external inputs in `data/inputs/`.
Git stores the mapset TOML files in that directory, but it does not store these terrain files.
SHA-256 values identify the exact products used by the corresponding configurations.

Calculate a file hash on macOS or Linux:

```bash
shasum -a 256 data/inputs/ldem_80s_20m.img
```

Calculate a file hash in Windows PowerShell:

```powershell
Get-FileHash data\inputs\ldem_80s_20m.img -Algorithm SHA256
```

| Data product | Path | SHA-256 | Use and format |
|---|---|---|---|
| Shirley south-polar LDEM | `data/inputs/ldem_80s_20m.img` | `caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b` | 20 m reference terrain. Raw 30,400 × 30,400 Int16, little-endian; 0.5 m per count. |
| Nobile 1 m site DEM | `data/inputs/nobile_1m.tif` | `e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4` | 1 m reference terrain for stack tests and baseline maps. |
| WUSTL 2017 south-polar LDEM | `data/inputs/ldem_80s_20m_wustl.img` | `db44c3b2444acff1af528301c8a6988934b490249aceff0ac55668bdf3eab365` | Historical local input; not the current reference terrain. |
| Barker et al. 2023 south-polar LDEM | `data/inputs/LDEM_80S_20MPP_ADJ.TIF` | `09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649` | GeoTIFF from GSFC PGDA product 90; used by Barker mapset specifications. |
| VIPER 8.0 crop | `data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif` | `85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19` | VIPER crop mapsets |
| VIPER medium extent | `data/inputs/viper_sfs_dem_8_0_v71_2027_medium_extent.tif` | `71c6318c84c11777495bb1fe39a2121ba70edc6ad62b500ac27a82a7e225333b` | Medium-extent mapsets and hilltop probe |
| Synthetic hill | `data/inputs/mock_lunar_south_pole_20m_hill_128.tif` | `e7f6b1a551167f7c0b6c070e0a6be736c37d79c4786d48ddcb420d9d5bb12116` | Local synthetic mapset input |

The full `viper_sfs_dem_8_0.tif` used by the radius command has no hash in this table.
A matching filename alone does not prove that two terrain files are identical.

If an input changes, examine the affected mapset hashes, test images, and correctness baselines.
Make new references only after you establish why the output changed.
The [data guide](../data.md) explains the formats and necessary inputs.
