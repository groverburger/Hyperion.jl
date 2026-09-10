# Validate external terrain inputs against their expected SHA-256 values.
# The files reside in data/inputs/ and are not stored in Git.
# This module does not download or derive terrain files.
# See docs/src/reference/input-data-hashes.md for the supported products.

const _LDEM_SHA   = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
const _NOBILE_1M_SHA = "e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4"
const _BARKER_2023_LDEM_SHA = "09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649"
const _VIPER8_NOBILE_CROP_SHA = "85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19"
const _LDEM_SIZE_GB = 1.85

_data_dir()    = joinpath(dirname(@__DIR__), "data", "inputs")
_ldem_path()   = joinpath(_data_dir(), "ldem_80s_20m.img")
_barker_2023_ldem_path() = joinpath(_data_dir(), "LDEM_80S_20MPP_ADJ.TIF")
_nobile_1m_path() = joinpath(_data_dir(), "nobile_1m.tif")
_viper8_nobile_crop_path() =
    joinpath(_data_dir(), "nobile_area_viper_sfs_dem_8_0_native_crop.tif")

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _ldem_ok()
    isfile(_ldem_path()) && _sha256_file(_ldem_path()) == _LDEM_SHA
end

function _nobile_1m_ok()
    isfile(_nobile_1m_path()) && _sha256_file(_nobile_1m_path()) == _NOBILE_1M_SHA
end

function _barker_2023_ldem_ok()
    isfile(_barker_2023_ldem_path()) &&
        _sha256_file(_barker_2023_ldem_path()) == _BARKER_2023_LDEM_SHA
end

function _viper8_nobile_crop_ok()
    isfile(_viper8_nobile_crop_path()) &&
        _sha256_file(_viper8_nobile_crop_path()) == _VIPER8_NOBILE_CROP_SHA
end

"""
    require_shirley_ldem!() -> String

Require the 1.85 GB Shirley LDEM_80S_20M raster to be present and
SHA-verified at `data/inputs/ldem_80s_20m.img`. Returns the path.

This function never downloads data. If the file is missing or has the
wrong SHA, it errors with placement instructions.
"""
function require_shirley_ldem!()
    if _ldem_ok()
        return _ldem_path()
    end

    if isfile(_ldem_path())
        actual = _sha256_file(_ldem_path())
        error("""
        Local LDEM is not the Shirley baseline.

          path:     $(_ldem_path())
          expected: $(_LDEM_SHA)
          actual:   $actual

        Download or copy the Shirley LDEM artifact and place it exactly at:

          $(_ldem_path())

        Expected format: raw 30400x30400 Int16 little-endian, scale 0.5 m,
        no header, about $(_LDEM_SIZE_GB) GB. See README.md -> Farfield LDEM versions.
        """)
    else
        error("""
        Shirley LDEM not found.

        Download or copy the Shirley LDEM artifact and place it exactly at:

          $(_ldem_path())

        Expected SHA-256:

          $(_LDEM_SHA)

        Expected format: raw 30400x30400 Int16 little-endian, scale 0.5 m,
        no header, about $(_LDEM_SIZE_GB) GB. See README.md -> Farfield LDEM versions.
        """)
    end
end

"""
    require_nobile_1m_tif!() -> String

Require the 1 m Nobile site DEM to be present and SHA-verified at
`data/inputs/nobile_1m.tif`. Returns the path.
"""
function require_nobile_1m_tif!()
    if _nobile_1m_ok()
        return _nobile_1m_path()
    end

    if isfile(_nobile_1m_path())
        actual = _sha256_file(_nobile_1m_path())
        error("""
        Local Nobile 1 m site DEM has the wrong SHA.

          path:     $(_nobile_1m_path())
          expected: $(_NOBILE_1M_SHA)
          actual:   $actual

        Place the SHA-pinned file at:

          $(_nobile_1m_path())

        See docs/input-data-hashes.md.
        """)
    else
        error("""
        Nobile 1 m site DEM not found.

        Place the SHA-pinned file at:

          $(_nobile_1m_path())

        Expected SHA-256:

          $(_NOBILE_1M_SHA)

        See docs/input-data-hashes.md.
        """)
    end
end

"""
    require_barker_2023_ldem!() -> String

Require the Barker et al. 2023 20 m south-polar LDEM GeoTIFF to be present
and SHA-verified at `data/inputs/LDEM_80S_20MPP_ADJ.TIF`.
"""
function require_barker_2023_ldem!()
    if _barker_2023_ldem_ok()
        return _barker_2023_ldem_path()
    end

    if isfile(_barker_2023_ldem_path())
        actual = _sha256_file(_barker_2023_ldem_path())
        error("""
        Local Barker 2023 LDEM has the wrong SHA.

          path:     $(_barker_2023_ldem_path())
          expected: $(_BARKER_2023_LDEM_SHA)
          actual:   $actual

        Place the SHA-pinned GeoTIFF at:

          $(_barker_2023_ldem_path())

        See docs/input-data-hashes.md.
        """)
    else
        error("""
        Barker 2023 LDEM not found.

        Place the SHA-pinned GeoTIFF at:

          $(_barker_2023_ldem_path())

        Expected SHA-256:

          $(_BARKER_2023_LDEM_SHA)

        See docs/input-data-hashes.md.
        """)
    end
end

"""
    require_viper8_nobile_crop_tif!() -> String

Require the native VIPER 8.0 crop covering the Nobile area to be present and
SHA-verified at `data/inputs/nobile_area_viper_sfs_dem_8_0_native_crop.tif`.
"""
function require_viper8_nobile_crop_tif!()
    if _viper8_nobile_crop_ok()
        return _viper8_nobile_crop_path()
    end

    if isfile(_viper8_nobile_crop_path())
        actual = _sha256_file(_viper8_nobile_crop_path())
        error("""
        Local VIPER 8.0 Nobile-area crop has the wrong SHA.

          path:     $(_viper8_nobile_crop_path())
          expected: $(_VIPER8_NOBILE_CROP_SHA)
          actual:   $actual

        Recreate or place the SHA-pinned crop at:

          $(_viper8_nobile_crop_path())

        See docs/input-data-hashes.md.
        """)
    else
        error("""
        VIPER 8.0 Nobile-area crop not found.

        Place the SHA-pinned crop at:

          $(_viper8_nobile_crop_path())

        Expected SHA-256:

          $(_VIPER8_NOBILE_CROP_SHA)

        See docs/input-data-hashes.md.
        """)
    end
end
