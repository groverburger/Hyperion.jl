# ─── Local test-data checks ───────────────────────────────────────────────
#
# The LDEM (1.85 GB raster, see README.md → Farfield LDEM versions)
# and its derived Nobile GTiff aren't in git. These helpers validate the
# local Shirley DEM and derive small local artifacts from it. They never
# download data.
#
# `_LDEM_SHA` below is pinned to the "Shirley" legacy artefact
# (caaf017f…), the file Mark Shirley shipped with the upstream C#
# Mapbuilder pipeline. The canonical PDS Geosciences Node @ WUSTL now
# serves a different version of the same nominal product with a
# different SHA. Do not auto-fetch from WUSTL: tests are pinned to
# Shirley. Migration plan: repin against either current WUSTL or 2023
# Barker, regenerate the bit-exact pins, drop the Shirley dependency.
#
# `require_shirley_ldem!` is the minimum needed for `test/runtests.jl`
# and `scripts/bitexact_test.jl`. `require_test_data!` additionally derives
# `data/inputs/nobile_20m.tif` for diagnostic scripts that want a
# pre-cropped Nobile window.

const _LDEM_SHA   = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
const _LDEM_DIM   = 30400                     # LDEM is 30400×30400 Int16
const _LDEM_SIZE_GB = 1.85

# Nobile test window in LDEM pixel coordinates. These values are verified
# byte-exact against the original nobile_20m.tif from the upstream reference
# pipeline (see scripts/fetch_test_data.jl history for the provenance).
const _NOBILE_ROW   = 8960
const _NOBILE_COL   = 18432
const _NOBILE_H     = 512
const _NOBILE_W     = 896
const _NOBILE_UL_E  = 64650.0
const _NOBILE_UL_N  = 124790.0
const _NOBILE_PIXEL = 20.0
const _NOBILE_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"

_data_dir()    = joinpath(dirname(@__DIR__), "data", "inputs")
_ldem_path()   = joinpath(_data_dir(), "ldem_80s_20m.img")
_nobile_path() = joinpath(_data_dir(), "nobile_20m.tif")

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _ldem_ok()
    isfile(_ldem_path()) && _sha256_file(_ldem_path()) == _LDEM_SHA
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

function _nobile_matches(expected::Matrix{Float32})
    isfile(_nobile_path()) || return false
    ds = ArchGDAL.read(_nobile_path())
    band = ArchGDAL.getband(ds, 1)
    actual = ArchGDAL.read(band)
    size(actual) == size(expected) && actual == expected
end

"""
    require_test_data!() -> (ldem_path, nobile_path)

Like `require_shirley_ldem!()` but additionally derives the Nobile-window GTiff
(`data/inputs/nobile_20m.tif`, 512×896 Float32) used by diagnostic
scripts (`azimuth_range.jl`, `select_test_timestamps.jl`).

Idempotent — skips the derive step if the GTiff already matches the
bytes extracted from the LDEM window.
"""
function require_test_data!()
    require_shirley_ldem!()

    # Read the Nobile window directly from the mmapped LDEM.
    raw = Mmap.mmap(open(_ldem_path(), "r"), Matrix{Int16}, (_LDEM_DIM, _LDEM_DIM))
    col_range = (_NOBILE_COL + 1):(_NOBILE_COL + _NOBILE_W)
    row_range = (_NOBILE_ROW + 1):(_NOBILE_ROW + _NOBILE_H)
    expected = Float32.(raw[col_range, row_range]) .* 0.5f0

    if _nobile_matches(expected)
        return (_ldem_path(), _nobile_path())
    end

    @info "Deriving Nobile GTiff from LDEM" path=_nobile_path()
    ArchGDAL.create(_nobile_path();
                    driver = ArchGDAL.getdriver("GTiff"),
                    width  = _NOBILE_W, height = _NOBILE_H,
                    nbands = 1, dtype = Float32) do dst
        ArchGDAL.setgeotransform!(dst, [_NOBILE_UL_E, _NOBILE_PIXEL, 0.0,
                                        _NOBILE_UL_N, 0.0, -_NOBILE_PIXEL])
        ArchGDAL.setproj!(dst, _NOBILE_PROJ4)
        band = ArchGDAL.getband(dst, 1)
        ArchGDAL.write!(band, expected)
    end
    return (_ldem_path(), _nobile_path())
end
