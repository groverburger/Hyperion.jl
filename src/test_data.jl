# ─── Test-data provisioning ───────────────────────────────────────────────
#
# The LDEM (1.85 GB raster, see README.md → Farfield LDEM versions)
# and its derived Nobile GTiff aren't in git. These helpers fetch +
# SHA-verify them idempotently so a user can go from `git clone` to
# `] test` with no manual setup step.
#
# KNOWN DOWNLOAD MISMATCH: `_LDEM_SHA` below is pinned to the "Shirley"
# legacy artefact (caaf017f…), the file Mark Shirley shipped with the
# upstream C# Mapbuilder pipeline. The download URL points at the
# canonical PDS Geosciences Node @ WUSTL — but a live download from
# there serves a different (newer) version of the same nominal product
# with a different SHA, so `ensure_ldem!()` fails SHA-verify on a
# fresh fetch. This is intentional: the Shirley file is preserved as
# our test baseline so the Hyperion bit-exact regression matches the
# Mapbuilder reference renders byte-for-byte. Migration plan: repin
# against either WUSTL 2017 (live) or 2023 Barker, regenerate the
# bit-exact pins, drop the Shirley dependency.
#
# `ensure_ldem!` is the minimum needed for `test/runtests.jl` and
# `scripts/bitexact_test.jl`. `ensure_test_data!` additionally derives
# `data/inputs/nobile_20m.tif` for diagnostic scripts that want a
# pre-cropped Nobile window.

import Downloads
using Printf

const _LDEM_URL   = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/LDEM_80S_20M.IMG"
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
    ensure_ldem!() -> String

Idempotently ensure the 1.85 GB LDEM_80S_20M raster is present and
SHA-verified at `data/inputs/ldem_80s_20m.img` (the "Shirley" baseline,
SHA `caaf017f…`; see README.md → Farfield LDEM versions). Downloads
from the PDS Geosciences Node @ WUSTL if missing or corrupted — but
note that a fresh download will fail SHA-verify because the live URL
serves a different version than Shirley (this is a known migration
issue). Returns the path.

Called automatically by `test/runtests.jl` and `scripts/bitexact_test.jl`.
"""
function ensure_ldem!()
    if _ldem_ok()
        return _ldem_path()
    end
    mkpath(_data_dir())

    if isfile(_ldem_path())
        actual = _sha256_file(_ldem_path())
        @warn "LDEM present but SHA mismatch — re-downloading" path=_ldem_path() expected=_LDEM_SHA actual
    else
        @info "LDEM not found; fetching one-time $(_LDEM_SIZE_GB) GB from PDS Geosciences @ WUSTL — note: fresh download may fail SHA-verify against Shirley pin (see test_data.jl header)" url=_LDEM_URL dest=_ldem_path()
    end

    t0 = time()
    last_pct = Ref(-5.0)
    Downloads.download(_LDEM_URL, _ldem_path(); progress = (total, now) -> begin
        total > 0 || return
        pct = 100 * now / total
        # Emit every 5% so the log stays readable instead of spammy.
        if pct - last_pct[] >= 5.0 || (now == total && last_pct[] < 100.0)
            last_pct[] = pct
            elapsed = time() - t0
            mbps = now / 1e6 / max(elapsed, 0.001)
            eta  = mbps > 0 ? (total - now) / 1e6 / mbps : 0.0
            @printf("  [%5.1f%%]  %.2f / %.2f GB  @  %.1f MB/s  (ETA %.0fs)\n",
                    pct, now/1e9, total/1e9, mbps, eta)
            flush(stdout)
        end
    end)
    @info "Download complete" seconds=round(time()-t0; digits=1)

    print("  Verifying SHA-256 ... "); flush(stdout)
    actual = _sha256_file(_ldem_path())
    println("done")
    if actual != _LDEM_SHA
        error("LDEM SHA mismatch after download.\n  expected: $_LDEM_SHA\n  actual:   $actual")
    end
    @info "LDEM SHA verified" path=_ldem_path()
    return _ldem_path()
end

function _nobile_matches(expected::Matrix{Float32})
    isfile(_nobile_path()) || return false
    ds = ArchGDAL.read(_nobile_path())
    band = ArchGDAL.getband(ds, 1)
    actual = ArchGDAL.read(band)
    size(actual) == size(expected) && actual == expected
end

"""
    ensure_test_data!() -> (ldem_path, nobile_path)

Like `ensure_ldem!()` but additionally derives the Nobile-window GTiff
(`data/inputs/nobile_20m.tif`, 512×896 Float32) used by diagnostic
scripts (`azimuth_range.jl`, `select_test_timestamps.jl`).

Idempotent — skips the derive step if the GTiff already matches the
bytes extracted from the LDEM window.
"""
function ensure_test_data!()
    ensure_ldem!()

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
