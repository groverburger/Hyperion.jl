#!/usr/bin/env julia
# Fetches the PDS LOLA 80°S 20 m/pixel LDEM and derives the Nobile target DEM.
#
# Usage:
#     julia --project scripts/fetch_test_data.jl
#
# Idempotent: re-running checks SHAs and skips work that's already done.

using Pkg
Pkg.activate(dirname(@__DIR__))

import Downloads
import SHA
import Mmap
import ArchGDAL
import ArchGDAL as AG
using Printf

const REPO_ROOT  = dirname(@__DIR__)
const DATA_DIR   = joinpath(REPO_ROOT, "data", "inputs")

const LDEM_URL   = "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/LDEM_80S_20M.IMG"
const LDEM_PATH  = joinpath(DATA_DIR, "ldem_80s_20m.img")
const LDEM_SHA   = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
const LDEM_H     = 30400
const LDEM_W     = 30400

const NOBILE_PATH = joinpath(DATA_DIR, "nobile_20m.tif")

# Nobile window in LDEM pixel coordinates (0-indexed rows/cols, row-major).
# Verified byte-exact against the original nobile_20m.tif: pixel values match,
# zero diff across all 458,752 pixels. UL pixel center lands at (64650, 124790)
# in polar-stereographic coords because the raw .img uses PDS pixel-center
# georeferencing with UL center at (-303990, 303990).
const NOBILE_ROW   = 8960
const NOBILE_COL   = 18432
const NOBILE_H     = 512
const NOBILE_W     = 896
const NOBILE_UL_E  = 64650.0
const NOBILE_UL_N  = 124790.0
const NOBILE_PIXEL = 20.0
const NOBILE_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"

function sha256_file(path)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function file_ok(path, expected_sha)
    isfile(path) || return false
    actual = sha256_file(path)
    if actual != expected_sha
        @warn "SHA mismatch" path expected=expected_sha actual
        return false
    end
    return true
end

function download_ldem()
    if file_ok(LDEM_PATH, LDEM_SHA)
        @info "LDEM already present with matching SHA — skipping download" path=LDEM_PATH
        return
    end
    mkpath(DATA_DIR)
    @info "Downloading LDEM from PDS (1.85 GB, one-time)" url=LDEM_URL
    t0 = time()
    last_pct = Ref(-5.0)
    Downloads.download(LDEM_URL, LDEM_PATH; progress=(total, now) -> begin
        total > 0 || return
        pct = 100 * now / total
        # Print every 5% (and once at 100%) so the log is readable, not a spam of updates
        if pct - last_pct[] >= 5.0 || (now == total && last_pct[] < 100.0)
            last_pct[] = pct
            elapsed = time() - t0
            mbps = now / 1e6 / max(elapsed, 0.001)
            eta = mbps > 0 ? (total - now) / 1e6 / mbps : 0.0
            @printf("  [%5.1f%%]  %.2f / %.2f GB  @  %.1f MB/s  (ETA %.0fs)\n",
                    pct, now/1e9, total/1e9, mbps, eta)
            flush(stdout)
        end
    end)
    @info "Download complete" seconds=round(time()-t0; digits=1)
    print("  Verifying SHA-256 ... "); flush(stdout)
    actual = sha256_file(LDEM_PATH)
    println("done")
    if actual != LDEM_SHA
        error("LDEM SHA mismatch after download.\n  expected: $LDEM_SHA\n  actual:   $actual")
    end
    @info "LDEM SHA verified"
end

function ldem_crop_window()
    raw = Mmap.mmap(open(LDEM_PATH, "r"), Matrix{Int16}, (LDEM_W, LDEM_H))
    col_range = (NOBILE_COL+1):(NOBILE_COL+NOBILE_W)
    row_range = (NOBILE_ROW+1):(NOBILE_ROW+NOBILE_H)
    return Float32.(raw[col_range, row_range]) .* 0.5f0   # (W=896, H=512)
end

function nobile_matches(expected::Matrix{Float32})
    isfile(NOBILE_PATH) || return false
    ds = AG.read(NOBILE_PATH)
    band = AG.getband(ds, 1)
    actual = AG.read(band)                               # (W, H)
    size(actual) == size(expected) && actual == expected
end

function derive_nobile()
    expected = ldem_crop_window()
    if nobile_matches(expected)
        @info "Nobile DEM already present and matches LDEM window — skipping" path=NOBILE_PATH
        return
    end
    @info "Writing Nobile DEM" path=NOBILE_PATH rows="$(NOBILE_ROW):$(NOBILE_ROW+NOBILE_H)" cols="$(NOBILE_COL):$(NOBILE_COL+NOBILE_W)"
    AG.create(NOBILE_PATH;
              driver = AG.getdriver("GTiff"),
              width  = NOBILE_W,
              height = NOBILE_H,
              nbands = 1,
              dtype  = Float32) do dst
        AG.setgeotransform!(dst, [NOBILE_UL_E, NOBILE_PIXEL, 0.0, NOBILE_UL_N, 0.0, -NOBILE_PIXEL])
        AG.setproj!(dst, NOBILE_PROJ4)
        band = AG.getband(dst, 1)
        AG.write!(band, expected)
    end
end

function main()
    download_ldem()
    derive_nobile()
    @info "Test data ready"
end

main()
