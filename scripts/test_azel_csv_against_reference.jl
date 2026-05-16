#!/usr/bin/env julia
# Regenerate the legacy reference `azimuths_elevations.csv` and verify
# numerical parity column-by-column. The reference is at
# data/inputs/nobile_27_28_20m/other/azimuths_elevations.csv
# and was generated with the same SPICE kernels Hyperion uses.
#
# Verification policy (per user request):
#   - Az/El + distances must match to numerical precision (< 1e-12 rel).
#   - Angular-diameter columns are expected to differ at ~1e-4 relative
#     because the reference uses slightly different physical body radii
#     than IAU 2015 nominal — accept anything ≤ 5e-4 relative.
#
# Run:
#   julia --project scripts/test_azel_csv_against_reference.jl

using Pkg; Pkg.activate(dirname(@__DIR__))
using Hyperion
using Dates
using Printf

const REF_CSV = joinpath(dirname(@__DIR__), "data", "inputs", "nobile_27_28_20m",
                         "other", "azimuths_elevations.csv")
const OUT_CSV = "/tmp/azel_yearlong.csv"

# Pixel that produced the reference: row=9216, col=18880 in the Shirley grid.
const PIXEL_ROW = 9216
const PIXEL_COL = 18880

# Hybrid tolerance: a value passes if |Δ| ≤ HARD_TOL_ABS OR
# |Δ|/|ref| ≤ HARD_TOL_REL. The absolute leg catches the elevation
# columns near zero (where 1e-14 abs becomes a 1e-11 relative error
# even though both implementations are at Float64 round-off).
const HARD_TOL_REL = 1e-12
const HARD_TOL_ABS = 1e-13     # ≥ Float64 epsilon at our typical scale
const SOFT_TOL_REL = 5e-4      # angular diameter (R_body convention diff)
const HARD_COLS = (
    "rover_to_sun_azimuth_deg",
    "rover_to_sun_elevation_deg",
    "rover_to_earth_azimuth_deg",
    "rover_to_earth_elevation_deg",
    "rover_to_sun_dist_km",
    "rover_to_sun_dist_au",
)
const SOFT_COLS = (
    "sun_angular_diameter_deg",
    "earth_angular_diameter_deg",
)


function shirley_pixel_to_lat_lon(row::Int, col::Int)
    e_km = (col - 15199.5) * 0.02
    n_km = (15199.5 - row) * 0.02
    rho = sqrt(n_km^2 + e_km^2)
    R = 1737.4
    u = rho / (2.0 * R); u2 = u * u; denom = 1.0 + u2
    lat = asin((u2 - 1.0) / denom)
    lon = rho > 0 ? atan(e_km, n_km) : 0.0
    return rad2deg(lat), rad2deg(lon)
end


function read_csv(path)
    lines = readlines(path)
    header = String.(strip.(split(lines[1], ',')))
    rows = []
    for line in lines[2:end]
        isempty(line) && continue
        parts = String.(strip.(split(line, ',')))
        push!(rows, Dict(zip(header, parts)))
    end
    return header, rows
end


function ts_match(a::AbstractString, b::AbstractString)
    # Both are ISO-8601 with `Z` suffix in second precision; allow any
    # equivalent representation by parsing.
    a2 = replace(a, r"Z$" => "")
    b2 = replace(b, r"Z$" => "")
    return DateTime(a2) == DateTime(b2)
end


function main()
    @info "regenerating CSV at the reference timestamps"
    Hyperion.init_spice(joinpath(dirname(@__DIR__), "kernels"))
    lat, lon = shirley_pixel_to_lat_lon(PIXEL_ROW, PIXEL_COL)
    @info "query point" pixel=(PIXEL_ROW, PIXEL_COL) lat_deg=lat lon_deg=lon

    # Use the reference's timestamp list verbatim so we don't introduce
    # drift from a re-derived range.
    ref_header, ref_rows = read_csv(REF_CSV)
    timestamps = [DateTime(replace(r["time"], r"Z$" => "")) for r in ref_rows]
    @info "reference rows" n=length(ref_rows)

    Hyperion.write_azel_csv(OUT_CSV, timestamps, lat, lon)
    @info "wrote regenerated CSV" path=OUT_CSV

    new_header, new_rows = read_csv(OUT_CSV)
    if new_header != ref_header
        @error "header mismatch" ref=ref_header got=new_header
        return
    end
    if length(new_rows) != length(ref_rows)
        @error "row count mismatch" ref=length(ref_rows) got=length(new_rows)
        return
    end

    # Per-column max relative + absolute error.
    function max_errs(col)
        mre = 0.0; mae = 0.0; argmax_idx = 0
        for i in eachindex(ref_rows)
            ref_v = parse(Float64, ref_rows[i][col])
            new_v = parse(Float64, new_rows[i][col])
            denom = max(abs(ref_v), 1e-30)
            ae = abs(new_v - ref_v)
            re = ae / denom
            if re > mre
                mre = re; argmax_idx = i
            end
            ae > mae && (mae = ae)
        end
        return mre, mae, argmax_idx
    end

    println()
    println("=== Per-column max errors vs reference ===")
    @printf "%-32s %14s %14s   %s\n" "column" "max_rel_err" "max_abs_err" "verdict"
    all_ok = true
    for col in HARD_COLS
        mre, mae, _ = max_errs(col)
        # Pass on absolute OR relative tolerance.
        ok = (mae <= HARD_TOL_ABS) || (mre <= HARD_TOL_REL)
        @printf "%-32s %14.3e %14.3e   %s\n" col mre mae (ok ? "OK" : "FAIL")
        ok || (all_ok = false)
    end
    for col in SOFT_COLS
        mre, mae, _ = max_errs(col)
        ok = mre <= SOFT_TOL_REL
        @printf "%-32s %14.3e %14.3e   %s (soft, R_body diff)\n" col mre mae (ok ? "OK" : "FAIL")
        ok || (all_ok = false)
    end

    println()
    if all_ok
        println("✓ All columns within tolerance.")
    else
        println("✗ At least one column exceeded its tolerance.")
        exit(1)
    end

    # Also show the worst row for the strictest column to confirm the
    # observed errors really are at machine epsilon.
    println()
    println("=== Worst row for each HARD column ===")
    for col in HARD_COLS
        mre, mae, idx = max_errs(col)
        ref_v = parse(Float64, ref_rows[idx][col])
        new_v = parse(Float64, new_rows[idx][col])
        @printf "  %-32s  ts=%s  ref=%.16g  api=%.16g  abs=%.3e  rel=%.3e\n" col ref_rows[idx]["time"] ref_v new_v mae mre
    end
end

main()
