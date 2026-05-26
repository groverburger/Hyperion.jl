#!/usr/bin/env julia
# Generate a CSV of Sun and Earth azimuths + elevations + distances +
# angular diameters at a fixed query point (rover) on the lunar
# surface. Two modes:
#
#   range — render at every step of a [start, stop, step] interval.
#           Step is parsed by Dates.parse(Period, …) ("01:00:00", "30s",
#           "PT2H", etc.). End-inclusive.
#
#   list  — render at every timestamp listed in an external CSV. The
#           input CSV must have a `time` or `Start Time` (or first)
#           column with parseable timestamps. (Format autodetected:
#           ISO-8601 with optional sub-seconds, with or without `Z`.)
#
# Query point can be specified two ways:
#   --lat / --lon                               in decimal degrees
#   --pixel <row> <col> [--pixel-grid shirley]  by LDEM pixel offset
#                                               (Shirley grid: S0=L0=15199.5,
#                                               20m, R=1737.4 km)
#
# Examples:
#   # Reference: 1-year hourly run, query at the LDEM pixel (9216, 18880).
#   julia --project scripts/generate_azel_csv.jl \\
#       --pixel 9216 18880 \\
#       --start 2027-06-01T00:00:00 --stop 2028-06-01T00:00:00 --step 01:00:00 \\
#       --out /tmp/azel_yearlong.csv
#
#   # NAC-list mode using LROC capture-time CSV.
#   julia --project scripts/generate_azel_csv.jl \\
#       --pixel 9216 18880 \\
#       --list path/to/timestamps.csv \\
#       --out path/to/nac_azel.csv

using Pkg; Pkg.activate(dirname(@__DIR__))
using Hyperion
using Dates
using Printf


# ─── LDEM pixel → (lat, lon) ──────────────────────────────────────────
const LDEM_S0_F64       = 15199.5
const LDEM_L0_F64       = 15199.5
const LDEM_PIX_KM_F64   = 0.02
const LDEM_R_KM_F64     = 1737.4

function shirley_pixel_to_lat_lon(row::Int, col::Int)
    e_km = (col - LDEM_S0_F64) * LDEM_PIX_KM_F64
    n_km = (LDEM_L0_F64 - row) * LDEM_PIX_KM_F64
    rho_km = sqrt(n_km^2 + e_km^2)
    u = rho_km / (2.0 * LDEM_R_KM_F64)
    u2 = u * u
    denom = 1.0 + u2
    lat = asin((u2 - 1.0) / denom)
    lon = rho_km > 0 ? atan(e_km, n_km) : 0.0
    return rad2deg(lat), rad2deg(lon)
end


# ─── CLI ──────────────────────────────────────────────────────────────
function usage()
    print("""
Usage: julia --project scripts/generate_azel_csv.jl [options]

Required (one of):
  --start <ISO> --stop <ISO> --step <duration>
                              Render at every step in [start, stop]
                              inclusive. Duration is HH:MM:SS or
                              "<n>s/m/h/d".
  --list <path>               Render at every timestamp in the listed
                              CSV file. The CSV must contain a column
                              named `time` or `Start Time`, or have
                              the timestamp in the first column.

Required (one of):
  --lat <deg> --lon <deg>     Query point (lat, lon) in decimal deg.
  --pixel <row> <col>         Query at the Shirley-grid pixel offset
                              (LDEM 80°S 20m).

Optional:
  --elev <m>                  Query elevation above the lunar
                              R = 1737.4 km sphere, in metres.
                              Default 0.
  --kernels <dir>             SPICE kernel directory. Default
                              <project_root>/kernels.
  --out <path>                Output CSV path. Default stdout.
  -h, --help                  This message.
""")
end


function parse_step(s::AbstractString)
    # HH:MM:SS form. Returns the smallest single Period type that
    # represents the interval cleanly, since Julia's `start:step:stop`
    # for DateTime doesn't accept CompoundPeriod.
    m = match(r"^(\d{1,2}):(\d{2}):(\d{2})$", s)
    if m !== nothing
        h, mn, sec = parse.(Int, m.captures)
        total_sec = h * 3600 + mn * 60 + sec
        total_sec == 0 && error("--step is zero")
        if total_sec % 3600 == 0
            return Hour(total_sec ÷ 3600)
        elseif total_sec % 60 == 0
            return Minute(total_sec ÷ 60)
        else
            return Second(total_sec)
        end
    end
    # <n>s / <n>m / <n>h / <n>d form
    m = match(r"^(\d+)([smhd])$", s)
    if m !== nothing
        n = parse(Int, m.captures[1])
        unit = m.captures[2]
        return unit == "s" ? Second(n) :
               unit == "m" ? Minute(n) :
               unit == "h" ? Hour(n)   : Day(n)
    end
    error("--step '$s' not parseable. Use HH:MM:SS or <n>{s,m,h,d}.")
end


function parse_iso_dt(s::AbstractString)
    s = strip(String(s))
    s = replace(s, r"Z$" => "")
    # Try with sub-seconds first, then without.
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS.s",
                dateformat"yyyy-mm-ddTHH:MM:SS",
                dateformat"yyyy-mm-dd HH:MM:SS.s",
                dateformat"yyyy-mm-dd HH:MM:SS")
        try
            return DateTime(s, fmt)
        catch
        end
    end
    error("Could not parse ISO timestamp: $s")
end


function read_list_csv(path::AbstractString)
    open(path) do f
        # Pull the header.  Strip a leading "# " if present
        # (lroc-nac-maps timestamps.csv writes "# Product ID,Start Time").
        header_line = readline(f)
        cleaned = lstrip(header_line, ['#', ' '])
        cols = String.(split(cleaned, ','))
        cols = strip.(cols)
        # Find the time column.
        idx = findfirst(c -> lowercase(c) in ("time", "start time"), cols)
        if idx === nothing
            idx = 1   # fall back to column 0
        end
        ts = DateTime[]
        for line in eachline(f)
            isempty(line) && continue
            parts = split(line, ',')
            length(parts) < idx && continue
            push!(ts, parse_iso_dt(parts[idx]))
        end
        return ts
    end
end


function main()
    args = ARGS
    if isempty(args) || any(a -> a in ("-h", "--help"), args)
        usage(); return
    end

    # Defaults.
    project_root = dirname(@__DIR__)
    kernels      = joinpath(project_root, "kernels")
    out_path     = nothing
    lat = nothing; lon = nothing
    elev_m = 0.0
    start_s = nothing; stop_s = nothing; step_s = nothing
    list_path = nothing

    # Manual flag parser.
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--start"
            start_s = args[i+1]; i += 2
        elseif a == "--stop"
            stop_s = args[i+1]; i += 2
        elseif a == "--step"
            step_s = args[i+1]; i += 2
        elseif a == "--list"
            list_path = args[i+1]; i += 2
        elseif a == "--lat"
            lat = parse(Float64, args[i+1]); i += 2
        elseif a == "--lon"
            lon = parse(Float64, args[i+1]); i += 2
        elseif a == "--pixel"
            row = parse(Int, args[i+1])
            col = parse(Int, args[i+2])
            lat, lon = shirley_pixel_to_lat_lon(row, col)
            i += 3
        elseif a == "--elev"
            elev_m = parse(Float64, args[i+1]); i += 2
        elseif a == "--kernels"
            kernels = args[i+1]; i += 2
        elseif a == "--out"
            out_path = args[i+1]; i += 2
        else
            error("Unknown argument: $a (use --help)")
        end
    end

    (lat === nothing || lon === nothing) && error("Specify --lat/--lon or --pixel.")
    (start_s !== nothing && list_path !== nothing) &&
        error("Specify either --start/--stop/--step OR --list, not both.")

    # Build the timestamp list.
    timestamps = if list_path !== nothing
        ts = read_list_csv(list_path)
        @info "loaded timestamp list" path=list_path n=length(ts)
        ts
    else
        (start_s === nothing || stop_s === nothing || step_s === nothing) &&
            error("Range mode requires --start, --stop, --step.")
        s = parse_iso_dt(start_s)
        e = parse_iso_dt(stop_s)
        step = parse_step(step_s)
        ts = collect(s:step:e)
        @info "built timestamp range" start=s stop=e step=step n=length(ts)
        ts
    end

    # Init SPICE.
    Hyperion.init_spice(kernels)

    @info "query point" lat_deg=lat lon_deg=lon elev_m=elev_m

    if out_path === nothing
        Hyperion.write_azel_csv(stdout, timestamps, lat, lon;
                                query_elev_m = elev_m)
    else
        Hyperion.write_azel_csv(out_path, timestamps, lat, lon;
                                query_elev_m = elev_m)
        @info "wrote CSV" path=out_path rows=length(timestamps)
    end
end

main()
