#!/usr/bin/env julia
# Scan 2026 at 6-hour cadence, pick a representative set of timestamps
# covering diverse sun/earth geometry at Nobile.

using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf
import JuliaMapbuilder as JM

const KERNELS = joinpath(dirname(@__DIR__), "kernels")

@info "Loading"
_, nobile_path = JM.ensure_test_data!()
dem = JM.load_shadow_dem(nobile_path)
H, W = dem.H, dem.W
mr, mc = H ÷ 2 + 1, W ÷ 2 + 1

JM.init_spice(KERNELS)

function azel_at(body_id, et)
    pos = JM.get_body_position(body_id, et)
    r = mr; c = mc
    lx = dem.R[r,c,1,1]*pos[1] + dem.R[r,c,1,2]*pos[2] + dem.R[r,c,1,3]*pos[3] + dem.T[r,c,1]
    ly = dem.R[r,c,2,1]*pos[1] + dem.R[r,c,2,2]*pos[2] + dem.R[r,c,2,3]*pos[3] + dem.T[r,c,2]
    lz = dem.R[r,c,3,1]*pos[1] + dem.R[r,c,3,2]*pos[2] + dem.R[r,c,3,3]*pos[3] + dem.T[r,c,3]
    az = rad2deg(atan(ly, lx) + π)
    el = rad2deg(atan(lz, sqrt(lx^2 + ly^2)))
    return (az, el)
end

samples = Tuple{DateTime,Float64,Float64,Float64,Float64}[]
let dt = DateTime(2026, 1, 1), stop = DateTime(2026, 12, 31, 23, 0, 0)
    while dt <= stop
        et = JM.datetime_to_et(dt)
        sun_az, sun_el = azel_at(JM.NAIF_SUN, et)
        earth_az, earth_el = azel_at(JM.NAIF_EARTH, et)
        push!(samples, (dt, sun_az, sun_el, earth_az, earth_el))
        dt += Dates.Hour(6)
    end
end

@info "Scanned" n=length(samples)
@printf("\nSun el range: %.2f to %.2f\n",
        minimum(s[3] for s in samples), maximum(s[3] for s in samples))
@printf("Earth el range: %.2f to %.2f\n",
        minimum(s[5] for s in samples), maximum(s[5] for s in samples))

# Pick representative scenarios
function find_sample(predicate; fallback_msg="")
    hit = findfirst(s -> predicate(s...), samples)
    if hit === nothing
        @warn "No sample matches" fallback_msg; return nothing
    end
    return samples[hit]
end

println("\n== Selected representative timestamps ==")
scenarios = [
    ("sun lit + earth lit",
     (_, sa, se, ea, ee) -> se > 2.0  && ee > 2.0),
    ("sun lit + earth occluded",
     (_, sa, se, ea, ee) -> se > 2.0  && ee < -2.0),
    ("sun near terminator + earth lit",
     (_, sa, se, ea, ee) -> abs(se) < 0.5 && ee > 2.0),
    ("sun shadowed + earth lit",
     (_, sa, se, ea, ee) -> se < -2.0 && ee > 2.0),
    ("sun shadowed + earth occluded",
     (_, sa, se, ea, ee) -> se < -2.0 && ee < -2.0),
    ("both near terminator",
     (_, sa, se, ea, ee) -> abs(se) < 0.5 && abs(ee) < 0.5),
    ("sun high + earth near horizon",
     (_, sa, se, ea, ee) -> se > 4.5 && abs(ee) < 1.0),
    ("sun very high (summer peak)",
     (_, sa, se, ea, ee) -> se > 5.5),
]

selected = DateTime[]
for (name, pred) in scenarios
    s = find_sample(pred; fallback_msg=name)
    if s !== nothing
        @printf("  %-40s  %s  sun(az=%.1f° el=%+6.2f°)  earth(az=%.1f° el=%+6.2f°)\n",
                name, s[1], s[2], s[3], s[4], s[5])
        push!(selected, s[1])
    end
end

# Write out to file for the test script
outpath = joinpath(@__DIR__, "test_timestamps.txt")
open(outpath, "w") do f
    for dt in unique(selected)
        println(f, dt)
    end
end
@printf("\nWrote %d unique timestamps to %s\n", length(unique(selected)), outpath)
