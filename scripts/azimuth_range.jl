#!/usr/bin/env julia
# Compute the az/el range swept by Sun + Earth at the Nobile midpoint pixel
# across June 2027 → June 2028 at 2h intervals. Used to sanity-check the
# "narrow-azimuth-band" claim.

using Pkg
Pkg.activate(dirname(@__DIR__))

using Dates, Printf
import JuliaMapbuilder as JM

const REPO    = dirname(@__DIR__)
const DEM     = joinpath(REPO, "data", "inputs", "nobile_20m.tif")
const KERNELS = joinpath(REPO, "kernels")

const F32_RAD2DEG = Float32(180.0) / Float32(π)

@info "Loading DEM"
dem = JM.load_shadow_dem(DEM)
H, W = dem.H, dem.W
mr = H ÷ 2 + 1    # midpoint row (1-indexed)
mc = W ÷ 2 + 1    # midpoint col (1-indexed)
@info "Midpoint pixel" row=mr col=mc size="$(H)×$(W)"

@info "Initializing SPICE"
JM.init_spice(KERNELS)

# Timestep grid: June 1 2027 00:00 → June 1 2028 00:00, 2h steps
start_dt = DateTime(2027, 6, 1, 0, 0, 0)
stop_dt  = DateTime(2028, 6, 1, 0, 0, 0)
step     = Dates.Hour(2)
timesteps = DateTime[]
let current = start_dt
    while current <= stop_dt
        push!(timesteps, current)
        current += step
    end
end
n = length(timesteps)
@info "Timestep grid" n=n

# Compute az/el at the midpoint for each timestep, each body
function azel_at(body_id::Int, et::Float64, dem::JM.ShadowDEM, r::Int, c::Int)
    pos = JM.get_body_position(body_id, et)
    lx = dem.R[r,c,1,1]*pos[1] + dem.R[r,c,1,2]*pos[2] + dem.R[r,c,1,3]*pos[3] + dem.T[r,c,1]
    ly = dem.R[r,c,2,1]*pos[1] + dem.R[r,c,2,2]*pos[2] + dem.R[r,c,2,3]*pos[3] + dem.T[r,c,2]
    lz = dem.R[r,c,3,1]*pos[1] + dem.R[r,c,3,2]*pos[2] + dem.R[r,c,3,3]*pos[3] + dem.T[r,c,3]
    az_rad = atan(ly, lx) + π
    el_rad = atan(lz, sqrt(lx^2 + ly^2))
    return rad2deg(az_rad), rad2deg(el_rad)
end

sun_az   = Vector{Float64}(undef, n)
sun_el   = Vector{Float64}(undef, n)
earth_az = Vector{Float64}(undef, n)
earth_el = Vector{Float64}(undef, n)

for (i, dt) in enumerate(timesteps)
    et = JM.datetime_to_et(dt)
    sun_az[i], sun_el[i]   = azel_at(10, et, dem, mr, mc)
    earth_az[i], earth_el[i] = azel_at(399, et, dem, mr, mc)
end

function summarize(label, az, el)
    az_min, az_max = minimum(az), maximum(az)
    el_min, el_max = minimum(el), maximum(el)
    az_span = az_max - az_min
    el_span = el_max - el_min
    @printf("%-8s  az: %7.2f° → %7.2f°  (span %6.2f°, %5.1f%% of circle)    el: %6.2f° → %6.2f°  (span %5.2f°)\n",
            label, az_min, az_max, az_span, 100 * az_span / 360, el_min, el_max, el_span)
    # Histogram of az over 24 bins of 15° each, to see if the sun wraps around
    bins = zeros(Int, 24)
    for a in az
        bi = clamp(div(Int(floor(a)), 15) + 1, 1, 24)
        bins[bi] += 1
    end
    non_empty = count(b -> b > 0, bins)
    @printf("          az-histogram: %d of 24 fifteen-degree bins occupied\n", non_empty)
end

println()
println("Annual sun + Earth az/el sweep at Nobile midpoint:")
summarize("SUN",   sun_az,   sun_el)
summarize("EARTH", earth_az, earth_el)

# Also: how many of the 1440 0.25°-az buckets are actually visited?
function bucket_count(az)
    buckets = Set{Int}()
    for a in az
        push!(buckets, Int(floor(a / 0.25)))
    end
    length(buckets)
end
sun_buckets   = bucket_count(sun_az)
earth_buckets = bucket_count(earth_az)
println()
@printf("SUN:   %d / 1440 az buckets visited (%.1f%%)\n", sun_buckets, 100 * sun_buckets / 1440)
@printf("EARTH: %d / 1440 az buckets visited (%.1f%%)\n", earth_buckets, 100 * earth_buckets / 1440)
println()
# Union of buckets needed for ANY body
union_buckets = Set{Int}()
for a in sun_az;   push!(union_buckets, Int(floor(a / 0.25))); end
for a in earth_az; push!(union_buckets, Int(floor(a / 0.25))); end
@printf("UNION: %d / 1440 az buckets needed (%.1f%%)\n", length(union_buckets), 100 * length(union_buckets) / 1440)
