# Per-pixel statistics over every frame of a mapset, for the GUI's mapset
# browser. Writes one Float32 raster per statistic plus stats.toml.
#
#   julia --project tools/mapset_stats.jl --dir=<mapset> --out=<folder>
#         [--sun-min=1] [--dsn-min=3] [--stride=1] [--start=<UTC>] [--stop=<UTC>]
#
# A pixel is lit when the sun value is at least --sun-min (1 to 255; 1 means
# any part of the solar disk). It has Earth contact when the DSN value is at
# least --dsn-min (tenths of a degree; 3 means 0.3°). Run lengths use frame
# times: a run lasts from its first frame to the first frame after it.

include("mapset_frames.jl")
import TOML

mutable struct RunTracker
    start::Matrix{Int32}      # frame index where the current run began; 0 if none
    longest::Matrix{Float32}  # hours
end
RunTracker(H, W) = RunTracker(zeros(Int32, H, W), zeros(Float32, H, W))

function update!(r::RunTracker, inrun::BitMatrix, k, times)
    @inbounds for i in eachindex(inrun)
        if inrun[i]
            r.start[i] == 0 && (r.start[i] = k)
        elseif r.start[i] != 0
            r.longest[i] = max(r.longest[i], hours(times[k] - times[r.start[i]]))
            r.start[i] = 0
        end
    end
end

function finish!(r::RunTracker, times, step_end)
    @inbounds for i in eachindex(r.start)
        r.start[i] == 0 && continue
        r.longest[i] = max(r.longest[i], hours(step_end - times[r.start[i]]))
    end
end

hours(p::Period) = Float32(Dates.value(Millisecond(p)) / 3_600_000)

function main(args)
    o = parse_options(args)
    dir, out = o["dir"], o["out"]
    sun_min = parse(Int, get(o, "sun-min", "1"))
    dsn_min = parse(Int, get(o, "dsn-min", "3"))
    stride = parse(Int, get(o, "stride", "1"))
    tags = intersect(frame_tags(dir, "sun"), frame_tags(dir, "dsn"))
    haskey(o, "start") && filter!(t -> frame_time(t) >= DateTime(replace(o["start"], "Z" => "")), tags)
    haskey(o, "stop") && filter!(t -> frame_time(t) <= DateTime(replace(o["stop"], "Z" => "")), tags)
    tags = tags[1:stride:end]
    length(tags) >= 2 || error("Need at least two frames with both sun and DSN images in $dir")
    times = frame_time.(tags)
    step_end = times[end] + (times[end] - times[end-1])
    println("Statistics over $(length(tags)) frames, $(times[1]) to $(times[end]); " *
            "lit: sun >= $sun_min; contact: DSN >= $dsn_min ($(dsn_min / 10)°)")

    first_dsn = frame_values(frame_path(dir, "dsn", tags[1]))
    first_dsn === nothing && error("DSN images in $dir are RGB and do not store values; statistics need palette PNGs.")
    H, W = size(first_dsn)
    lit = zeros(Int32, H, W); contact = zeros(Int32, H, W); both = zeros(Int32, H, W)
    sun_sum = zeros(Float32, H, W)
    shadow_runs, outage_runs = RunTracker(H, W), RunTracker(H, W)
    for (k, tag) in enumerate(tags)
        sun = frame_values(frame_path(dir, "sun", tag))
        dsn = k == 1 ? first_dsn : frame_values(frame_path(dir, "dsn", tag))
        (sun === nothing || dsn === nothing) && error("Frame $tag has no stored values")
        size(sun) == (H, W) == size(dsn) || error("Frame $tag has a different size")
        is_lit = sun .>= sun_min
        has_contact = dsn .>= dsn_min
        lit .+= is_lit
        contact .+= has_contact
        both .+= is_lit .& has_contact
        sun_sum .+= sun ./ 255f0
        update!(shadow_runs, .!is_lit, k, times)
        update!(outage_runs, .!has_contact, k, times)
        report_progress(k, length(tags), "Reading frames")
    end
    finish!(shadow_runs, times, step_end)
    finish!(outage_runs, times, step_end)

    n = Float32(length(tags))
    stats = [
        ("lit_percent", 100 .* lit ./ n, "%", "Frames with sun >= $sun_min"),
        ("mean_sun_percent", 100 .* sun_sum ./ n, "%", "Mean visible solar-disk fraction"),
        ("longest_shadow_hours", shadow_runs.longest, "h", "Longest continuous run with sun < $sun_min"),
        ("contact_percent", 100 .* contact ./ n, "%", "Frames with DSN >= $(dsn_min / 10)°"),
        ("longest_outage_hours", outage_runs.longest, "h", "Longest continuous run with DSN < $(dsn_min / 10)°"),
        ("lit_and_contact_percent", 100 .* both ./ n, "%", "Frames both lit and in contact"),
    ]
    mkpath(out)
    meta = Dict{String,Any}("mapset" => abspath(dir), "rows" => H, "columns" => W,
        "frames" => length(tags), "start" => string(times[1]), "stop" => string(times[end]),
        "sun_min" => sun_min, "dsn_min" => dsn_min, "stride" => stride, "statistics" => Any[])
    for (name, data, unit, description) in stats
        write(joinpath(out, name * ".f32"), Float32.(data))
        push!(meta["statistics"], Dict("name" => name, "unit" => unit, "description" => description,
                                       "file" => name * ".f32"))
    end
    open(io -> TOML.print(io, meta), joinpath(out, "stats.toml"), "w")
    println("Wrote statistics to $out")
end

main(ARGS)
