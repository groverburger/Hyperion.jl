# Read one pixel's sun and DSN values from every frame of a mapset and write
# them as a CSV time series. Used by the GUI's mapset browser.
#
#   julia --project tools/point_series.jl --dir=<mapset> --row=<r> --col=<c> --out=<csv> [--stride=1]
#
# Row and column are output-image indices starting at zero.

include("mapset_frames.jl")

function main(args)
    o = parse_options(args)
    dir, out = o["dir"], o["out"]
    row, col = parse(Int, o["row"]) + 1, parse(Int, o["col"]) + 1
    stride = parse(Int, get(o, "stride", "1"))
    tags = sort(union(frame_tags(dir, "sun"), frame_tags(dir, "dsn")))[1:stride:end]
    isempty(tags) && error("No sun or DSN frames in $dir")
    println("Reading pixel col $(col - 1), row $(row - 1) from $(length(tags)) frames of $dir")
    mkpath(dirname(abspath(out)))
    tmp = out * ".partial"
    open(tmp, "w") do io
        println(io, "time_utc,sun_u8,sun_percent,dsn_u8,dsn_deg")
        for (i, tag) in enumerate(tags)
            fields = String[string(frame_time(tag)) * "Z"]
            for kind in ("sun", "dsn")
                p = frame_path(dir, kind, tag)
                v = isfile(p) ? frame_values(p) : nothing
                if v === nothing
                    append!(fields, ["", ""])
                else
                    x = Int(v[row, col])
                    push!(fields, string(x), kind == "sun" ? string(round(100x / 255; digits = 2)) : string(x / 10))
                end
            end
            println(io, join(fields, ','))
            report_progress(i, length(tags), "Reading frames")
        end
    end
    mv(tmp, out; force = true)
    println("Wrote $out")
end

main(ARGS)
