# Shared helpers for the GUI's mapset analysis tools: frame discovery and
# reading stored values from sun and DSN PNGs.

using Dates
import FileIO

const TAG_FORMAT = dateformat"yyyy-mm-ddTHH-MM-SS"

"Timestamps tags present in `dir/<kind>/<kind>.<tag>.png`, sorted."
function frame_tags(dir::AbstractString, kind::AbstractString)
    d = joinpath(dir, kind)
    isdir(d) || return String[]
    tags = String[]
    for f in readdir(d)
        startswith(f, "._") && continue
        m = match(Regex("^$(kind)\\.(.+)\\.png\$"), f)
        m === nothing || push!(tags, m[1])
    end
    return sort(tags)
end

frame_time(tag) = DateTime(tag, TAG_FORMAT)
frame_path(dir, kind, tag) = joinpath(dir, kind, "$kind.$tag.png")

"""
    frame_values(path) -> Matrix{UInt8} or nothing

Stored values of a palette PNG. Sun RGB files use a grey palette, so their red
channel is the value. DSN RGB files keep only colour bands and return nothing.
"""
function frame_values(path::AbstractString)
    img = FileIO.load(path)
    hasproperty(img, :index) && return Matrix{UInt8}(img.index)
    startswith(basename(path), "sun") || return nothing
    return [reinterpret(UInt8, c.r) for c in img]
end

function report_progress(i, n, label; every = max(1, n ÷ 100))
    (i % every == 0 || i == n) || return
    print(stdout, "\r", label, " ", round(Int, 100i / n), "% (", i, "/", n, ")")
    i == n && println(stdout)
    flush(stdout)
end

# --key=value options.
function parse_options(args)
    opts = Dict{String,String}()
    for a in args
        m = match(r"^--([a-z-]+)=(.*)$", a)
        m === nothing && error("Expected --key=value, got $a")
        opts[m[1]] = m[2]
    end
    return opts
end
