#!/usr/bin/env julia
using Hyperion
isempty(ARGS) && error("Usage: julia --project=. scripts/filter_psr.jl path/to/psr.tif [...]")
for path in ARGS
    filter_psr(path)
end
