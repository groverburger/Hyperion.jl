#!/usr/bin/env julia
# Compare data/outputs/big_smoke/metal/SHAs.txt vs .../cuda/SHAs.txt
# and print a side-by-side report of matching / differing entries.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Printf

const BASE = joinpath(dirname(@__DIR__), "data", "outputs", "big_smoke")
const M = joinpath(BASE, "metal", "SHAs.txt")
const C = joinpath(BASE, "cuda",  "SHAs.txt")

if !isfile(M); error("Missing $M — run big_smoke_test.jl on Mac first"); end
if !isfile(C); error("Missing $C — run big_smoke_test.jl on Windows with JM_BACKEND=cuda"); end

"Parse a SHAs.txt file into OrderedDict of (section => Dict(key => sha))."
function parse_shas(path)
    sections = Tuple{String, Dict{String, String}}[]
    cur = nothing
    for line in eachline(path)
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            continue
        elseif startswith(line, "[") && endswith(line, "]")
            cur = (line[2:end-1], Dict{String, String}())
            push!(sections, cur)
        elseif occursin("=", line) && cur !== nothing
            k, v = split(line, "=", limit=2)
            cur[2][strip(k)] = strip(v)
        end
    end
    return sections
end

m = parse_shas(M); c = parse_shas(C)

@printf("%-22s  %-6s  %-10s\n", "timestamp", "key", "status")
println("─" ^ 50)

total = 0; matching = 0
for (sec_name, m_shas) in m
    c_section = findfirst(x -> x[1] == sec_name, c)
    c_shas = c_section === nothing ? Dict{String, String}() : c[c_section][2]
    for (k, v_m) in m_shas
        total += 1
        v_c = get(c_shas, k, "")
        status = if v_c == ""
            "MISSING"
        elseif v_c == v_m
            matching += 1
            "MATCH"
        else
            "DIFFER"
        end
        @printf("%-22s  %-6s  %-10s%s\n", sec_name, k, status,
                status == "DIFFER" ? "  $(first(v_m, 10))… vs $(first(v_c, 10))…" : "")
    end
end

println()
@printf("%d / %d SHAs match\n", matching, total)
if matching == total
    println("✓ Fully bit-exact across Metal ↔ CUDA")
else
    println("✗ Cross-platform divergence")
end
