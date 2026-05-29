#!/usr/bin/env julia
# Compare SHAs.txt across every backend directory present in
# data/outputs/bitexact/ (metal/, cuda/, cpu/). Prints the toolchain
# triplet per backend (so regressions can be localized to a version bump)
# then an all-pairs MATCH / DIFFER report.
using Pkg; Pkg.activate(dirname(dirname(@__DIR__)))
using Printf

const BASE = joinpath(dirname(dirname(@__DIR__)), "data", "outputs", "bitexact")

"Parse a SHAs.txt file into (header_lines, Vector{(section, Dict(key => sha))})."
function parse_shas(path)
    headers = String[]
    sections = Tuple{String, Dict{String, String}}[]
    cur = nothing
    for line in eachline(path)
        line = strip(line)
        if isempty(line)
            continue
        elseif startswith(line, "#")
            push!(headers, line)
        elseif startswith(line, "[") && endswith(line, "]")
            cur = (line[2:end-1], Dict{String, String}())
            push!(sections, cur)
        elseif occursin("=", line) && cur !== nothing
            k, v = split(line, "=", limit=2)
            cur[2][strip(k)] = strip(v)
        end
    end
    return headers, sections
end

# Discover which backends have SHAs.txt on disk. Auto-scans every
# subdirectory of `data/outputs/bitexact/`, so cross-machine runs that
# rename their dirs (e.g. `win_cuda/`, `win_cpu/` from a Windows + CUDA
# audit) are picked up alongside the local-machine `cpu/` / `metal/`.
backends = String[]
if isdir(BASE)
    for entry in sort(readdir(BASE))
        startswith(entry, ".") && continue
        isfile(joinpath(BASE, entry, "SHAs.txt")) && push!(backends, entry)
    end
end

if length(backends) < 2
    error("Need at least two backends with SHAs.txt under $BASE. " *
          "Found: $backends. Run bitexact_test.jl with HYP_BACKEND={cpu,metal,cuda} " *
          "or copy in cross-machine results.")
end

parsed = Dict(b => parse_shas(joinpath(BASE, b, "SHAs.txt")) for b in backends)

# ─── Toolchain report ─────────────────────────────────────────────────────
println("Backends discovered: $(join(backends, ", "))")
println()
for b in backends
    println("── $b ──")
    for h in parsed[b][1]
        println("  $h")
    end
    println()
end

# ─── All-pairs comparison ────────────────────────────────────────────────
function compare_pair(a_name, a_secs, b_name, b_secs)
    @printf("\n═══ %s ↔ %s ═══\n", a_name, b_name)
    @printf("%-22s  %-6s  %-10s\n", "timestamp", "key", "status")
    println("─" ^ 50)
    total = 0; matching = 0
    for (sec_name, a_shas) in a_secs
        b_idx = findfirst(x -> x[1] == sec_name, b_secs)
        b_shas = b_idx === nothing ? Dict{String, String}() : b_secs[b_idx][2]
        for (k, v_a) in a_shas
            total += 1
            v_b = get(b_shas, k, "")
            status = if v_b == ""
                "MISSING"
            elseif v_b == v_a
                matching += 1
                "MATCH"
            else
                "DIFFER"
            end
            if status != "MATCH"
                @printf("%-22s  %-6s  %-10s%s\n", sec_name, k, status,
                        status == "DIFFER" ? "  $(first(v_a, 10))… vs $(first(v_b, 10))…" : "")
            end
        end
    end
    @printf("\n  %d / %d SHAs match\n", matching, total)
    matching == total
end

# Pair (cpu, metal, cuda) → compare each pair.
function run_all_pairs(backends, parsed)
    ok = true
    for i in 1:length(backends), j in (i+1):length(backends)
        ok &= compare_pair(backends[i], parsed[backends[i]][2],
                           backends[j], parsed[backends[j]][2])
    end
    ok
end
all_ok = run_all_pairs(backends, parsed)

println()
if all_ok
    println("✓ Fully bit-exact across: $(join(backends, " ↔ "))")
else
    println("✗ Cross-backend divergence detected — see DIFFERs above")
end
