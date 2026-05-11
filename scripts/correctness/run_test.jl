#!/usr/bin/env julia
# Convenience runner for the Tier 0 correctness regression test.
# Invokes `test/correctness.jl` directly (not through `Pkg.test()`)
# so that the GPU backend (Metal / CUDA) loaded in the user's
# environment is visible. The test refuses to run on CPU because
# each LNSI render takes ~200 s and the 25-NAC sweep would be ~85 min.
#
# Run:
#   HYP_RUN_CORRECTNESS=1 julia --project scripts/correctness/run_test.jl
#
# Backend override (defaults to first of metal / cuda available):
#   HYP_BACKEND=cuda HYP_RUN_CORRECTNESS=1 julia --project ...

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))

# Mirror the bootstrapping that `test/runtests.jl` does so
# correctness.jl finds the constants it expects.
using Test
using Hyperion
const Hyp = Hyperion
import SHA
using Dates

const PROJECT_ROOT = joinpath(@__DIR__, "..", "..")

const LDEM_PATH = begin
    override = get(ENV, "HYP_LDEM_PATH", "")
    if !isempty(override)
        isfile(override) || error("HYP_LDEM_PATH does not point to a file: $override")
        abspath(override)
    else
        try
            Hyp.require_shirley_ldem!()
        catch e
            @warn "Could not locate Shirley LDEM" exception=e
            ""
        end
    end
end
const HAS_LDEM = !isempty(LDEM_PATH) && isfile(LDEM_PATH)

# Force the gate on so `runtests.jl`-style scripts don't have to.
ENV["HYP_RUN_CORRECTNESS"] = "1"

include(joinpath(PROJECT_ROOT, "test", "correctness.jl"))
