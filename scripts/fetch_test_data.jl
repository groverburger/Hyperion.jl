#!/usr/bin/env julia
# Validates the project's baseline 80°S 20 m/pixel LDEM (the "Shirley"
# legacy artefact, SHA `caaf017f…`; see README.md → Farfield LDEM
# versions for the disambiguation table) and derives the Nobile
# Float32 GTiff crop used by diagnostic scripts. This script never
# downloads data; it errors with placement instructions if Shirley is
# missing or has the wrong SHA.
#
# Usage:
#     julia --project scripts/fetch_test_data.jl
#
# Use this after placing Shirley at `data/inputs/ldem_80s_20m.img`
# to verify the SHA and produce `data/inputs/nobile_20m.tif` for
# local validation and fixture-generation tools.

using Pkg
Pkg.activate(dirname(@__DIR__))
import Hyperion as Hyp

Hyp.require_test_data!()
@info "Test data ready"
