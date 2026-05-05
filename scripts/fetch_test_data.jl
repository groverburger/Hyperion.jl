#!/usr/bin/env julia
# Fetches the project's baseline 80°S 20 m/pixel LDEM (the "Shirley"
# legacy artefact, SHA `caaf017f…`; see README.md → Farfield LDEM
# versions for the disambiguation table) and derives the Nobile
# Float32 GTiff crop used by diagnostic scripts. Idempotent — SHA-
# verifies existing files and only downloads / derives what's missing.
# Note: a fresh fetch from the PDS Geosciences Node @ WUSTL URL would
# serve a different version of the LDEM than Shirley (different SHA),
# and `ensure_ldem!()` would fail its SHA check — see the header of
# `src/test_data.jl` for context.
#
# Usage:
#     julia --project scripts/fetch_test_data.jl
#
# You usually don't need to run this explicitly — `julia --project -e
# 'using Pkg; Pkg.test()'` auto-provisions the LDEM before the first
# test. This script exists for users who want to trigger setup
# separately from testing (e.g. to pre-seed data on a workstation, or
# to produce `data/inputs/nobile_20m.tif` for diagnostic scripts like
# scripts/azimuth_range.jl).

using Pkg
Pkg.activate(dirname(@__DIR__))
import Hyperion as Hyp

Hyp.ensure_test_data!()
@info "Test data ready"
