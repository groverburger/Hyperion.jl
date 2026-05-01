#!/usr/bin/env julia
# Fetches the PDS LOLA 80°S 20 m/pixel LDEM and derives the Nobile
# Float32 GTiff crop used by diagnostic scripts. Idempotent — SHA-
# verifies existing files and only downloads / derives what's missing.
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
