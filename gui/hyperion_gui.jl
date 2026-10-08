#!/usr/bin/env julia
# Desktop GUI for Hyperion, built on Mirage.
# Run from the repository root:
#
#   julia --project=gui -e 'using Pkg; Pkg.instantiate()'   # first time only
#   julia --project=gui gui/hyperion_gui.jl
#
# Calculations run in separate, low-priority Julia processes that call the
# scripts in scripts/ and tools/, so the window stays responsive.
module HyperionGUI

using Mirage
import CImGui
import GLFW
import FileIO
import TOML
import Hyperion
using Dates
using Printf
using ColorTypes: RGB, RGBA
using FixedPointNumbers: N0f8

const REPO = normpath(joinpath(@__DIR__, ".."))
# Scratch space for previews, edited specs, and the hash cache. Git ignores data/outputs/.
const GUI_DIR = joinpath(REPO, "data", "outputs", ".hyperion_gui")

include("src/util.jl")     # text fields, widget helpers, paths
include("src/jobs.jl")     # background Julia processes
include("src/view.jl")     # central view: map images and CSV plots
include("src/specs.jl")    # mapset specification loading and editing
include("src/run.jl")      # mapset, preview, light-curve, az/el, and probe panels
include("src/browse.jl")   # mapset browser, contact maps, comparison, statistics
include("src/layers_view.jl")  # DEM overviews, windows, and layer extents
include("src/editor.jl")   # form editor for mapset specifications
include("src/house.jl")    # input data, tests, and settings panels
include("src/app.jl")      # window layout and main loop

end # module HyperionGUI

if abspath(PROGRAM_FILE) == @__FILE__
    HyperionGUI.main()
end
