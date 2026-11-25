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
# Scratch space for edited settings, overviews, and the hash cache. Git ignores data/outputs/.
const GUI_DIR = joinpath(REPO, "data", "outputs", ".hyperion_gui")

include("src/util.jl")     # text fields, widget helpers, paths
include("src/jobs.jl")     # background Julia processes
include("src/view.jl")     # central view: images and CSV plots
include("src/horizon.jl")  # horizon panorama with the Sun's path
include("src/export.jl")   # save the shown image as GeoTIFF or PNG
include("src/specs.jl")    # presets: loading and saving settings
include("src/run.jl")      # rendering and the point tools
include("src/browse.jl")   # results browser, contact maps, comparison, statistics
include("src/layers_view.jl")  # terrain overviews, image area, and the marker
include("src/marker.jl")   # marker position by coordinates
include("src/editor.jl")   # controls for the settings
include("src/render_tab.jl")  # Render tab: preset, area, times, and saving images
include("src/psr_tab.jl")  # PSR tab: permanently shadowed regions
include("src/house.jl")    # input data, tests, and settings panels
include("src/history.jl")  # undo and redo of the settings and the marker
include("src/app.jl")      # window layout and main loop

end # module HyperionGUI

if abspath(PROGRAM_FILE) == @__FILE__
    HyperionGUI.main()
end
