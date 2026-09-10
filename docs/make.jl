push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using Hyperion

makedocs(;
    sitename = "Hyperion.jl",
    modules = [Hyperion],
    warnonly = [:missing_docs],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Mapsets" => "mapsets.md",
        "Azimuth and elevation CSV" => "azel.md",
        "Tests" => "testing.md",
        "Terms" => "terms.md",
        "Correctness" => "correctness.md",
        "Bit-exactness" => "bitexact.md",
        "Data I/O" => "data.md",
        "Raycasting" => "raycasting.md",
        "Tooling" => "tooling.md",
        "Reference Notes" => [
            "Remote and working comparison 2026-09-10" => "reference/working-tree-validation-2026-09-10.md",
            "Algorithms" => "reference/algorithms.md",
            "Terrain Stack Kernel" => "reference/terrain-stack-kernel.md",
            "Cross-vendor Determinism" => "reference/cross-vendor-determinism.md",
            "Cross-vendor Verification 2026-05-01" => "reference/cross-vendor-verification-2026-05-01.md",
            "Input Data Hashes" => "reference/input-data-hashes.md",
            "VIPER/Shirley Investigation Notes" => "reference/viper8-shirley-investigation-notes.md",
            "Shadowed Hilltop Investigation 2026-07-21" => "reference/shadowed-hilltop-investigation-2026-07-21.md",
        ],
    ],
)
