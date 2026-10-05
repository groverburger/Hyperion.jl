# Find external terrain inputs in data/inputs/ by SHA-256, not by filename.
# The files are not stored in Git, and this module does not download them.
# See docs/src/reference/input-data-hashes.md for the supported products.

const _LDEM_SHA   = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
const _NOBILE_1M_SHA = "e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4"
const _BARKER_2023_LDEM_SHA = "09b7ca80f9e6a146f970225d18af72fc02787669b3ef51b888e347d2b6845649"
const _VIPER8_NOBILE_CROP_SHA = "85988a26542fb2ed51322b802b5bd47467e7bd009eb67b8ab127999b8ca24e19"

# `file` is the conventional name, used in messages and as the fallback path.
# `bytes` limits hashing to files of the right size.
const _SHIRLEY_LDEM = (name = "Shirley LDEM 80S 20 m", file = "ldem_80s_20m.img",
    sha = _LDEM_SHA, bytes = 1_848_320_000,
    format = "raw 30400x30400 Int16 little-endian, scale 0.5 m, no header")
const _NOBILE_1M = (name = "Nobile 1 m site DEM", file = "nobile_1m.tif",
    sha = _NOBILE_1M_SHA, bytes = 81_822_336, format = "GeoTIFF")
const _BARKER_2023_LDEM = (name = "Barker 2023 LDEM 80S 20 m", file = "LDEM_80S_20MPP_ADJ.TIF",
    sha = _BARKER_2023_LDEM_SHA, bytes = 2_696_082_051, format = "GeoTIFF")
const _VIPER8_NOBILE_CROP = (name = "VIPER 8.0 Nobile-area crop",
    file = "nobile_area_viper_sfs_dem_8_0_native_crop.tif",
    sha = _VIPER8_NOBILE_CROP_SHA, bytes = 27_052_991, format = "GeoTIFF")

_data_dir() = joinpath(dirname(@__DIR__), "data", "inputs")

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

# SHA-256 => matched path, or nothing. Each product is searched once per session.
const _FOUND_INPUTS = Dict{String,Union{Nothing,String}}()

# Return the file in data/inputs/ whose SHA-256 matches `product`, or nothing.
# A file with the conventional name is checked first.
function _find_input(product)
    get!(_FOUND_INPUTS, product.sha) do
        dir = _data_dir()
        isdir(dir) || return nothing
        conventional = joinpath(dir, product.file)
        candidates = [joinpath(dir, f) for f in readdir(dir) if !startswith(f, "._")]
        filter!(p -> isfile(p) && filesize(p) == product.bytes, candidates)
        sort!(candidates; by = p -> p != conventional)
        for path in candidates
            if _sha256_file(path) == product.sha
                @info "Using $(product.name), matched by SHA-256" path sha256 = product.sha
                return path
            end
        end
        return nothing
    end
end

_input_path(product) = something(_find_input(product), joinpath(_data_dir(), product.file))
_input_ok(product) = _find_input(product) !== nothing

_ldem_path() = _input_path(_SHIRLEY_LDEM)
_nobile_1m_path() = _input_path(_NOBILE_1M)
_barker_2023_ldem_path() = _input_path(_BARKER_2023_LDEM)
_viper8_nobile_crop_path() = _input_path(_VIPER8_NOBILE_CROP)

_ldem_ok() = _input_ok(_SHIRLEY_LDEM)
_nobile_1m_ok() = _input_ok(_NOBILE_1M)
_barker_2023_ldem_ok() = _input_ok(_BARKER_2023_LDEM)
_viper8_nobile_crop_ok() = _input_ok(_VIPER8_NOBILE_CROP)

function _require_input!(product)
    path = _find_input(product)
    path === nothing || return path
    error("""
    $(product.name) not found.

    No file in $(_data_dir()) has the expected SHA-256:

      $(product.sha)

    Copy the file into that directory under any name. The conventional name
    is $(product.file). Expected format: $(product.format), $(product.bytes) bytes.
    See docs/src/reference/input-data-hashes.md.
    """)
end

"""
    require_shirley_ldem!() -> String

Return the path of the Shirley LDEM_80S_20M raster in `data/inputs/`,
found by SHA-256. This function never downloads data. If no file matches,
it errors with placement instructions.
"""
require_shirley_ldem!() = _require_input!(_SHIRLEY_LDEM)

"""
    require_nobile_1m_tif!() -> String

Return the path of the 1 m Nobile site DEM in `data/inputs/`, found by SHA-256.
"""
require_nobile_1m_tif!() = _require_input!(_NOBILE_1M)

"""
    require_barker_2023_ldem!() -> String

Return the path of the Barker et al. 2023 20 m south-polar LDEM GeoTIFF in
`data/inputs/`, found by SHA-256.
"""
require_barker_2023_ldem!() = _require_input!(_BARKER_2023_LDEM)

"""
    require_viper8_nobile_crop_tif!() -> String

Return the path of the native VIPER 8.0 crop covering the Nobile area in
`data/inputs/`, found by SHA-256.
"""
require_viper8_nobile_crop_tif!() = _require_input!(_VIPER8_NOBILE_CROP)
