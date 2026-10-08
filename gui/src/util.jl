# Small helpers shared by the panels.

# A fixed-capacity, NUL-terminated buffer for ImGui text inputs.
mutable struct TextField
    buf::Vector{UInt8}
end

function TextField(s::AbstractString = ""; capacity::Integer = 512)
    f = TextField(zeros(UInt8, max(capacity, ncodeunits(s) + 1)))
    return set!(f, s)
end

function set!(f::TextField, s::AbstractString)
    n = ncodeunits(s)
    n + 1 > length(f.buf) && resize!(f.buf, 2 * (n + 1))
    fill!(f.buf, 0x00)
    copyto!(f.buf, codeunits(s))
    return f
end

function value(f::TextField)
    n = something(findfirst(iszero, f.buf), length(f.buf) + 1) - 1
    return String(f.buf[1:n])
end

input!(label, f::TextField; hint = "", flags = 0) = isempty(hint) ?
    CImGui.InputText(label, f.buf, length(f.buf), flags) :
    CImGui.InputTextWithHint(label, hint, f.buf, length(f.buf), flags)

# Per-panel widget state, created on first use, so panels need no struct fields.
struct Fields
    text::Dict{Symbol,TextField}
    bool::Dict{Symbol,Base.RefValue{Bool}}
    int::Dict{Symbol,Base.RefValue{Int32}}
end
Fields() = Fields(Dict(), Dict(), Dict())

tf(F::Fields, k::Symbol, default = ""; capacity = 512) =
    get!(() -> TextField(default; capacity), F.text, k)
bref(F::Fields, k::Symbol, default = false) = get!(() -> Ref(default), F.bool, k)
iref(F::Fields, k::Symbol, default = 0) = get!(() -> Ref(Int32(default)), F.int, k)
str(F::Fields, k::Symbol) = strip(value(tf(F, k)))

# Labelled text input on one line: "label [.........]".
function field!(F::Fields, label, k::Symbol; default = "", hint = "", width = -1)
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted(label)
    CImGui.SameLine(150)
    CImGui.SetNextItemWidth(width)
    return input!("##$k", tf(F, k, default); hint)
end

function combo!(label, idx::Base.RefValue{Int32}, items::Vector{String})
    isempty(items) && return false
    idx[] = clamp(idx[], 0, length(items) - 1)
    return CImGui.Combo(label, idx, items)
end

function help(s::AbstractString)
    CImGui.SameLine()
    CImGui.TextDisabled("(?)")
    if CImGui.IsItemHovered()
        CImGui.BeginTooltip()
        CImGui.PushTextWrapPos(CImGui.GetFontSize() * 30)
        CImGui.TextUnformatted(s)
        CImGui.PopTextWrapPos()
        CImGui.EndTooltip()
    end
end

const GOOD = (0.45f0, 0.85f0, 0.45f0, 1f0)
const BAD = (1f0, 0.45f0, 0.4f0, 1f0)
const WARN = (1f0, 0.8f0, 0.3f0, 1f0)
const DIM = (0.6f0, 0.6f0, 0.6f0, 1f0)
colored(col, s) = CImGui.TextColored(col, replace(s, "%" => "%%"))
note(s) = (CImGui.PushStyleColor(CImGui.ImGuiCol_Text, DIM);
           CImGui.TextWrapped(replace(s, "%" => "%%"));
           CImGui.PopStyleColor())

# ImGui packs draw-list colours as 0xAABBGGRR.
col32(r, g, b, a = 255) = UInt32(r) | UInt32(g) << 8 | UInt32(b) << 16 | UInt32(a) << 24

projpath(p::AbstractString) = isabspath(p) ? String(p) : normpath(joinpath(REPO, p))
function shortpath(p::AbstractString)
    r = relpath(p, REPO)
    return startswith(r, "..") ? String(p) : r
end

function human_bytes(n::Integer)
    n < 1024 && return "$n B"
    for (unit, size) in (("GB", 1024^3), ("MB", 1024^2), ("KB", 1024))
        n >= size && return @sprintf("%.1f %s", n / size, unit)
    end
end

function human_duration(seconds::Real)
    s = round(Int, seconds)
    s < 60 && return "$(s)s"
    s < 3600 && return "$(s ÷ 60)m $(lpad(s % 60, 2, '0'))s"
    return "$(s ÷ 3600)h $(lpad((s % 3600) ÷ 60, 2, '0'))m"
end

function parse_utc(s::AbstractString)
    cleaned = replace(strip(s), r"Z$" => "")
    t = tryparse(DateTime, cleaned)
    t === nothing || return t
    try
        return DateTime(cleaned, dateformat"yyyy-mm-ddTHH-MM-SS")
    catch
        return nothing
    end
end

safe_name(s) = replace(strip(s), r"[^A-Za-z0-9_.-]+" => "_")
