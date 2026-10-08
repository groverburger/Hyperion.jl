# Housekeeping panels: input data, tests, settings, and the job list.

# ─── Input data ───────────────────────────────────────────────────────────

const INPUT_DIR = joinpath(REPO, "data", "inputs")
const HASH_CACHE = joinpath(GUI_DIR, "hashes.toml")

mutable struct InputsState
    files::Vector{NamedTuple{(:path, :bytes, :mtime),Tuple{String,Int,Float64}}}
    hashes::Dict{String,NamedTuple{(:sha256, :bytes, :mtime),Tuple{String,Int,Float64}}}
    subdirs::Base.RefValue{Bool}
    scanned::Bool
end
InputsState() = InputsState([], Dict(), Ref(false), false)

# Products that Hyperion's tests and require_* functions find by SHA-256.
known_products() = [
    (Hyperion._SHIRLEY_LDEM, "tests: mipmaps, bit-exact, correctness, 1 m bit-exact"),
    (Hyperion._NOBILE_1M, "tests: 1 m bit-exact"),
    (Hyperion._BARKER_2023_LDEM, ""),
    (Hyperion._VIPER8_NOBILE_CROP, ""),
]

function load_hash_cache!(s::InputsState)
    isfile(HASH_CACHE) || return
    for (path, e) in TOML.parsefile(HASH_CACHE)
        s.hashes[path] = (sha256 = e["sha256"], bytes = Int(e["bytes"]), mtime = Float64(e["mtime"]))
    end
end

function save_hash_cache(s::InputsState)
    mkpath(GUI_DIR)
    d = Dict(p => Dict("sha256" => h.sha256, "bytes" => h.bytes, "mtime" => h.mtime) for (p, h) in s.hashes)
    open(io -> TOML.print(io, d), HASH_CACHE, "w")
end

function scan_inputs!(s::InputsState)
    files = []
    walk(dir, depth) = for f in readdir(dir)
        startswith(f, ".") && continue
        p = joinpath(dir, f)
        if isdir(p)
            depth == 0 && s.subdirs[] && f != "mapsets" && walk(p, 1)
        elseif isfile(p)
            push!(files, (path = p, bytes = Int(filesize(p)), mtime = mtime(p)))
        end
    end
    isdir(INPUT_DIR) && walk(INPUT_DIR, 0)
    s.files = files
    s.scanned = true
end

function cached_hash(s::InputsState, f)
    h = get(s.hashes, f.path, nothing)
    (h === nothing || h.bytes != f.bytes || h.mtime != f.mtime) && return nothing
    return h.sha256
end

# Which specs refer to each SHA-256 or path.
function spec_references()
    refs = Dict{String,Vector{String}}()
    for spec in list_specs()
        cfg = TOML.tryparse(read(spec, String))
        cfg isa Dict || continue
        for l in get(cfg, "layers", Any[])
            for key in (lowercase(String(get(l, "sha256", ""))), projpath(String(get(l, "path", ""))))
                isempty(key) || push!(get!(refs, key, String[]), basename(spec))
            end
        end
    end
    return Dict(k => unique(v) for (k, v) in refs)
end

function hash_command(settings::Settings, paths)
    tool = Sys.which("shasum") !== nothing ? `shasum -a 256` :
           Sys.which("sha256sum") !== nothing ? `sha256sum` : nothing
    tool === nothing && return nothing
    cmd = `$tool $paths`
    nice = Sys.which("nice")
    nice === nothing || (cmd = `$nice -n $(settings.niceness[]) $cmd`)
    return cmd
end

function inputs_panel!(app, st)
    s = st.inputs
    s.scanned || (scan_inputs!(s); st.spec_refs = spec_references())
    note("Files in data/inputs/. Hyperion's tests find terrain by SHA-256 regardless of file name; " *
         "mapset specs use the paths in their TOML. Hashes are cached in data/outputs/.hyperion_gui/.")
    if CImGui.Button("Rescan")
        scan_inputs!(s)
        st.spec_refs = spec_references()
    end
    CImGui.SameLine()
    CImGui.Checkbox("Include subfolders", s.subdirs) && scan_inputs!(s)
    unhashed = [f for f in s.files if cached_hash(s, f) === nothing]
    CImGui.SameLine()
    total = sum((f.bytes for f in unhashed); init = 0)
    if job_button("Hash $(length(unhashed)) file(s), $(human_bytes(total))"; disabled = isempty(unhashed))
        cmd = hash_command(st.settings, [f.path for f in unhashed])
        if cmd === nothing
            st.message = "Neither shasum nor sha256sum is available."
        else
            job = start_job!(app, "Hash input files", cmd; on_done = j -> begin
                for line in j.lines
                    m = match(r"^([0-9a-f]{64})\s+\*?(.+)$", line)
                    m === nothing && continue
                    p = String(m[2])
                    isfile(p) && (s.hashes[p] = (sha256 = String(m[1]), bytes = Int(filesize(p)), mtime = mtime(p)))
                end
                save_hash_cache(s)
            end)
            st.selected_job = job.id
        end
    end

    products = known_products()
    by_sha = Dict(p.sha => (p, use) for (p, use) in products)
    found = Set{String}()
    sha_count = Dict{String,Int}()
    for f in s.files
        h = cached_hash(s, f)
        h === nothing || (sha_count[h] = get(sha_count, h, 0) + 1)
    end
    flags = CImGui.ImGuiTableFlags_Borders | CImGui.ImGuiTableFlags_RowBg |
            CImGui.ImGuiTableFlags_SizingFixedFit | CImGui.ImGuiTableFlags_ScrollX | CImGui.ImGuiTableFlags_ScrollY
    if CImGui.BeginTable("inputs", 5, flags, CImGui.ImVec2(0, CImGui.GetTextLineHeightWithSpacing() * 16))
        CImGui.TableSetupScrollFreeze(0, 1)
        for h in ("file", "size", "SHA-256", "recognised as", "used by")
            CImGui.TableSetupColumn(h)
        end
        CImGui.TableHeadersRow()
        for f in s.files
            h = cached_hash(s, f)
            CImGui.TableNextRow()
            CImGui.TableNextColumn(); CImGui.TextUnformatted(relpath(f.path, INPUT_DIR))
            CImGui.TableNextColumn(); CImGui.TextUnformatted(human_bytes(f.bytes))
            CImGui.TableNextColumn()
            if h === nothing
                colored(DIM, "not hashed")
            else
                CImGui.TextUnformatted(h[1:12] * "…")
                CImGui.IsItemHovered() && (CImGui.BeginTooltip(); CImGui.TextUnformatted(h); CImGui.EndTooltip())
                sha_count[h] > 1 && (CImGui.SameLine(); colored(WARN, "duplicate"))
            end
            CImGui.TableNextColumn()
            uses = String[]
            if h !== nothing && haskey(by_sha, h)
                p, use = by_sha[h]
                push!(found, h)
                colored(GOOD, p.name)
                isempty(use) || push!(uses, use)
            else
                CImGui.TextUnformatted("")
            end
            CImGui.TableNextColumn()
            specs = unique(vcat(get(st.spec_refs, something(h, "-"), String[]), get(st.spec_refs, f.path, String[])))
            isempty(specs) || push!(uses, "specs: " * join(specs, ", "))
            CImGui.TextUnformatted(join(uses, "; "))
        end
        CImGui.EndTable()
    end
    missing_products = [(p, use) for (p, use) in products if !(p.sha in found)]
    if !isempty(missing_products)
        CImGui.SeparatorText("Not found among hashed files")
        for (p, use) in missing_products
            CImGui.Bullet()
            CImGui.TextWrapped(replace("$(p.name) (usual name $(p.file), $(human_bytes(p.bytes)))" *
                               (isempty(use) ? "" : "; needed by $use"), "%" => "%%"))
        end
        isempty(unhashed) || note("Hash the remaining files to check them against these products.")
    end
end

# ─── Tests ────────────────────────────────────────────────────────────────

const TEST_NOTES = Dict(
    "png_output.jl" => "palette PNG round trip; under a second, no data",
    "terrain_stack.jl" => "synthetic terrain renderer checks; no data",
    "light_curve.jl" => "light-curve command on synthetic terrain; needs SPICE kernels",
    "mapset_workers.jl" => "mapset command on synthetic terrain; starts two extra Julia processes; needs gdaldem",
    "bitexact.jl" => "full-frame regression; needs the Shirley LDEM (and nobile_1m.tif); heavy",
    "correctness.jl" => "25 NAC comparisons; needs the Shirley LDEM and a GPU; heavy",
)

test_files() = sort([f for f in readdir(joinpath(REPO, "test"))
                     if endswith(f, ".jl") && !startswith(f, "._") && !(f in ("runtests.jl", "test_backend.jl"))])

# Defines the globals that runtests.jl normally provides, then runs one file.
test_prelude(file) = """
    using Test, Hyperion, Dates
    import SHA
    const Hyp = Hyperion
    const PROJECT_ROOT = $(repr(REPO))
    const LDEM_PATH = Hyp._ldem_ok() ? Hyp._ldem_path() : ""
    const HAS_LDEM = !isempty(LDEM_PATH) && isfile(LDEM_PATH)
    sha256_bytes(v::AbstractArray) = bytes2hex(SHA.sha256(collect(reinterpret(UInt8, vec(v)))))
    include(joinpath(PROJECT_ROOT, "test", "test_backend.jl"))
    @testset $(repr(file)) begin
        include(joinpath(PROJECT_ROOT, "test", $(repr(file))))
    end
    """

function tests_panel!(app, st)
    F = st.fields
    note("Runs the selected test files one after another, each in its own process with the " *
         "thread and priority limits from Settings. A failure stops the sequence.")
    files = test_files()
    for f in files
        CImGui.Checkbox(f, bref(F, Symbol("test_" * f), f == "png_output.jl"))
        CImGui.SameLine()
        note(get(TEST_NOTES, f, ""))
    end
    chosen = [f for f in files if bref(F, Symbol("test_" * f))[]]
    backend = backend_combo!(F, :test_backend, st.settings)
    if job_button("Run $(length(chosen)) selected"; disabled = isempty(chosen) || any(j -> startswith(j.title, "Test ") && running(j), JOBS))
        items = [("Test " * f, julia_job(st.settings, "-e", test_prelude(f); env = ("HYP_BACKEND" => backend,)))
                 for f in chosen]
        start_sequence!(app, items; on_each = j -> (st.selected_job = j.id))
        st.selected_job = length(JOBS)
    end
    CImGui.SeparatorText("Full suite")
    CImGui.Checkbox("Skip the correctness comparison (HYP_SKIP_CORRECTNESS=1)", bref(F, :test_skip, true))
    if job_button("Run Pkg.test()")
        code = "using Pkg; Pkg.test(; julia_args = [\"--threads=$(st.settings.threads[])\"])"
        env = ("HYP_BACKEND" => backend, "HYP_SKIP_CORRECTNESS" => bref(F, :test_skip)[] ? "1" : "0")
        job = start_job!(app, "Full test suite", julia_job(st.settings, "-e", code; env))
        st.selected_job = job.id
    end
    help("Runs everything in test/runtests.jl. Tests whose data is missing are skipped with a warning.")
end

# ─── Settings ─────────────────────────────────────────────────────────────

function settings_panel!(app, st)
    s = st.settings
    CImGui.SeparatorText("Background processes")
    CImGui.SliderInt("Julia threads", s.threads, 1, Sys.CPU_THREADS)
    help("Threads for each Hyperion process. Lower values leave more of the computer free; " *
         "CPU renders get slower.")
    CImGui.SliderInt("Priority (nice)", s.niceness, 0, 19)
    help("19 is the lowest priority: other programs come first. Ignored where nice is unavailable.")
    CImGui.AlignTextToFramePadding()
    CImGui.TextUnformatted("Default backend")
    CImGui.SameLine()
    CImGui.SetNextItemWidth(160)
    combo!("##default_backend", s.backend, BACKENDS)
    help("Used by panels whose backend you have not changed. 'auto' picks Metal or CUDA when available.")
    note("This computer has $(Sys.CPU_THREADS) CPU threads. New jobs use the current settings; " *
         "running jobs keep theirs.")
    CImGui.SeparatorText("About")
    CImGui.TextUnformatted("Repository: $REPO")
    CImGui.TextUnformatted("Julia $(VERSION); Mirage $(pkgversion(Mirage)); Hyperion $(pkgversion(Hyperion))")
end

# ─── Jobs ─────────────────────────────────────────────────────────────────

function jobs_window!(app, st)
    if CImGui.Begin("Jobs")
        if isempty(JOBS)
            note("No jobs yet.")
        else
            CImGui.BeginChild("joblist", CImGui.ImVec2(CImGui.GetContentRegionAvail().x * 0.38, 0), CImGui.ImGuiChildFlags_Borders)
            for job in reverse(JOBS)
                col = job.status == :running ? WARN : job.status == :done ? GOOD : job.status == :cancelled ? DIM : BAD
                CImGui.PushStyleColor(CImGui.ImGuiCol_Text, col)
                label = "#$(job.id) $(job.title) — $(job.status), $(human_duration(elapsed(job)))##job$(job.id)"
                CImGui.Selectable(label, st.selected_job == job.id) && (st.selected_job = job.id)
                CImGui.PopStyleColor()
                isempty(job.warning) || (CImGui.SameLine(); colored(WARN, "(CPU)"))
                if running(job)
                    f = progress_fraction(job)
                    f === nothing || CImGui.ProgressBar(f, (-1, 0), replace(job.progress, "%" => "%%"))
                end
            end
            CImGui.EndChild()
            CImGui.SameLine()
            st.selected_job in eachindex(JOBS) || (st.selected_job = length(JOBS))
            job = JOBS[st.selected_job]
            CImGui.BeginGroup()
            CImGui.TextUnformatted("#$(job.id) $(job.title)")
            CImGui.SameLine()
            job_button("Cancel"; disabled = !running(job)) && cancel!(job)
            CImGui.SameLine()
            CImGui.Button("Copy log") && CImGui.SetClipboardText(join(job.lines, '\n'))
            isempty(job.warning) || (CImGui.PushTextWrapPos(0); colored(WARN, job.warning); CImGui.PopTextWrapPos())
            CImGui.SameLine()
            if job_button("Clear finished"; disabled = all(running, JOBS))
                filter!(running, JOBS)
                for (i, j) in enumerate(JOBS)
                    j.id = i
                end
                st.selected_job = isempty(JOBS) ? 0 : length(JOBS)
            end
            if !isempty(JOBS) && st.selected_job in eachindex(JOBS)
                job = JOBS[st.selected_job]
                CImGui.BeginChild("log", (0, 0), CImGui.ImGuiChildFlags_Borders, CImGui.ImGuiWindowFlags_HorizontalScrollbar)
                at_bottom = CImGui.GetScrollY() >= CImGui.GetScrollMaxY() - 2
                n = length(job.lines)
                CImGui.TextUnformatted(join(@view(job.lines[max(1, n - 2000):n]), '\n'))
                running(job) && !isempty(job.progress) && colored(WARN, job.progress)
                at_bottom && CImGui.SetScrollHereY(1.0f0)
                CImGui.EndChild()
            end
            CImGui.EndGroup()
        end
    end
    CImGui.End()
end
