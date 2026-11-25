# PSR tab: classify permanently shadowed regions over the current preset's
# terrain and image area with scripts/generate_psr.jl, follow its checkpoint,
# and show the raw and size-filtered masks in the View.

const PSR_DEFAULTS = (start = "1970-01-01T00:00:00", stop = "2044-01-01T00:00:00",
                      step = "6", height = "0")
const PSR_SHADOW = RGBA{N0f8}(0.90, 0.15, 0.15, 1)
const PSR_LIT = RGBA{N0f8}(0, 0, 0, 1)
const PSR_NODATA = RGBA{N0f8}(0, 0, 0, 0)
const PSR_CHECKPOINT_MISMATCH = "checkpoint inputs/settings/code differ"

# Study settings from the form; `nothing` for a field that cannot be read.
function psr_settings(F::Fields)
    start = parse_utc(str(F, :psr_start))
    stop = parse_utc(str(F, :psr_stop))
    step = tryparse(Int, str(F, :psr_step))
    height = tryparse(Float64, str(F, :psr_height))
    step !== nothing && step <= 0 && (step = nothing)
    height !== nothing && !isfinite(height) && (height = nothing)
    return (; start, stop, step, height, cutoff = bref(F, :psr_cutoff)[])
end

function psr_problems(p)
    out = String[]
    p.start === nothing && push!(out, "Cannot read the start time.")
    p.stop === nothing && push!(out, "Cannot read the stop time.")
    p.start !== nothing && p.stop !== nothing && p.stop < p.start && push!(out, "Stop is before start.")
    p.step === nothing && push!(out, "The step must be a whole number of hours above zero.")
    p.height === nothing && push!(out, "Cannot read the observer height.")
    return out
end

psr_samples(p) = isempty(psr_problems(p)) ? fld(Dates.value(p.stop - p.start), 3_600_000p.step) + 1 : 0

function psr_dir(st)
    F = st.fields
    default = joinpath("data", "outputs", spec_name(st.spec), bref(F, :psr_cutoff)[] ? "psr-mapbuilder" : "psr")
    return default, projpath(isempty(str(F, :psr_out)) ? default : str(F, :psr_out))
end

const PSR_META_CACHE = Dict{String,Tuple{Float64,Any}}()

# metadata.toml of a run in `dir`, reread when the file changes.
function psr_metadata(dir)
    path = joinpath(dir, "metadata.toml")
    isfile(path) || return nothing
    t, meta = get(PSR_META_CACHE, path, (0.0, nothing))
    if t != mtime(path)
        meta = try
            TOML.parsefile(path)
        catch
            nothing     # being replaced; keep the previous reading next frame
        end
        meta === nothing && return get(PSR_META_CACHE, path, (0.0, nothing))[2]
        PSR_META_CACHE[path] = (mtime(path), meta)
    end
    return meta
end

# Differences between a saved run and the form, which would stop it resuming.
function psr_differences(meta, p)
    (meta === nothing || !isempty(psr_problems(p))) && return String[]
    out = String[]
    string(p.start) == get(meta, "start", "") || push!(out, "start $(get(meta, "start", "?"))")
    string(p.stop) == get(meta, "stop", "") || push!(out, "stop $(get(meta, "stop", "?"))")
    3_600_000p.step == get(meta, "step_ms", -1) ||
        push!(out, @sprintf("step %.4g h", get(meta, "step_ms", 0) / 3_600_000))
    Float64(p.height) == Float64(get(meta, "observer_height_m", NaN)) ||
        push!(out, "observer $(get(meta, "observer_height_m", "?")) m")
    p.cutoff == get(meta, "mapbuilder_cutoff", false) ||
        push!(out, get(meta, "mapbuilder_cutoff", false) ? "mapbuilder cutoff" : "strict cutoff")
    return out
end

# Latest progress line of a running job, e.g.
# "PSR 12/108113 samples (0.01%) | 171464 of 458752 pixels unlit (37.38%) | 1m 05s elapsed | ~9h 12m left".
const PSR_PROGRESS = r"^PSR (\d+)/(\d+) samples \([^)]*\) \| (.*)$"
function psr_progress(job::Job)
    for line in Iterators.reverse(job.lines)
        m = match(PSR_PROGRESS, line)
        m === nothing && continue
        return (; done = parse(Int, m[1]), total = parse(Int, m[2]), detail = replace(m[3], " | " => "; "))
    end
    return nothing
end

function psr_job!(app, st; dir)
    F = st.fields
    p = psr_settings(F)
    name = spec_name(st.spec)
    args = ["scripts/generate_psr.jl", "--spec=$(effective_spec_path(st.spec))", "--out=$dir",
            "--start=$(p.start)", "--stop=$(p.stop)", "--step-hours=$(p.step)",
            "--observer-height-m=$(p.height)", "--backend=$(panel_backend(st, :psr_backend))"]
    p.cutoff && push!(args, "--mapbuilder-cutoff")
    limit = iref(F, :psr_max_samples)[]
    limit > 0 && push!(args, "--max-samples=$limit")
    origin = output_origin(st.spec.cfg)
    return launch!(app, st, "PSR " * name, args...; on_done = j -> begin
        delete!(PSR_META_CACHE, joinpath(dir, "metadata.toml"))
        if j.status == :failed && any(l -> occursin(PSR_CHECKPOINT_MISMATCH, l), j.lines)
            st.message = "The checkpoint in $(shortpath(dir)) is from other settings, terrain, or code. " *
                         "Choose another output folder in the PSR tab."
        elseif j.status == :done && isfile(joinpath(dir, "psr.tif"))
            follow_up(() -> show_psr!(st, dir, origin))
        end
    end)
end

function psr_image(path, label; origin)
    values = AG.read(path) do ds
        Matrix{UInt8}(permutedims(AG.read(AG.getband(ds, 1))))
    end
    n = count(==(0x01), values)
    color = v -> v == 1 ? PSR_SHADOW : v == 0 ? PSR_LIT : PSR_NODATA
    describe = v -> v == 1 ? "PSR: no sunlight at any sample" : v == 0 ? "sunlit at some sample" : "no data"
    return value_image("$label ($n PSR pixels)", values, color; kind = :psr, origin, path, describe,
                       legend = [(PSR_SHADOW, "PSR"), (PSR_LIT, "sunlit at some time")])
end

function show_psr!(st, dir, origin)
    images = MapImage[]
    for (file, label) in (("psr-filtered.tif", "PSR, regions ≥ 5 pixels"), ("psr.tif", "PSR, raw"))
        path = joinpath(dir, file)
        isfile(path) && push!(images, psr_image(path, label; origin))
    end
    set_images!(st.view, "PSR: " * shortpath(dir), images)
end

# Remove the files of a PSR run in `dir`, keeping anything else there.
function delete_psr_run!(dir)
    for f in ("psr.tif", "psr-filtered.tif", "checkpoint.bin", "metadata.toml", "progress.csv")
        rm(joinpath(dir, f); force = true)
    end
    delete!(PSR_META_CACHE, joinpath(dir, "metadata.toml"))
end

function psr_tab!(app, st)
    s, F = st.spec, st.fields
    label!("Terrain")
    CImGui.TextUnformatted("preset " * preset_label(s.path) * (dirty(s) ? " (changed)" : ""))
    if s.cfg === nothing
        note("Choose a preset in the Render tab.")
        return
    end
    o = output_size(s.cfg)
    label!("Image area")
    CImGui.TextUnformatted(o === nothing ? "unknown" : @sprintf("%d × %d pixels at %.4g m", o.cols, o.rows, o.pixel))
    help("The terrain and image area of the Render tab. The preset's times and observer height are not used.")

    heading("Study")
    field!(F, "Start (UTC)", :psr_start; default = PSR_DEFAULTS.start)
    field!(F, "Stop (UTC)", :psr_stop; default = PSR_DEFAULTS.stop)
    field!(F, "Step (hours)", :psr_step; default = PSR_DEFAULTS.step, width = 100)
    field!(F, "Observer height (m)", :psr_height; default = PSR_DEFAULTS.height, width = 100)
    help("0 for ground-level PSRs.")
    label!("")
    CImGui.Checkbox("Mapbuilder sunlight cutoff", bref(F, :psr_cutoff))
    help("Off: any sunlight above zero counts. On: below 1/255 of the solar disk counts as shadow, as in mapbuilder.")
    p = psr_settings(F)
    problems = vcat(psr_problems(p), spec_problems(st))
    n = psr_samples(p)
    if n > 0
        label!("")
        CImGui.TextDisabled("$n sample times")
    end

    heading("Run")
    default_dir, dir = psr_dir(st)
    meta = psr_metadata(dir)
    complete = meta !== nothing && get(meta, "complete", false) && isfile(joinpath(dir, "psr.tif"))
    differs = psr_differences(meta, p)
    job = running_job("PSR " * spec_name(s))
    card("psr") do
        if job !== nothing
            line = psr_progress(job)
            if line !== nothing
                CImGui.ProgressBar(line.total > 0 ? line.done / line.total : 0.0, CImGui.ImVec2(-1, 0),
                                   "$(line.done) of $(line.total) samples")
                colored(DIM, line.detail)
            else
                CImGui.ProgressBar(0.0, CImGui.ImVec2(-1, 0), "preparing terrain and ephemeris")
            end
            colored(DIM, "running for $(human_duration(elapsed(job))); the checkpoint is saved about every minute")
            isempty(job.warning) || CImGui.TextWrapped(replace(job.warning, "%" => "%%"))
            CImGui.Button("Stop") && cancel!(job)
            help("Run again later to resume from the last checkpoint.")
        elseif meta !== nothing && !isempty(differs)
            colored(WARN, "$(shortpath(dir)) has a run with other settings: " * join(differs, ", ") * ".")
            other = free_name(dirname(dir), basename(dir))
            if primary_button("Run in $other"; disabled = !isempty(problems))
                set!(tf(F, :psr_out), relpath(joinpath(dirname(dir), other), REPO))
                psr_job!(app, st; dir = joinpath(dirname(dir), other))
            end
        elseif complete
            colored(GOOD, @sprintf("Complete: %d PSR pixels before filtering", get(meta, "remaining_candidates", 0)))
            CImGui.SameLine()
            CImGui.SmallButton("Show") && show_psr!(st, dir, output_origin(s.cfg))
            other = free_name(dirname(dir), basename(dir))
            if primary_button("Run again in $other"; disabled = !isempty(problems))
                set!(tf(F, :psr_out), relpath(joinpath(dirname(dir), other), REPO))
                psr_job!(app, st; dir = joinpath(dirname(dir), other))
            end
            CImGui.SameLine()
            CImGui.BeginDisabled(!isempty(problems))
            CImGui.Button("Replace...##psr") && CImGui.OpenPopup("Replace PSR run")
            CImGui.EndDisabled()
            if CImGui.BeginPopup("Replace PSR run")
                CImGui.TextUnformatted("Delete the PSR maps and checkpoint in $(shortpath(dir)) and run again?")
                if CImGui.Button("Delete and run")
                    delete_psr_run!(dir)
                    psr_job!(app, st; dir)
                    CImGui.CloseCurrentPopup()
                end
                CImGui.SameLine()
                CImGui.Button("Cancel##psr_replace") && CImGui.CloseCurrentPopup()
                CImGui.EndPopup()
            end
        else
            done = meta === nothing ? 0 : get(meta, "completed_samples", 0)
            label = done > 0 ? "Resume ($done of $(get(meta, "sample_count", n)) samples done)" : "Find PSRs"
            primary_button(label; disabled = !isempty(problems)) && psr_job!(app, st; dir)
            CImGui.SameLine()
            colored(DIM, "to " * shortpath(dir))
        end
        foreach(x -> colored(BAD, x), problems)
        if isdir(dir)
            CImGui.Button((Sys.isapple() ? "Show in Finder" : "Show in folder") * "##psr") && open_external(dir)
        end
    end
    note("A run can take hours. Stopping keeps the checkpoint; changing the terrain, settings, or code " *
         "needs a new output folder.")
    CImGui.Spacing()
    if CImGui.CollapsingHeader("Options##psr")
        backend_combo!(F, :psr_backend, st.settings)
        field!(F, "Output folder", :psr_out; hint = default_dir)
        label!("Stop after")
        CImGui.SetNextItemWidth(120)
        CImGui.InputInt("samples##psr_max", iref(F, :psr_max_samples, 0))
        iref(F, :psr_max_samples)[] = max(0, iref(F, :psr_max_samples)[])
        help("0: no limit. A limited run saves a checkpoint to resume, not a PSR map.")
    end
end
