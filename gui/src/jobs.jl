# Background jobs: each Hyperion operation runs in its own Julia process at
# reduced priority with a thread limit, and the GUI collects its output.

const BACKENDS = ["auto", "metal", "cuda", "cpu"]

mutable struct Settings
    threads::Base.RefValue{Int32}
    niceness::Base.RefValue{Int32}
    backend::Base.RefValue{Int32}     # index into BACKENDS
end
Settings() = Settings(Ref(Int32(max(1, Sys.CPU_THREADS ÷ 4))), Ref(Int32(15)), Ref(Int32(0)))

backend_name(s::Settings) = BACKENDS[s.backend[] + 1]

mutable struct Job
    id::Int
    title::String
    cmd::Cmd
    lines::Vector{String}
    partial::String
    progress::String              # latest carriage-return line, e.g. from ProgressMeter
    status::Symbol                # :running, :done, :failed, :cancelled
    proc::Union{Nothing,Base.Process}
    started::Float64
    finished::Float64
    on_done::Union{Nothing,Function}
    warning::String               # shown with the job, e.g. a GPU fallback
end

const JOBS = Job[]

const GPU_FALLBACK = "The GPU backend did not load, so this job runs on the CPU. " *
    "Metal or CUDA must be installed in the default environment of the Julia version that runs " *
    "the job (Julia $(VERSION.major).$(VERSION.minor)); see Settings."

# Repaint the window after output arrives; `app` is nothing in headless use.
wake(app) = app === nothing || request_frame!(app)
const LOG_LIMIT = 20_000
const ANSI = r"\e\[[0-9;?]*[A-Za-z]"

running(job::Job) = job.status == :running
elapsed(job::Job) = (running(job) ? time() : job.finished) - job.started
any_running() = any(running, JOBS)

"""
    julia_job(settings, script, args...; env = ()) -> Cmd

Command that runs `script` (relative to the repository) in the main Hyperion
environment with the configured thread count and `nice` priority.
"""
function julia_job(settings::Settings, args...; env = ())
    jl = Base.julia_cmd()
    cmd = `$jl --project=$REPO --threads=$(settings.threads[]) --startup-file=no --color=no $(collect(String, args))`
    nice = Sys.which("nice")
    nice === nothing || (cmd = `$nice -n $(settings.niceness[]) $cmd`)
    return addenv(Cmd(cmd; dir = REPO), "JULIA_NUM_THREADS" => string(settings.threads[]),
                  "OPENBLAS_NUM_THREADS" => "1", env...)
end

"""
    start_job!(app, title, cmd; on_done = nothing) -> Job

Start `cmd` without waiting. Output is captured line by line. `on_done(job)`
runs on the GUI task after the process exits.
"""
function start_job!(app, title::AbstractString, cmd::Cmd; on_done = nothing)
    job = Job(length(JOBS) + 1, String(title), cmd, String[], "", "", :running,
              nothing, time(), 0.0, on_done, "")
    push!(JOBS, job)
    push!(job.lines, "\$ " * join(cmd.exec, " "))
    out = Pipe()
    try
        job.proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
    catch e
        push!(job.lines, "Could not start: " * sprint(showerror, e))
        job.status = :failed
        job.finished = time()
        return job
    end
    close(out.in)
    @async begin
        try
            while !eof(out)
                feed!(job, String(readavailable(out)))
                wake(app)
            end
        catch e
            push!(job.lines, "Output reader error: " * sprint(showerror, e))
        end
        wait(job.proc)
        isempty(job.partial) || push!(job.lines, job.partial)
        job.partial = ""
        if job.status != :cancelled
            job.status = success(job.proc) ? :done : :failed
        end
        job.finished = time()
        push!(job.lines, "[$(job.status) after $(human_duration(elapsed(job))), exit code $(job.proc.exitcode)]")
        if job.on_done !== nothing
            try
                job.on_done(job)
            catch e
                push!(job.lines, "GUI follow-up failed: " * sprint(showerror, e))
            end
        end
        wake(app)
    end
    return job
end

function feed!(job::Job, chunk::AbstractString)
    text = job.partial * replace(chunk, ANSI => "")
    parts = split(text, '\n')
    for line in parts[1:end-1]
        # Keep the final state of lines redrawn with carriage returns.
        segs = filter(!isempty, split(line, '\r'))
        isempty(segs) || push!(job.lines, String(last(segs)))
        occursin("falling back to CPU", line) && (job.warning = GPU_FALLBACK)
    end
    rest = parts[end]
    if occursin('\r', rest)
        segs = filter(!isempty, split(rest, '\r'))
        isempty(segs) || (job.progress = String(last(segs)))
        rest = String(split(rest, '\r')[end])
    end
    job.partial = String(rest)
    length(job.lines) > LOG_LIMIT && deleteat!(job.lines, 2:length(job.lines) - LOG_LIMIT + 1)
    return job
end

function cancel!(job::Job)
    running(job) || return
    job.status = :cancelled
    try
        kill(job.proc)
    catch
    end
end

cancel_all!() = foreach(cancel!, JOBS)

# Fraction from a ProgressMeter line such as "Rendering  42%|████ | ETA: 0:01:10".
function progress_fraction(job::Job)
    m = match(r"(\d{1,3})%", job.progress)
    m === nothing && return nothing
    return clamp(parse(Int, m[1]) / 100, 0, 1)
end

# Run `cmds` one after another; stop at the first failure or cancellation.
function start_sequence!(app, items::Vector{<:Tuple{String,Cmd}}; on_each = nothing)
    isempty(items) && return
    (title, cmd), rest = items[1], items[2:end]
    start_job!(app, title, cmd; on_done = job -> begin
        on_each === nothing || on_each(job)
        job.status == :done && start_sequence!(app, rest; on_each)
    end)
end
