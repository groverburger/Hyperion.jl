using KernelAbstractions

# TODO: This test-time automatic backend detection logic does not work with
# Julia 1.12 and newer versions. Need to fix eventually.

const TEST_BACKEND_ENV = lowercase(get(ENV, "HYP_BACKEND", "auto"))

function _try_with_default_env(f)
    had_default = any(x -> x == "@v#.#", LOAD_PATH)
    had_default || push!(LOAD_PATH, "@v#.#")
    try
        return f()
    finally
        had_default || filter!(x -> x != "@v#.#", LOAD_PATH)
    end
end

function _try_metal_backend()
    Sys.isapple() || return nothing
    _try_with_default_env() do
        try
            @eval using Metal
            ("metal", Base.invokelatest(Metal.MetalBackend), Metal.MtlArray)
        catch e
            @debug "Metal test backend not loadable" exception=e
            nothing
        end
    end
end

function _try_cuda_backend()
    _try_with_default_env() do
        try
            @eval using CUDA
            if Base.invokelatest(CUDA.functional)
                ("cuda", Base.invokelatest(CUDA.CUDABackend), CUDA.CuArray)
            else
                nothing
            end
        catch e
            @debug "CUDA test backend not loadable" exception=e
            nothing
        end
    end
end

function _select_test_backend()
    selected = if TEST_BACKEND_ENV == "metal"
        _try_metal_backend()
    elseif TEST_BACKEND_ENV == "cuda"
        _try_cuda_backend()
    elseif TEST_BACKEND_ENV == "cpu"
        ("cpu", KernelAbstractions.CPU(), Array)
    elseif TEST_BACKEND_ENV == "auto"
        if Sys.isapple()
            metal = _try_metal_backend()
            metal === nothing ? _try_cuda_backend() : metal
        else
            cuda = _try_cuda_backend()
            cuda === nothing ? _try_metal_backend() : cuda
        end
    else
        error("Unknown HYP_BACKEND='$TEST_BACKEND_ENV' - use auto|metal|cuda|cpu")
    end

    if selected === nothing
        allow_cpu = get(ENV, "HYP_ALLOW_CPU_TEST_BACKEND", "0") == "1"
        allow_cpu && return ("cpu", KernelAbstractions.CPU(), Array)
        @warn "No GPU test backend loadable; skipping expensive render regressions. " *
              "Install Metal/CUDA in the active or default Julia environment, " *
              "set HYP_BACKEND=metal|cuda, or set HYP_ALLOW_CPU_TEST_BACKEND=1."
        return ("none", nothing, nothing)
    end

    return selected
end

const TEST_BACKEND_NAME, TEST_BACKEND, TEST_DEVICE_ARRAY = _select_test_backend()
@info "Selected test backend" backend=TEST_BACKEND_NAME
