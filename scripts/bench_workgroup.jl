#!/usr/bin/env julia
# A/B benchmark for GPU workgroup size on the live shadow kernel.
# Pick your backend below.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, Statistics
import Hyperion as Hyp

using Metal
const BACKEND    = Metal.MetalBackend()
const DEVICE_ARR = Metal.MtlArray

# using CUDA
# const BACKEND    = CUDA.CUDABackend()
# const DEVICE_ARR = CUDA.CuArray

ldem = Hyp.load_ldem(Hyp.ensure_ldem!())
Hyp.init_spice(joinpath(dirname(@__DIR__), "kernels"))
max_mm, min_mm = Hyp.build_ldem_mipmaps_minmax(ldem.data)

# Use a high-sun timestamp (slowest in the year) to get the most signal
dts = [DateTime(2027, 7, 16, 8, 0, 0),   # slowest from year gen
       DateTime(2027, 6, 23, 12, 0, 0),  # mid-sun
       DateTime(2027, 6, 1, 0, 0, 0)]    # night-side
origin_r, origin_c = 8960, 18432
H, W = 512, 896

function bench(wg_size::Int, n_reps::Int=3)
    # Warmup
    for dt in dts
        et = Hyp.datetime_to_et(dt)
        s_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
        e_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
        Hyp.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
            s_t, e_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm,
            workgroup_size=wg_size)
    end

    total_t = Float64[]
    for dt in dts
        et = Hyp.datetime_to_et(dt)
        s_t = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
        e_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))
        times = Float64[]
        for _ in 1:n_reps
            t = @elapsed Hyp.generate_live_shadow_frame_gpu(ldem.data,
                origin_r, origin_c, H, W, s_t, e_t, 0.0;
                max_mipmaps=max_mm, min_mipmaps=min_mm, workgroup_size=wg_size,
            backend=BACKEND, DeviceArray=DEVICE_ARR)
            push!(times, t)
        end
        push!(total_t, median(times))
    end
    return total_t
end

@printf("\n%-6s  %-12s  %-12s  %-12s  %-8s\n", "wg", "high-sun", "mid-sun", "night", "total")
println("-" ^ 60)
# Keep well below Metal's maxTotalThreadsPerThreadgroup (1024) to avoid
# register spill / watchdog-timeout crashes on the unified M-series GPU.
for wg in [128, 256, 512]
    times = bench(wg, 3)
    total = sum(times)
    @printf("%-6d  %-12.3f  %-12.3f  %-12.3f  %-8.3f\n",
            wg, times[1], times[2], times[3], total)
end
