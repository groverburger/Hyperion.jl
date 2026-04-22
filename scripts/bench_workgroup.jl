#!/usr/bin/env julia
# A/B benchmark for Metal workgroup size on the live shadow kernel.
using Pkg; Pkg.activate(dirname(@__DIR__))
using Dates, Printf, Statistics
import JuliaMapbuilder as JM

ldem = JM.load_ldem(joinpath(dirname(@__DIR__), "data", "inputs", "ldem_80s_20m.img"))
JM.init_spice(joinpath(dirname(@__DIR__), "kernels"))
max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)

# Use a high-sun timestamp (slowest in the year) to get the most signal
dts = [DateTime(2027, 7, 16, 8, 0, 0),   # slowest from year gen
       DateTime(2027, 6, 23, 12, 0, 0),  # mid-sun
       DateTime(2027, 6, 1, 0, 0, 0)]    # night-side
origin_r, origin_c = 8960, 18432
H, W = 512, 896

function bench(wg_size::Int, n_reps::Int=3)
    # Warmup
    for dt in dts
        et = JM.datetime_to_et(dt)
        s_t = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
        e_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
        JM.generate_live_shadow_frame_gpu(ldem.data, origin_r, origin_c, H, W,
            s_t, e_t, 0.0; max_mipmaps=max_mm, min_mipmaps=min_mm,
            workgroup_size=wg_size)
    end

    total_t = Float64[]
    for dt in dts
        et = JM.datetime_to_et(dt)
        s_t = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
        e_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))
        times = Float64[]
        for _ in 1:n_reps
            t = @elapsed JM.generate_live_shadow_frame_gpu(ldem.data,
                origin_r, origin_c, H, W, s_t, e_t, 0.0;
                max_mipmaps=max_mm, min_mipmaps=min_mm, workgroup_size=wg_size)
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
