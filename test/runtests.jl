using Test
using JuliaMapbuilder

@testset "JuliaMapbuilder" begin
    @testset "LUT SHA verification" begin
        @test JuliaMapbuilder.verify_lut_integrity()
    end

    @testset "atan2_lut basic" begin
        # (0, 0) → 0
        @test JuliaMapbuilder.atan2_lut(0f0, 0f0) === 0f0

        # Cardinal directions
        @test JuliaMapbuilder.atan2_lut(0f0, 1f0) ≈ 0f0 atol=1f-5
        @test JuliaMapbuilder.atan2_lut(1f0, 0f0) ≈ Float32(π/2) atol=1f-4
        @test JuliaMapbuilder.atan2_lut(0f0, -1f0) ≈ Float32(π) atol=1f-4
        @test JuliaMapbuilder.atan2_lut(-1f0, 0f0) ≈ Float32(-π/2) atol=1f-4

        # Diagonal
        @test JuliaMapbuilder.atan2_lut(1f0, 1f0) ≈ Float32(π/4) atol=1f-4
    end

    @testset "cos_sin_lut basic" begin
        c, s = JuliaMapbuilder.cos_sin_lut(0f0)
        @test c ≈ 1f0 atol=1f-5
        @test s ≈ 0f0 atol=1f-5

        c, s = JuliaMapbuilder.cos_sin_lut(JuliaMapbuilder.PI_HALF_F32)
        @test c ≈ 0f0 atol=1f-4
        @test s ≈ 1f0 atol=1f-4

        c, s = JuliaMapbuilder.cos_sin_lut(JuliaMapbuilder.PI_F32)
        @test c ≈ -1f0 atol=1f-4
        @test s ≈ 0f0 atol=1f-4
    end
end
