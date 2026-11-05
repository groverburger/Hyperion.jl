using Test, Hyperion
@testset "PSR component filtering" begin
    mask = zeros(UInt8, 8, 9)
    mask[1,1:5] .= 1 # exactly five survives, including boundary
    mask[2,6] = 1 # diagonal contact does not connect
    mask[4,2:5] .= 1 # four removed
    mask[6:8,8:9] .= 1 # six survives
    mask[5,5] = 255
    original = copy(mask)
    r = Hyperion._filter_psr_mask(mask)
    @test r.removed_pixels == 5
    @test r.removed_regions == 2
    @test count(==(1), r.mask) == 11
    @test r.mask[5,5] == 255
    @test mask == original
    @test Hyperion._filter_psr_mask(mask; min_pixels=1).mask == mask
    @test_throws ArgumentError Hyperion._filter_psr_mask(mask; min_pixels=0)
    @test_throws ErrorException Hyperion._filter_psr_mask(fill(UInt8(2),2,2))
end
