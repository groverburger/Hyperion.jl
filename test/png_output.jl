using Test, Hyperion
import FileIO
import ArchGDAL

# PNG colour type (IHDR byte 26) and chunk names, read from the file bytes.
png_color_type(bytes) = bytes[26]
has_chunk(bytes, name) = occursin(name, String(copy(bytes)))

@testset "Palette PNG output" begin
    data = UInt8[(7r + 3c) % 256 for r in 1:24, c in 1:40]
    data[1, 1:4] .= (0x00, 0x02, 0x03, 0xfa)

    mktempdir() do dir
        @testset "DSN values round-trip exactly" begin
            path = joinpath(dir, "dsn.png")
            Hyperion.save_indexed_png(data, Hyperion.DSN_PALETTE, path;
                transparent_from = Hyperion.DSN_TRANSPARENT_FROM)
            bytes = read(path)
            @test png_color_type(bytes) == 0x03
            @test has_chunk(bytes, "tRNS")

            img = FileIO.load(path)
            @test img.index == data
            for i in 0:255
                c = img.values[i]
                @test reinterpret.(UInt8, (c.r, c.g, c.b)) ==
                      Tuple(Hyperion.DSN_PALETTE[i + 1, :])
                @test reinterpret(UInt8, c.alpha) == (i >= 71 ? 0x00 : 0xff)
            end
        end

        @testset "Sun values round-trip exactly" begin
            path = joinpath(dir, "sun.png")
            Hyperion.save_indexed_png(data, Hyperion.SUN_PALETTE, path)
            bytes = read(path)
            @test png_color_type(bytes) == 0x03
            @test !has_chunk(bytes, "tRNS")

            img = FileIO.load(path)
            @test img.index == data
            @test all(reinterpret(UInt8, img[r, c].r) == data[r, c] &&
                      img[r, c].r == img[r, c].g == img[r, c].b
                      for r in axes(data, 1), c in axes(data, 2))
        end

        @testset "GeoTIFF values, palette, and georeference" begin
            path = joinpath(dir, "dsn.tif")
            gt = [-100.0, 5.0, 0.0, 200.0, 0.0, -5.0]
            wkt = Hyperion._polar_stereographic_wkt(; radius_m = Hyperion.MOON_RADIUS_M,
                latitude_of_origin = -90.0, central_meridian = 0.0, scale_factor = 1.0,
                false_easting = 0.0, false_northing = 0.0)
            Hyperion.save_indexed_geotiff(data, Hyperion.DSN_PALETTE, path;
                geotransform = gt, projection = wkt)
            ArchGDAL.read(path) do ds
                @test ArchGDAL.getgeotransform(ds) == gt
                @test occursin("Polar", ArchGDAL.getproj(ds))
                band = ArchGDAL.getband(ds, 1)
                @test permutedims(ArchGDAL.read(band)) == data
                @test ArchGDAL.getcolorinterp(band) == ArchGDAL.GCI_PaletteIndex
                ArchGDAL.getcolortable(band) do ct
                    for i in (0, 70, 71, 255)
                        e = ArchGDAL.getcolorentryasrgb(ct, i)
                        @test (e.c1, e.c2, e.c3) == Tuple(Hyperion.DSN_PALETTE[i + 1, :])
                    end
                end
            end
        end
    end
end
