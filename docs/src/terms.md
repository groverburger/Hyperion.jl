# Documentation terms

These terms have one meaning throughout the current user guides.

| Term | Meaning |
|---|---|
| Backend | The CPU or GPU implementation that executes a kernel |
| Baseline | Stored reference values for a comparison |
| Bit-exact | Identical bits in the compared output buffers |
| DEM | Digital elevation model; a raster of terrain elevations |
| DSN | Deep Space Network; the output name for Earth visibility information |
| Farfield | An outer terrain layer used after a ray leaves the inner raster |
| Fixture | Stored input or reference data for a test |
| Frame | One image for one timestamp |
| Kernel | A calculation executed for many output pixels |
| Light curve | Illumination values for one location across time |
| Mapset | A directory of frames and related metadata |
| Mipmap | One level of a terrain pyramid used to bound ray samples |
| NAC | Narrow Angle Camera on the Lunar Reconnaissance Orbiter |
| Nearfield | The inner terrain raster used by the ray calculation |
| Observer height | Height above the terrain at the query pixel |
| Ray casting | Terrain samples along a direction to calculate a horizon |
| SHA-256 | The hash algorithm used to identify file or buffer bytes |
| Site | The lunar area represented by the inner DEM |
| SPICE | The ephemeris and geometry system used for body positions |
| Tile | An output window used to limit temporary buffers |
| UTC | Coordinated Universal Time |

Software names, code identifiers, commands, and mathematical symbols keep their exact spelling.
Technical verbs include compile, decode, encode, render, and serialize.
In this documentation, “render” means calculate image values from geometry and terrain.

## Writing procedure

The style reference is [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).
The standard contains writing rules and a controlled dictionary.

1. Use the dictionary meaning and part of speech for each general word.
2. Use the defined technical terms consistently.
3. Use active voice and short sentences.
4. Limit a procedure sentence to 20 words.
5. Limit a descriptive sentence to 25 words.
6. Put one instruction in each procedure sentence.
7. Put one topic in each paragraph.
8. Keep each paragraph to six sentences or fewer.
9. Use American English spelling.
10. Compare commands with the current code.
11. Identify historical results by date and test scope.

The standard is the authority for its full rules and dictionary.
A sentence-length check alone does not establish compliance.
