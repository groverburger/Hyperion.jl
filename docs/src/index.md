# Hyperion.jl

Hyperion makes lunar illumination maps from terrain data and SPICE ephemerides.
The primary output is a mapset: a directory of images and metadata for selected times.

## Available operations

| Operation | Guide |
|---|---|
| Make Sun and Earth visibility maps | [Mapsets](mapsets.md) |
| Export Sun and Earth geometry for one location | [Azimuth and elevation CSV](azel.md) |
| Select terrain inputs | [Data](data.md) |
| Test the software | [Tests](testing.md) |
| Find a command | [Workflow commands](tooling.md) |
| Understand the renderer | [Ray casting](raycasting.md) |

The Sun maps show the visible fraction of the solar disk.
The DSN maps show the Earth elevation above the terrain horizon.
The CSV command gives geometry without terrain shadow calculations.
A point light-curve command is not available yet.

## Installation

The [repository README](https://github.com/groverburger/Hyperion.jl) gives the installation procedure.
The tested Julia version is 1.11.5.
Local terrain files and the `gdaldem` command are necessary for map generation.
Those inputs are not necessary for the small synthetic tests.

## Terms and evidence

The [terms page](terms.md) defines the technical vocabulary.
Dated reference pages describe specific investigations and test runs.
A previous result does not prove that every later code version or backend gives the same result.
