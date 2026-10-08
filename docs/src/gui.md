# Desktop GUI

The desktop GUI runs the Hyperion commands and displays their results.
It uses [Mirage](https://github.com/nasa/Mirage.jl) and has its own Julia environment in `gui/`.
The main Hyperion environment does not depend on the GUI packages.

## Start the GUI

Run these commands from the repository root:

```bash
julia --project=gui -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --project=gui gui/hyperion_gui.jl
```

The window needs an active display.
The GUI exits with a message if the screen is locked or asleep.

## Layout

The **Tools** window on the left contains one tab for each operation.
The **View** window shows map images and time-series plots.
The **Jobs** window lists commands and shows the output of the selected command.

Each calculation runs in a separate Julia process.
The process calls the same scripts that the documentation describes.
The **Settings** tab sets the thread count and the `nice` priority of these processes.
The defaults are a quarter of the CPU threads and priority 15, so other programs stay responsive.
**Cancel** in the Jobs window stops a running command.

## Tools

| Tab | Operation |
|---|---|
| Mapset | Edit a specification, then run, dry-run, or resume a [mapset](mapsets.md) |
| Preview | Render one timestamp, optionally for a smaller window, before a long run |
| Light curve | Run the [light-curve command](light-curves.md) and plot the result |
| Az/el | Run the [azimuth and elevation export](azel.md) and plot the result |
| Probe | Run `tools/debug/probe_pixel.jl` for one pixel |
| Browse | Step through a mapset, show contact maps, compare mapsets, and compute statistics |
| Inputs | Hash the files in `data/inputs/` and identify known terrain products |
| Tests | Run selected test files, or the full suite |

The Mapset, Preview, Light curve, and Probe tabs use the specification selected at the top of the tab.

## Edit a specification

The Mapset tab shows the selected specification as a form.
It has the mapset name, the times, optional settings, and one section for each terrain layer.
A layer section has the kind, file, display name, SHA-256 value, window, and the far-field grid options.
The arrow next to the file field lists the files in `data/inputs/`.
**Hash file** calculates the SHA-256 value of a file, and **Use this file's hash** copies it into the form.

The form marks values that it cannot read and the layer combinations that Hyperion does not accept.
Runs are not possible until you correct them.
Unsaved edits apply to runs through a scratch copy.
**Save** writes the form to the TOML file, and **Save as** writes it to a new file in `data/inputs/mapsets/`.
**New** makes an empty specification.
Saving does not keep comments from the original file.
**TOML that Save writes** shows the file contents before you save.

## DEMs and windows

**Show DEMs and windows** displays one panel for each terrain layer in the View.
Each panel shows a shaded overview of the DEM, with these outlines:

| Outline | Meaning |
|---|---|
| Output area (cyan) | The calculated area: the first layer's window, or the full first layer |
| Window (orange) | A layer's own window |
| Other layers' extents (magenta) | The full extent of each other layer, converted to this grid |

The outlines use the same latitude and longitude conversion as the light-curve command.
A site extent on the polar grid is therefore rotated.
The pointer readout gives the pixel, the latitude and longitude, the approximate elevation, and the matching pixel in each other layer.
**Output area** zooms to the calculated area.
With **Drag on layer 1 to set the output window**, a drag on the first panel replaces the first layer's window in the form.

A background process makes each overview once and keeps it in the scratch directory.
A large DEM also gets a more detailed overview around the output area.

## View

Scroll to zoom and drag to pan.
The pointer readout gives the output pixel, the first-DEM pixel, and the stored value of every loaded layer.
A click picks the pixel for the Light curve, Probe, and Browse tabs.

Palette PNGs give exact values.
RGB PNGs from earlier versions give only colours, so the readout omits their DSN values.

## Browse

Browse opens any folder with `sun/` and `dsn/` subfolders, from Hyperion or mapbuilder.
It displays `other/hillshade.tif` beneath the maps when that file has the map size.

The **DSN threshold** sets the Earth-contact limit in degrees.
The contact layer marks pixels at or above the limit.
The sun-and-contact layer has four classes: dark without contact, lit without contact, contact while dark, and lit with contact.
Lit means a nonzero Sun value.

**Compare** loads a second mapset at the same timestamp.
The difference layers need the same grid size.
The GUI does not resample between grids.

**Time series at a pixel** reads the picked pixel from every frame and plots the Sun and DSN values.
**Statistics over all frames** calculates these values for each pixel:

| Layer | Meaning |
|---|---|
| `lit_percent` | Percentage of frames with Sun value at or above the limit |
| `mean_sun_percent` | Mean visible fraction of the solar disk |
| `longest_shadow_hours` | Longest continuous run below the Sun limit |
| `contact_percent` | Percentage of frames at or above the DSN threshold |
| `longest_outage_hours` | Longest continuous run below the DSN threshold |
| `lit_and_contact_percent` | Percentage of frames both lit and in contact |

A run lasts from its first frame to the first frame after it.
A run at the end of the mapset lasts one more frame interval.
These two calculations decode every frame in a separate process.
Use **every Nth frame** for a faster estimate.
The statistics need palette DSN files.

## Scratch files

The GUI writes previews, edited specifications, DEM overviews, time series, statistics, and the hash cache to `data/outputs/.hyperion_gui/`.
Git ignores that directory.
