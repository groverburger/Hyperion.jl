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

Background jobs use the same Julia version as the GUI.
Start the GUI with Julia 1.11.5, the version used for Hyperion at present.
With juliaup, `juliaup override set 1.11.5` in the repository root selects that version for `julia` in this directory.
Metal or CUDA must be installed in the default environment of that Julia version, as the [README](https://github.com/groverburger/Hyperion.jl) describes.
If the GPU package does not load, the job runs on the CPU, and the Jobs window marks it with "(CPU)".

## Layout

The **Tools** window on the left has three tabs:

| Tab | Purpose |
|---|---|
| Mapset | Choose a specification, check what it calculates, and run it |
| Results | Step through the frames of a finished mapset and analyse them |
| Point tools | Light curve, azimuth and elevation, and pixel probe at one location |

The **View** window shows the selected specification's DEMs and windows, map frames, or a plot.
It follows the active tab, and its buttons switch between the three displays.
The **Jobs** window lists commands and shows the output of the selected command.
The **Tools** menu opens the Input files, Tests, and Settings windows.

Each calculation runs in a separate Julia process.
The process calls the same scripts that the documentation describes.
The Settings window sets the thread count and the `nice` priority of these processes.
The defaults are a quarter of the CPU threads and priority 15, so other programs stay responsive.
**Cancel** in the Jobs window stops a running command.

## Make a mapset

1. Choose a specification in the Mapset tab, or make one with **New...**.
   The View shows its DEMs and the output area.
2. Read the summary: the times, the frame size, and a status for each terrain file.
   **Verify files** compares the files with the SHA-256 values in the specification.
3. Optionally, click **Preview one frame** to render the first timestamp.
4. Click **Run mapset**.
   The progress bar shows the frames that are done.
   A stopped run continues from the last complete frame when you click **Resume**.
5. Click **Browse results** to open the frames in the Results tab.

**Check** verifies the files and prints the plan without rendering.
**Run options** changes the output name or folder, the times for one run, the backend, and the preview time and window.

## Edit a specification

**Edit spec** opens the specification as a form.
It has the mapset name, the times, optional settings, and one section for each terrain layer.
A layer section has the kind, file, and window.
**More settings** has the display name, the SHA-256 value, and the far-field grid options.
The arrow next to the file field lists the files in `data/inputs/`.
**Hash file** calculates the SHA-256 value of a file, and **Use this file's hash** copies it into the form.

The form marks values that it cannot read and the layer combinations that Hyperion does not accept.
Runs are not possible until you correct them.
Unsaved edits apply to runs through a scratch copy.
**Save** writes the form to the TOML file, and **Save as** writes it to a new file in `data/inputs/mapsets/`.
Saving does not keep comments from the original file.
**TOML that Save writes** shows the file contents before you save.

## DEMs and windows

The View shows one panel for each terrain layer of the selected specification.
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
A click picks the pixel for the point tools and the Results tab.
A pink crosshair marks the picked pixel.
**Clear** next to the picked coordinates, or a right-click on the map, removes the pick.
Plots, light curves, and azimuth and elevation results have **Open CSV** and **Show in Finder** buttons.

Palette PNGs give exact values.
RGB PNGs from earlier versions give only colours, so the readout omits their DSN values.

## Results

The Results tab opens any folder with `sun/` and `dsn/` subfolders, from Hyperion or mapbuilder.
**Look in another folder** selects a folder other than `data/outputs/`.
It displays `other/hillshade.tif` beneath the maps when that file has the map size.

The analysis sections are folded until you open them.
In **Earth contact map**, the **DSN threshold** sets the Earth-contact limit in degrees.
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
