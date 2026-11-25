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

The **Tools** window on the left has four tabs:

| Tab | Purpose |
|---|---|
| Render | Start from a preset, set the area and times, and save sun and DSN images |
| Results | Step through rendered images and analyse them |
| Light curve | Solar-disk visibility over time at the marker, with a horizon panorama |
| PSR | Map permanently shadowed regions over the image area |

The **View** window shows the terrain of the current settings, images, or a plot.
It follows the active tab, and its buttons switch between the three displays.
The **Jobs** window lists commands and shows the output of the selected command.
The **Marker** menu places the pink marker at typed coordinates or removes it.
The **Tools** menu opens the Sun and Earth az/el, Pixel probe, Input files, Tests, and Settings windows.

Each calculation runs in a separate Julia process.
The process calls the same scripts that the documentation describes.
The Settings window sets the thread count and the `nice` priority of these processes.
The defaults are a quarter of the CPU threads and priority 15, so other programs stay responsive.
**Cancel** in the Jobs window stops a running command.

## Presets

A preset is a set of saved render settings: the terrain files, the image area, the times, the image format, and the observer height.
Presets are the TOML files in `data/inputs/mapsets/`, the specifications that the [mapset command](mapsets.md) reads.

Choosing a preset loads its settings.
Every control in the Render tab changes the current settings, not the file.
When the settings differ from the file, the tab says so and offers **Save to preset** and **Revert**.
**Save as new preset...** writes the current settings to a new file, and **New...** starts a preset with one empty site layer.
Renders and light curves use the current settings, saved or not.
Saving does not keep comments from the original file.

## Render images

1. Choose a preset.
   The View shows its terrain and the image area.
2. Check **Area and terrain**: the image size and a status for each terrain file.
   **Verify files** compares the files with the SHA-256 values in the preset.
3. Set the **Times**: one time, a range with a step in hours, or a list of times.
4. Under **Save images**, choose the format, the name, and the output folder.
5. Click **Render**.
   The progress bar shows the times that are done.
   A stopped render continues from the last complete time when you click **Resume**.
6. Click **Open in Results** to step through the images.

Each time gives a sun image in `<folder>/<name>/sun/` and a DSN image in `<folder>/<name>/dsn/`.
GeoTIFF is the default format.
A GeoTIFF has the first terrain layer's projection and geotransform, the stored values, and a colour table.
A PNG is a palette image with the same values and no georeference.
TIFF colour tables have no transparency, so DSN values from 7.1° up are opaque in GeoTIFFs and transparent in PNGs.

**Options** has the backend, the GPU count, **Replace existing images**, the az/el table step, the dataset description file, and the preset file contents.

## Terrain and area

**Change terrain or area...** shows one section for each terrain layer.
A layer section has the kind, file, and window.
**More settings** has the display name, the SHA-256 value, and the far-field grid options.
The arrow next to the file field lists the files in `data/inputs/`.
**Hash file** calculates the SHA-256 value of a file, and **Use this file's hash** copies it into the settings.
**Observer and performance** has the observer height, the tile size, and the GPU workgroup size.

The tab marks values that it cannot read and the layer combinations that Hyperion does not accept.
Renders are not possible until you correct them.

The Terrain view shows one panel for each terrain layer.
Each panel shows a shaded overview of the DEM, with these outlines:

| Outline | Meaning |
|---|---|
| Image area (cyan) | The rendered area: the first layer's window, or the full first layer |
| Window (orange) | A layer's own window |
| Other layers' extents (magenta) | The full extent of each other layer, converted to this grid |

The outlines use the same latitude and longitude conversion as the light-curve command.
A site extent on the polar grid is therefore rotated.
The pointer readout gives the pixel, the latitude and longitude, the approximate elevation, and the matching pixel in each other layer.
**Image area** zooms to the rendered area.
On the first panel, the image area is a box with handles, as a selection in a paint program.
Drag a side or a corner to resize it, or the centre square to move it; a drag elsewhere pans.
The box stays inside the layer and snaps to whole pixels.
A box that covers the whole layer removes the window.

A background process makes each overview once and keeps it in the scratch directory.
A large DEM also gets a more detailed overview around the image area.

## The marker

The pink marker is one location on the terrain of the current settings.
A click on a terrain panel or on an image places it, and a right-click removes it.
**Marker > Set position...** places it at a latitude and longitude, at projected x and y in metres, or at a column and row of the first DEM.
The marker snaps to the nearest pixel centre, as the light-curve command does.

A marker placed on an image is a point of the current terrain only when the image has the current image area.
The light curve needs such a marker.
Otherwise, place the marker again on the Terrain view or with **Set position...**.

## Light curve

The Light curve tab uses the marker as its location and the terrain of the preset chosen in the Render tab.
It calculates the visible fraction of the solar disk from the start to the stop time, and the terrain horizon around the marker.
The result is a CSV in `data/outputs/light_curves/`, a metadata file, and the horizon profile; the [light-curve guide](light-curves.md) describes them.

The View shows the result as a horizon panorama above the visible fraction over time.
The panorama has bearing across, clockwise from north, and elevation up.
The brown area is the terrain below the horizon profile.
The Sun's path crosses it with the solar disk drawn to scale, coloured from grey (in shadow) to yellow (fully visible).
Elevation is stretched relative to bearing so that the horizon is readable; the label gives the factor.
Scroll to zoom the bearings and drag to pan; the panorama wraps around at north.
**Whole horizon** shows all bearings again.
Pointing at a time in the lower plot marks the Sun's position at that time, and a click on either plot keeps the mark.
Hyperion's CSV files give azimuth counterclockwise from east.

**Tools > Sun and Earth az/el** exports the Sun's and Earth's directions at a point without terrain, and shows both paths on the same panorama.
**Tools > Pixel probe** explains one image pixel at one time and writes its report to the Jobs window.

## PSR

The PSR tab maps permanently shadowed regions with the terrain and image area of the preset chosen in the Render tab.
It runs `scripts/generate_psr.jl`; the [PSR guide](psr.md) describes the method and the files.
The preset's times and observer height are not used.
The tab has its own study settings, with defaults 1970-01-01 through 2044-01-01 every 6 hours at ground level.
**Mapbuilder sunlight cutoff** counts exposures below 1/255 of the solar disk as shadow, as mapbuilder does.

**Find PSRs** writes to `data/outputs/<preset name>/psr/`, or `psr-mapbuilder/` with the cutoff on.
**Options** sets another output folder, the backend, and a sample limit for test runs.
A full run can take hours.
The progress bar and the count of pixels that are still unlit come from the checkpoint, which is saved about every minute.
**Stop** keeps the checkpoint, and **Resume** continues from it.
A folder that holds a run with other settings cannot be resumed; the tab names the differences and offers a new folder.
A changed terrain or a changed program also invalidates the checkpoint; the job then fails and the tab asks for another output folder.

When a run completes, the View shows the filtered mask (regions of at least 5 pixels) and the raw mask.
PSR pixels are red and pixels sunlit at some sample are black; NoData is transparent.
**Show** opens a completed result again. **Run again in ...** starts the same study in a new folder; **Replace...** deletes the result and checkpoint and runs again in the same folder.

## View

Scroll to zoom and drag to pan.
The pointer readout gives the image pixel, the first-DEM pixel, and the stored value of every loaded layer.
**Save image...** saves the image shown, including contact maps, comparisons, and statistics, as a GeoTIFF or a PNG.
A GeoTIFF keeps the values and takes its georeference from the image itself or from `other/hillshade.tif` of the same results.
A PNG of a sun or DSN image is a palette PNG; a PNG of another layer has the colours shown.
Plots, light curves, and azimuth and elevation results have **Open CSV** and **Show in Finder** buttons.

GeoTIFFs and palette PNGs give exact values.
RGB PNGs from earlier versions give only colours, so the readout omits their DSN values.

## Results

The Results tab opens any folder with `sun/` and `dsn/` subfolders, from the GUI, Hyperion, or mapbuilder.
It reads GeoTIFF and PNG images; when a time has both, it shows the GeoTIFF.
**Look in another folder** selects a folder other than `data/outputs/`.
It displays `other/hillshade.tif` beneath the maps when that file has the map size.

The analysis sections are folded until you open them.
In **Earth contact map**, the **DSN threshold** sets the Earth-contact limit in degrees.
The contact layer marks pixels at or above the limit.
The sun-and-contact layer has four classes: dark without contact, lit without contact, contact while dark, and lit with contact.
Lit means a nonzero Sun value.

**Compare** loads a second set of results at the same timestamp.
The difference layers need the same grid size.
The GUI does not resample between grids.

**Time series at a pixel** reads the marker's pixel from every frame and plots the Sun and DSN values.
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
A run at the end of the results lasts one more frame interval.
These two calculations decode every frame in a separate process.
Use **every Nth frame** for a faster estimate.
The statistics need DSN values: GeoTIFFs or palette PNGs.

## Scratch files

The GUI writes the current settings for runs, DEM overviews, time series, statistics, and the hash cache to `data/outputs/.hyperion_gui/`.
Saved images go to `data/outputs/saved_images/` unless you choose another file.
Git ignores that directory.
