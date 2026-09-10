# Remote and working version comparison: September 10, 2026

The targeted checks found identical output between the remote version and the working version for the cases below.
They do not prove that every workflow or input has identical behavior.
The full suite was stopped at the user's request.

## Versions and environment

| Item | Value |
|---|---|
| Remote | `origin/master`, commit `da3132c4b61154ed8f5b2130d705d4896f9863a3` |
| Local parent | Commit `2bb96cd` with the pending cleanup and diagnostic files |
| Julia | 1.11.5 |
| KernelAbstractions | 0.9.41 |
| Metal | 1.9.3 |

The two checkouts used the same dependency manifest and terrain files.
The remote branch was checked against GitHub on the comparison date.
The subsequent documentation edits do not change the renderer calculation.

## Completed checks

| Check | Result for the two versions |
|---|---|
| Math and projection tests | 20 assertions passed |
| Synthetic CPU terrain tests | 47 assertions passed |
| Synthetic Metal terrain tests | 47 assertions passed |
| Small Metal comparison harness | All 120 output hashes matched |
| Selected mapset workflows | All 28 output files matched after path normalization in manifests |
| Continuation with existing frames | Frame bytes and modification times remained unchanged |
| Azimuth/elevation CSV | Hourly and 25-observation timestamp outputs matched |
| Entry-script help and radius dry run | Completed with matching relevant results |

### The 30 small Metal renders

The comparison harness was separate from the committed test suite.
It rendered 8 × 8 pixel synthetic windows for five configurations:
polar, site, two layers, three layers, and explicit geometry.
Each configuration used three Sun elevations: −3°, 1°, and 6°.
Each elevation used observer heights of 0 m and 0.5 m.

Thus, each version produced 30 renders.
Each render produced four checked outputs: Sun, DSN, Earth debug, and all eight Sun-ray debug channels.
All 120 hashes matched between versions.
This check did not use a large external DEM.

### Mapset workflow checks

The selected workflows covered a small polar range, explicit timestamps, 16 × 16 tiles, and a 1 m site cutoff window.
The site check used a 32 × 32 output and a 0.5 m observer height.
These checks exercised command parsing, output files, and resume behavior.
They did not cover every mapset specification or full terrain extent.

## Incomplete checks

The remote full-suite run completed math, projection, mipmap, and all 20 m reference cases before cancellation.
It also completed the initial 1 m checks and four 1 m reference cases.
The working version did not complete the full suite.
Neither version completed the observational correctness sweep in this comparison.
CUDA was not tested in this comparison.

## Known interface changes

The cleanup removes preset CLI options, the external correctness-tier interface, and obsolete renderer helpers.
The removed Barker/VIPER fixture had an empty image-hash table.
The `v8_medium_barker_sep.toml` time range and output name also changed.
These interface and configuration changes can change a command's behavior despite matching kernel output in the targeted tests.

## Local evidence

The local report, harness, CSV results, and logs are in:

```text
data/outputs/validation/remote-vs-working-2026-09-10/
```

This directory is ignored by Git and is not included in a normal clone.
The committed synthetic terrain tests remain available in `test/terrain_stack.jl`.
To do the exact 30-render comparison again, transfer the archived harness and its environment details.

## Documentation update checks

The documentation update passed the HTML build and the local Markdown link checks.
The small CPU terrain tests passed all 47 assertions again.
A separate smoke check passed 16 assertions for the documented API, bundled observation metadata, mapset TOML files, and diagnostic source syntax.
The presentation passed ZIP integrity and XML syntax checks.
The full suite was not started again.
