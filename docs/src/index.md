# Hyperion.jl

Hyperion renders lunar illumination and Earth-visibility maps from DEMs and
SPICE ephemerides.

The main workflow is **mapset generation**: define terrain layers, choose a
time range or explicit timestamps, and write reproducible Sun/DSN PNG frames
with a manifest.

## Core Workflows

- Generate the 20 m Shirley Nobile extent.
- Generate VIPER 8.0 1 m maps with the Shirley 20 m farfield.
- Run correctness tests against the Tier 0 LROC NAC fixture.
- Audit bit-exactness across CPU, Metal, and CUDA.

## First Commands

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
```

Generate a single 20 m Nobile frame:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/nobile_20m_shirley.toml \
  --backend=auto --overwrite
```

Generate a VIPER 8.0 + Shirley mapset:

```bash
julia --project scripts/generate_mapset.jl \
  --spec=data/inputs/mapsets/viper8_shirley_range.toml \
  --backend=auto
```
