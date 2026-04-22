# ─── Pipeline constants ────────────────────────────────────────────────────

const MOON_RADIUS_KM = 1737.4
const MOON_RADIUS_M  = 1737400.0

# LDEM 80s (20m) pixel grid anchor. 30400×30400 pixels, center at (15199.5, 15199.5).
const LDEM_S0       = 15199.5        # column corresponding to 0° easting
const LDEM_L0       = 15199.5        # row corresponding to 0° northing
const LDEM_SCALE_KM = 20.0 / 1000.0  # km per pixel

const HORIZON_SAMPLES         = 1440          # 360° × 4 bins/degree
const NEAR_HORIZON_OVERSAMPLE = 3
const NEAR_FIELD_RAY_COUNT    = HORIZON_SAMPLES * NEAR_HORIZON_OVERSAMPLE  # 4320
const NEAR_FIELD_RAY_STEP     = Float32(0.70710698)
const RAY_CAST_DISTANCE_PIXELS = 230
const PATCH_SIZE              = 128

const F32_PI     = Float32(3.141592653589)
const F32_TWO_PI = Float32(2.0) * F32_PI
const F32_RAD2DEG = Float32(180.0) / F32_PI  # derived from hardcoded F32_PI, not Julia's π

# Sun angular radius (deg). Used by the live shadow kernel.
const SUN_HALF_ANGLE_DEG = Float32(0.27)

# SPICE body NAIF IDs.
const NAIF_SUN   = 10
const NAIF_EARTH = 399
const NAIF_MOON  = 301
