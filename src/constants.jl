# ─── Pipeline constants ────────────────────────────────────────────────────

const MOON_RADIUS_KM = 1737.4
const MOON_RADIUS_M  = 1737400.0

# LDEM 80s (20m) pixel grid anchor. 30400×30400 pixels, center at (15199.5, 15199.5).
const LDEM_S0       = 15199.5        # column corresponding to 0° easting
const LDEM_L0       = 15199.5        # row corresponding to 0° northing

const F32_PI     = Float32(3.141592653589)
const F32_RAD2DEG = Float32(180.0) / F32_PI  # derived from hardcoded F32_PI, not Julia's π

# Sun angular radius (deg). Used by the live shadow kernel.
const SUN_HALF_ANGLE_DEG = Float32(0.27)

# SPICE body NAIF IDs.
const NAIF_SUN   = 10
const NAIF_EARTH = 399
const NAIF_MOON  = 301
