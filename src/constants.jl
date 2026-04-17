# ─── Pipeline constants ────────────────────────────────────────────────────

const MOON_RADIUS_KM = 1737.4
const MOON_RADIUS_M  = 1737400.0

const HORIZON_SAMPLES         = 1440          # 360° × 4 bins/degree
const NEAR_HORIZON_OVERSAMPLE = 3
const NEAR_FIELD_RAY_COUNT    = HORIZON_SAMPLES * NEAR_HORIZON_OVERSAMPLE  # 4320
const NEAR_FIELD_RAY_STEP     = Float32(0.70710698)
const RAY_CAST_DISTANCE_PIXELS = 230
const PATCH_SIZE              = 128

const F32_PI     = Float32(3.141592653589)
const F32_TWO_PI = Float32(2.0) * F32_PI
