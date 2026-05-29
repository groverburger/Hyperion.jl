# windows_cuda_unified_stack.ps1
# -----------------------------------------------------------------------------
# Windows + NVIDIA CUDA validation driver for the unified terrain-stack branch.
#
# Run from PowerShell on the Windows machine:
#
#   cd D:\Hyperion.jl
#   .\tools\bitexact\windows_cuda_unified_stack.ps1
#
# Optional:
#
#   .\tools\bitexact\windows_cuda_unified_stack.ps1 -SkipFullPkgTest
#   .\tools\bitexact\windows_cuda_unified_stack.ps1 -SkipTier0Products
#   .\tools\bitexact\windows_cuda_unified_stack.ps1 -SiteTif "E:\data\nobile_1m.tif"
#
# What this does:
#   1. Verifies/pulls the unified-terrain-kernel branch.
#   2. Verifies the Shirley LDEM at data\inputs\ldem_80s_20m.img.
#   3. Sets HYPERION_SITE_TIF for the 1 m Nobile DEM.
#   4. Instantiates project deps and verifies CUDA.jl is functional.
#   5. Runs targeted CUDA tests for the terrain stack, 1 m site path,
#      and forensic bit-exact harness.
#   6. Optionally runs full Pkg.test() with HYP_BACKEND=cuda.
#   7. Optionally generates Tier 0 baseline products:
#        data\outputs\baseline_maps_win\tier0\20m_lnsi
#        data\outputs\baseline_maps_win\tier0\1m_full_farfield
#
# Notes:
#   - GeoTIFF file bytes can differ across GDAL builds because of metadata.
#     Compare decoded raster bands/raw pixel buffers for bit-exactness.
#   - CUDA.jl is intentionally not in this project. Install it in the default
#     Julia environment before or during this script's CUDA check.
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Branch = "unified-terrain-kernel",
    [string]$Remote = "origin",
    [string]$SiteTif = "",
    [string]$LdemSource = "",
    [string]$Tier0Out = "data\outputs\baseline_maps_win\tier0",
    [switch]$NoPull,
    [switch]$SkipFullPkgTest,
    [switch]$SkipTier0Products,
    [switch]$SkipBitexactHarness
)

$ErrorActionPreference = "Stop"

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "data\outputs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$RunId = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"
$LogFile = Join-Path $LogDir "windows_cuda_unified_stack_$RunId.log"

function Append-LogLine([string]$Line) {
    [System.IO.File]::AppendAllText($LogFile, $Line + [Environment]::NewLine, $Utf8NoBom)
}

function Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    Append-LogLine $line
}

function Run-Step([string]$Name, [scriptblock]$Body) {
    Log ""
    Log "================================================================"
    Log "STAGE: $Name"
    Log "================================================================"
    $start = Get-Date
    try {
        & $Body 2>&1 | ForEach-Object {
            $line = if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else {
                [string]$_
            }
            Write-Host $line
            Append-LogLine $line
        }
        $exit = if ($LASTEXITCODE -is [int]) { $LASTEXITCODE } else { 0 }
        $duration = [int]((Get-Date) - $start).TotalSeconds
        Log "STAGE EXIT: $Name  (exit=$exit, duration=${duration}s)"
        if ($exit -ne 0) {
            throw "Stage failed: $Name (exit=$exit)"
        }
    } catch {
        $duration = [int]((Get-Date) - $start).TotalSeconds
        Log "STAGE FAILED: $Name  (duration=${duration}s)"
        throw
    }
}

function Require-Command([string]$Command) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command not found on PATH: $Command"
    }
}

function Resolve-DefaultSiteTif() {
    if ($SiteTif -ne "") {
        return $SiteTif
    }
    return Join-Path $ProjectRoot "data\inputs\nobile_1m.tif"
}

function Resolve-DefaultLdemSource() {
    if ($LdemSource -ne "") {
        return $LdemSource
    }
    return ""
}

function Ensure-Ldem() {
    $target = Join-Path $ProjectRoot "data\inputs\ldem_80s_20m.img"
    $source = Resolve-DefaultLdemSource
    $expectedSha = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null

    Log "Checking Shirley LDEM at $target"
    if (Test-Path $target) {
        $actual = (Get-FileHash $target -Algorithm SHA256).Hash.ToLower()
        if ($actual -eq $expectedSha) {
            Log "  LDEM present and SHA matches."
            return
        }
        Log "  LDEM SHA mismatch at target."
        Log "  expected: $expectedSha"
        Log "  actual:   $actual"
    } else {
        Log "  LDEM target missing."
    }

    if ($source -eq "" -or -not (Test-Path $source)) {
        throw "Shirley LDEM is not available at $target. Put the file under data\inputs or pass -LdemSource to copy it there."
    }

    Log "Copying Shirley LDEM from $source"
    Copy-Item $source $target -Force
    $copiedSha = (Get-FileHash $target -Algorithm SHA256).Hash.ToLower()
    if ($copiedSha -ne $expectedSha) {
        throw "Copied LDEM SHA mismatch. Expected $expectedSha, got $copiedSha"
    }
    Log "  Copy verified."
}

function Ensure-SiteTif() {
    $path = Resolve-DefaultSiteTif
    $expectedSha = "e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4"
    if (-not (Test-Path $path)) {
        throw "Site TIF not found at $path. Pass -SiteTif `"C:\path\to\nobile_1m.tif`"."
    }
    $resolved = (Resolve-Path $path).Path
    $actualSha = (Get-FileHash $resolved -Algorithm SHA256).Hash.ToLower()
    if ($actualSha -ne $expectedSha) {
        throw "Site TIF SHA mismatch. Expected $expectedSha, got $actualSha at $resolved"
    }
    $env:HYPERION_SITE_TIF = $resolved
    Log "Set HYPERION_SITE_TIF=$resolved"
}

Log "=================================================================="
Log "Windows CUDA unified terrain-stack validation"
Log "Project root: $ProjectRoot"
Log "Log file:     $LogFile"
Log "Hostname:     $env:COMPUTERNAME"
Log "User:         $env:USERNAME"
Log "PSVersion:    $($PSVersionTable.PSVersion)"
Log "=================================================================="

Require-Command "git"
Require-Command "julia"

Run-Step "Julia version" {
    julia --version
}

Run-Step "Checkout and update branch" {
    git fetch $Remote
    git checkout $Branch
    if (-not $NoPull) {
        git pull --ff-only $Remote $Branch
    }
    git status --short --branch
    git log --oneline -5
}

Ensure-Ldem
Ensure-SiteTif

$env:HYP_BACKEND = "cuda"
Remove-Item Env:\HYP_SKIP_CORRECTNESS -ErrorAction SilentlyContinue

Run-Step "Pkg.instantiate" {
    julia --project -e 'using Pkg; Pkg.instantiate()'
}

Run-Step "CUDA.jl availability" {
    julia -e 'using Pkg; try; using CUDA; catch; Pkg.add("CUDA"); using CUDA; end; CUDA.versioninfo(); @assert CUDA.functional()'
}

Run-Step "CUDA targeted terrain stack tests" {
    julia --project test/terrain_stack.jl
}

if (-not $SkipBitexactHarness) {
    Run-Step "CUDA forensic bitexact harness" {
        julia --project tools/bitexact/bitexact_test.jl
    }
} else {
    Log "Skipping tools/bitexact/bitexact_test.jl because -SkipBitexactHarness was supplied."
}

if (-not $SkipFullPkgTest) {
    Run-Step "Full Pkg.test with CUDA" {
        julia --project -e 'using Pkg; Pkg.test()'
    }
} else {
    Log "Skipping full Pkg.test() because -SkipFullPkgTest was supplied."
}

if (-not $SkipTier0Products) {
    Run-Step "Generate Tier 0 20m LNSI + full 1m farfield products" {
        julia --project tools/fixtures/generate_tier0_baseline_maps.jl "--out=$Tier0Out"
    }

    Run-Step "Verify generated Tier 0 product dimensions" {
        $env:HYP_TIER0_OUT = $Tier0Out
        $VerifyCode = @'
using ArchGDAL

root = ENV["HYP_TIER0_OUT"]
checks = [
    ("20m", joinpath(root, "20m_lnsi"), 896, 896),
    ("1m", joinpath(root, "1m_full_farfield"), 4992, 4096),
]

for (label, dir, expected_w, expected_h) in checks
    files = filter(f -> endswith(f, ".tif") && !startswith(basename(f), "._"),
                   readdir(dir; join = true))
    bad = String[]
    for f in files
        ds = ArchGDAL.read(f)
        w = ArchGDAL.width(ds)
        h = ArchGDAL.height(ds)
        if w != expected_w || h != expected_h
            push!(bad, "$(basename(f)):$(w)x$(h)")
        end
    end
    println(label, " tif_count=", length(files), " bad_dims=", length(bad))
    for entry in bad
        println("  ", entry)
    end
    @assert length(files) == 50
    @assert isempty(bad)
end
'@
        julia --project -e $VerifyCode
    }
} else {
    Log "Skipping Tier 0 product generation because -SkipTier0Products was supplied."
}

Log ""
Log "=================================================================="
Log "DONE"
Log "=================================================================="
Log "Branch:       $(git rev-parse --abbrev-ref HEAD)"
Log "Commit:       $(git rev-parse HEAD)"
Log "HYP_BACKEND:  $env:HYP_BACKEND"
Log "Site TIF:     $env:HYPERION_SITE_TIF"
Log "Tier 0 out:   $Tier0Out"
Log "Log file:     $LogFile"
Log ""
Log "When comparing GeoTIFF outputs across platforms, compare decoded raster"
Log "bands rather than .tif file bytes; GDAL metadata/sidecars may differ."
