# cross_vendor_test.ps1
# ─────────────────────────────────────────────────────────────────────────
# Cross-vendor bit-exact verification on Windows + NVIDIA CUDA.
#
# Workflow (Mac → Windows → Mac):
#   1. Plug the WD_BLACK drive into a Windows machine with Julia + an
#      NVIDIA GPU + the CUDA.jl package available in the global Julia env.
#   2. Open a PowerShell window and `cd` into Hyperion.jl (on the
#      WD_BLACK drive's mount letter, e.g. D:\Hyperion.jl).
#   3. Run:    .\scripts\cross_vendor_test.ps1
#   4. Wait. The script will write everything it does to
#      data\outputs\cross_vendor_test_results.log and the per-timestamp
#      SHAs to data\outputs\bitexact\{cpu,cuda}\SHAs.txt.
#   5. Eject the drive cleanly, plug it back into the Mac, and tell
#      Claude to compare the SHAs against the pinned Mac values.
#
# What the script tests:
#   A) Pkg.test() — runs the full test/runtests.jl on Windows x86 CPU.
#      Includes the 20-timestamp LDEM bit-exact regression and the 1m
#      site DEM regression. PASS = Windows CPU Float32 produces output
#      byte-identical to the pinned Apple-Silicon CPU SHAs.
#   B) bitexact_test.jl HYP_BACKEND=cuda — runs the same 20 timestamps on
#      the NVIDIA GPU, writing per-stage SHAs and raw .bin buffers under
#      data\outputs\bitexact\cuda\. After bringing the drive back to the
#      Mac, `scripts/diff_bitexact_shas.jl` compares against the Mac
#      CPU SHAs to confirm CUDA produces identical output.
#   C) bitexact_test.jl HYP_BACKEND=cpu — same on Windows x86 CPU
#      (redundant with (A) but written in the same SHAs.txt format,
#      handy for direct CUDA-vs-CPU diffing on the same machine).
#
# What kernel state this verifies (commit at time of writing):
#   - The dz-cancellation fixes (commits bfda7f6, 533f9ed, facfe85).
#   - The mipmap-pool/skip/d-alignment fixes (commits b048496, 3a57940).
#   - The default `mipmap_base = 100` for the site driver (commit 9ecfb43).
#   - Re-pinned LDEM + 1m site SHAs (commit 122e31d).
# ─────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"

# ─── Locate project root and log file ─────────────────────────────────────
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

$LogFile = Join-Path $ProjectRoot "data\outputs\cross_vendor_test_results.log"
New-Item -ItemType Directory -Force (Split-Path $LogFile) | Out-Null

# ─── Logging helper ──────────────────────────────────────────────────────
# Single-writer design: every line goes through AppendLogLine, which uses
# [System.IO.File]::AppendAllText (open-write-close, UTF-8, no held handle).
# Avoids the lock contention + UTF-16/UTF-8 encoding mismatch we'd get from
# mixing Add-Content with Tee-Object -Append on the same file.
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false

function AppendLogLine($Line) {
    [System.IO.File]::AppendAllText($LogFile, [string]$Line + [Environment]::NewLine, $Utf8NoBom)
}

function Log($Msg) {
    $TimestampedLine = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $TimestampedLine
    AppendLogLine $TimestampedLine
}

function RunAndLog($Description, $ScriptBlock) {
    Log ""
    Log "================================================================"
    Log "STAGE: $Description"
    Log "================================================================"
    $StageStart = Get-Date

    # Stream each line through Write-Host + AppendLogLine so terminal and log
    # update in real time. ErrorRecord (from native stderr via 2>&1) is
    # stringified so it lands in the log as text, not as a thrown error.
    & $ScriptBlock 2>&1 | ForEach-Object {
        $line = if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $_.Exception.Message
        } else {
            [string]$_
        }
        Write-Host $line
        AppendLogLine $line
    }
    $ExitCode = $LASTEXITCODE

    $StageEnd = Get-Date
    Log "STAGE EXIT: $Description  (exit=$ExitCode, duration=$([int]($StageEnd - $StageStart).TotalSeconds)s)"
    return $ExitCode
}

# ─── Header ──────────────────────────────────────────────────────────────
Log "=================================================================="
Log "Cross-vendor bit-exact test"
Log "Project root: $ProjectRoot"
Log "Hostname:     $env:COMPUTERNAME"
Log "User:         $env:USERNAME"
Log "OS:           $((Get-CimInstance Win32_OperatingSystem).Caption)"
Log "PSVersion:    $($PSVersionTable.PSVersion)"
$JuliaVersion = & julia --version 2>&1 | Out-String
Log "Julia:        $($JuliaVersion.Trim())"
Log "=================================================================="

# ─── Pre-flight: LDEM file ───────────────────────────────────────────────
$LDEMTarget = Join-Path $ProjectRoot "data\inputs\ldem_80s_20m.img"
$LDEMExpectedSHA = "caaf017f6bd49cc96f8de1e2620de38931ec4733a5cf1bbfa2aa778d625b523b"

Log ""
Log "Pre-flight: ensuring LDEM is in place at $LDEMTarget"
if (-not (Test-Path $LDEMTarget) -or ((Get-FileHash $LDEMTarget -Algorithm SHA256).Hash.ToLower() -ne $LDEMExpectedSHA)) {
    Log "  ERROR: missing or wrong SHA. Place the Shirley LDEM at $LDEMTarget."
    Log "  Expected SHA: $LDEMExpectedSHA"
    exit 1
} else {
    Log "  Target present and SHA matches; skipping copy."
}

# ─── Pre-flight: site TIF ────────────────────────────────────────────────
$SiteTIF = Join-Path $ProjectRoot "data\inputs\nobile_1m.tif"
$SiteExpectedSHA = "e8cc7e5b530972d1d84083b335f961f0aa87e64c39697d942f10589930dd69f4"
if (Test-Path $SiteTIF) {
    $SiteActualSHA = (Get-FileHash $SiteTIF -Algorithm SHA256).Hash.ToLower()
    if ($SiteActualSHA -ne $SiteExpectedSHA) {
        Log "ERROR: site TIF SHA mismatch. Expected $SiteExpectedSHA, got $SiteActualSHA"
        exit 1
    }
    $env:HYPERION_SITE_TIF = $SiteTIF
    Log "Set HYPERION_SITE_TIF = $SiteTIF"
} else {
    Log "WARNING: site TIF not at $SiteTIF — site test will skip"
}

# ─── Stage 0: instantiate the project ────────────────────────────────────
$Exit0 = RunAndLog "Pkg.instantiate() — fetch project dependencies" {
    julia --project -e 'using Pkg; Pkg.instantiate()'
}

# ─── Stage A: full test suite (Windows CPU) ──────────────────────────────
$ExitA = RunAndLog "Pkg.test() — full regression on Windows CPU" {
    julia --project -e 'using Pkg; Pkg.test()'
}

# ─── Stage B: bitexact_test.jl with CUDA ─────────────────────────────────
$env:HYP_BACKEND = "cuda"
$ExitB = RunAndLog "bitexact_test.jl  HYP_BACKEND=cuda" {
    julia --project scripts/bitexact_test.jl
}

# ─── Stage C: bitexact_test.jl with CPU (Windows x86) ────────────────────
$env:HYP_BACKEND = "cpu"
$ExitC = RunAndLog "bitexact_test.jl  HYP_BACKEND=cpu" {
    julia --project scripts/bitexact_test.jl
}

# ─── Rename output dirs so they don't collide with Mac side runs ─────────
# bitexact_test.jl writes to data/outputs/bitexact/<backend>/. After the
# drive is plugged back into a Mac and Mac-side runs happen, those would
# overwrite these Windows results. Move to win_cpu/ + win_cuda/ now.
$BitexactBase = Join-Path $ProjectRoot "data\outputs\bitexact"
foreach ($pair in @(@("cpu", "win_cpu"), @("cuda", "win_cuda"))) {
    $src = Join-Path $BitexactBase $pair[0]
    $dst = Join-Path $BitexactBase $pair[1]
    if (Test-Path $src) {
        if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
        Move-Item $src $dst
        Log "Renamed $src → $dst"
    }
}

# ─── Summary ─────────────────────────────────────────────────────────────
Log ""
Log "=================================================================="
Log "FINAL SUMMARY"
Log "=================================================================="
Log "Stage 0 (Pkg.instantiate):                   exit=$Exit0  $(if ($Exit0 -eq 0) { 'OK' } else { 'FAIL' })"
Log "Stage A (Pkg.test on Windows CPU):           exit=$ExitA  $(if ($ExitA -eq 0) { 'PASS' } else { 'FAIL' })"
Log "Stage B (bitexact_test.jl with CUDA):        exit=$ExitB"
Log "Stage C (bitexact_test.jl with Windows CPU): exit=$ExitC"
Log ""
Log "Stage A=PASS means Windows CPU Float32 is byte-identical to the"
Log "pinned Apple Silicon CPU SHAs in test/bitexact.jl + test/site_1m.jl."
Log ""
Log "Stages B and C produced SHA tables at:"
Log "  data\outputs\bitexact\win_cuda\SHAs.txt   (NVIDIA CUDA)"
Log "  data\outputs\bitexact\win_cpu\SHAs.txt    (Windows x86 CPU)"
Log ""
Log "Bring the drive back to the Mac and run:"
Log "  julia --project scripts/diff_bitexact_shas.jl"
Log "to compare across all available backends."
Log ""
Log "Log file: $LogFile"
Log "=================================================================="

if ($Exit0 -ne 0 -or $ExitA -ne 0 -or $ExitB -ne 0 -or $ExitC -ne 0) {
    exit 1
}
exit 0
