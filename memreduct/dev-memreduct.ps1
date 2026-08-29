# dev-memreduct.ps1 — Build memreduct-viewstage.exe for cargo tauri dev
# Usage: cd memreduct; .\dev-memreduct.ps1 [-Arch x64|Win32]

param([string]$Arch = 'x64')

$ErrorActionPreference = 'Stop'
$memreductDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$archDir = if ($Arch -eq 'x64') { '64' } elseif ($Arch -eq 'Win32') { '32' } else { throw "Unsupported architecture: $Arch" }
$outExe = Join-Path $memreductDir "bin\$archDir\memreduct-viewstage.exe"

# Locate vcvarsall.bat — prefer vswhere, fallback to known paths
$vcvarsall = $null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -requires Microsoft.Component.MSBuild -property installationPath
    if ($vsPath) {
        $candidate = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
        if (Test-Path $candidate) { $vcvarsall = $candidate }
    }
}

if (-not $vcvarsall) {
    $vsEditions = @('Community', 'Professional', 'Enterprise')
    $vsVersions = @(@{ Major = 18 }, @{ Major = 17 })
    foreach ($ver in $vsVersions) {
        foreach ($ed in $vsEditions) {
            foreach ($pf in @("${env:ProgramFiles}", "${env:ProgramFiles(x86)}")) {
                $candidate = "$pf\Microsoft Visual Studio\$($ver.Major)\$ed\VC\Auxiliary\Build\vcvarsall.bat"
                if (Test-Path $candidate) { $vcvarsall = $candidate; break }
            }
            if ($vcvarsall) { break }
        }
        if ($vcvarsall) { break }
    }
}

if (-not $vcvarsall) {
    Write-Host '[ERROR] Visual Studio not found. Install VS with "Desktop development with C++" workload.' -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] vcvarsall: $vcvarsall" -ForegroundColor Cyan
Write-Host "[INFO] Building $Arch..." -ForegroundColor Cyan

$msbuildLog = Join-Path $memreductDir "msbuild_log.txt"

$msbuildCmd = @"
call "$vcvarsall" $Arch >nul 2>&1
msbuild "$memreductDir\memreduct.vcxproj" -property:Configuration=Release -property:Platform=$Arch -verbosity:minimal > "$msbuildLog" 2>&1
if errorlevel 1 exit /b 1
"@

cmd /c $msbuildCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] Build failed; see msbuild_log.txt' -ForegroundColor Red
    exit 1
}

if (Test-Path $outExe) {
    $size = (Get-Item $outExe).Length / 1KB
    Write-Host ("[OK] {0}  ({1:N0} KB)" -f $outExe, $size) -ForegroundColor Green
} else {
    Write-Host ("[ERROR] Build succeeded but {0} not found" -f $outExe) -ForegroundColor Red
    exit 1
}
