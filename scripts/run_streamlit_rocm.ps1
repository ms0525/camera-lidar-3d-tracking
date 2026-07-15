<#
.SYNOPSIS
Runs the Streamlit tracking dashboard with the local AMD ROCm environment.

.DESCRIPTION
Validates the Python environment, KITTI Tracking sequence, model weights, and
dashboard entry point before starting Streamlit. It also discovers the MSVC and
ROCm Clang headers needed by MIOpen HIPRTC and verifies that PyTorch can access
the selected AMD GPU. Environment changes are scoped to this PowerShell process
and restored when Streamlit exits.

.EXAMPLE
.\scripts\run_streamlit_rocm.ps1 -DatasetRoot "<path-to-kitti-tracking>" -Sequence 0000
#>
# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DatasetRoot,
    [ValidatePattern("^\d{1,4}$")]
    [string]$Sequence = "0000",
    [string]$Yolo11Model = "",
    [string]$Yolo26Model = "",
    [string]$Device = "0",
    [ValidateSet(0, 1)]
    [int]$EmbedderGpu = 0,
    [string]$VisibleGpu = "0",
    [string]$CacheRoot = "",
    [string]$MsvcInclude = "",
    [string]$RocmClangInclude = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraStreamlitArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [ValidateSet("Leaf", "Container")]
        [string]$PathType
    )

    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) {
        throw "$Description was not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-ContainsFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory,
        [Parameter(Mandatory = $true)]
        [string]$Filter,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $match = Get-ChildItem -LiteralPath $Directory -File -Filter $Filter |
        Select-Object -First 1
    if (-not $match) {
        throw "$Description contains no $Filter files: $Directory"
    }
}

function Get-DefaultCacheRoot {
    $localAppData = [Environment]::GetFolderPath(
        [Environment+SpecialFolder]::LocalApplicationData
    )
    if (-not $localAppData) {
        $localAppData = $env:LOCALAPPDATA
    }
    if (-not $localAppData) {
        throw (
            "Could not determine the current user's LocalAppData directory. " +
            "Pass an explicit writable directory with -CacheRoot."
        )
    }
    return (Join-Path $localAppData "3D_Detection\ROCm")
}

function Find-MsvcInclude {
    $candidates = @()
    if ($env:VCToolsInstallDir) {
        $candidates += (Join-Path $env:VCToolsInstallDir "include")
    }

    $programFilesX86 = ${env:ProgramFiles(x86)}
    $toolRoots = @(
        (Join-Path $programFilesX86 "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
        (Join-Path $programFilesX86 "Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    )
    foreach ($toolRoot in $toolRoots) {
        if (Test-Path -LiteralPath $toolRoot -PathType Container) {
            $candidates += Get-ChildItem -LiteralPath $toolRoot -Directory |
                ForEach-Object { Join-Path $_.FullName "include" }
        }
    }

    $valid = $candidates |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "type_traits") -PathType Leaf } |
        Sort-Object {
            $versionText = Split-Path (Split-Path $_ -Parent) -Leaf
            try { [version]$versionText } catch { [version]"0.0" }
        } -Descending
    return $valid | Select-Object -First 1
}

$repoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$entryPoint = Resolve-RequiredPath `
    -Path (Join-Path $repoRoot "app\streamlit_app.py") `
    -Description "Streamlit dashboard entry point" `
    -PathType Leaf
if (-not $PythonPath) {
    $PythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
$PythonPath = Resolve-RequiredPath `
    -Path $PythonPath `
    -Description "ROCm Python" `
    -PathType Leaf
$DatasetRoot = Resolve-RequiredPath `
    -Path $DatasetRoot `
    -Description "KITTI Tracking dataset root" `
    -PathType Container

$Sequence = ([int]$Sequence).ToString("0000")

# KittiTrackingDataset accepts either the dataset root or the split directory.
$nestedTrainingRoot = Join-Path $DatasetRoot "training"
if (Test-Path -LiteralPath (Join-Path $nestedTrainingRoot "image_02") -PathType Container) {
    $splitRoot = $nestedTrainingRoot
}
elseif (Test-Path -LiteralPath (Join-Path $DatasetRoot "image_02") -PathType Container) {
    $splitRoot = $DatasetRoot
}
else {
    throw (
        "Could not find a KITTI Tracking training split below $DatasetRoot. " +
        "Expected training\image_02 or image_02."
    )
}

$imageDirectory = Resolve-RequiredPath `
    -Path (Join-Path $splitRoot "image_02\$Sequence") `
    -Description "KITTI image sequence" `
    -PathType Container
$pointCloudDirectory = Resolve-RequiredPath `
    -Path (Join-Path $splitRoot "velodyne\$Sequence") `
    -Description "KITTI Velodyne sequence" `
    -PathType Container
$null = Resolve-RequiredPath `
    -Path (Join-Path $splitRoot "calib\$Sequence.txt") `
    -Description "KITTI calibration file" `
    -PathType Leaf
Assert-ContainsFile -Directory $imageDirectory -Filter "*.png" -Description "KITTI image sequence"
Assert-ContainsFile -Directory $pointCloudDirectory -Filter "*.bin" -Description "KITTI Velodyne sequence"

if (-not $Yolo11Model) {
    $Yolo11Model = Join-Path $repoRoot "yolo11s.pt"
}
if (-not $Yolo26Model) {
    $Yolo26Model = Join-Path $repoRoot "yolo26s.pt"
}
$Yolo11Model = Resolve-RequiredPath `
    -Path $Yolo11Model `
    -Description "YOLO11 model" `
    -PathType Leaf
$Yolo26Model = Resolve-RequiredPath `
    -Path $Yolo26Model `
    -Description "YOLO26 model" `
    -PathType Leaf

if (-not $MsvcInclude) {
    $MsvcInclude = Find-MsvcInclude
}
if (-not $MsvcInclude -or -not (Test-Path -LiteralPath (Join-Path $MsvcInclude "type_traits") -PathType Leaf)) {
    throw (
        "No complete MSVC C++ include directory containing type_traits was found. " +
        "Install the Visual Studio 2022 v143 x64/x86 build tools or pass -MsvcInclude."
    )
}
$MsvcInclude = (Resolve-Path -LiteralPath $MsvcInclude).Path

if (-not $RocmClangInclude) {
    $sitePackages = (& $PythonPath -c "import sysconfig; print(sysconfig.get_path('purelib'))").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $sitePackages) {
        throw "Could not locate site-packages from the ROCm Python environment."
    }
    $clangRoot = Join-Path $sitePackages "_rocm_sdk_core\lib\llvm\lib\clang"
    if (Test-Path -LiteralPath $clangRoot -PathType Container) {
        $RocmClangInclude = Get-ChildItem -LiteralPath $clangRoot -Directory |
            Where-Object {
                Test-Path -LiteralPath (Join-Path $_.FullName "include\stddef.h") -PathType Leaf
            } |
            Sort-Object {
                try { [version]$_.Name } catch { [version]"0.0" }
            } -Descending |
            Select-Object -First 1 |
            ForEach-Object { Join-Path $_.FullName "include" }
    }
}
if (-not $RocmClangInclude -or -not (Test-Path -LiteralPath (Join-Path $RocmClangInclude "stddef.h") -PathType Leaf)) {
    throw (
        "No ROCm Clang resource include directory containing stddef.h was found. " +
        "Verify the ROCm SDK installation or pass -RocmClangInclude."
    )
}
$RocmClangInclude = (Resolve-Path -LiteralPath $RocmClangInclude).Path

if (-not $CacheRoot) {
    $CacheRoot = Get-DefaultCacheRoot
}
$tempRoot = Join-Path $CacheRoot "temp"
foreach ($directory in @($CacheRoot, $tempRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
$CacheRoot = (Resolve-Path -LiteralPath $CacheRoot).Path
$tempRoot = (Resolve-Path -LiteralPath $tempRoot).Path

$streamlitProbe = & $PythonPath -c "import streamlit; print(streamlit.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit could not be imported from the selected Python environment."
}
$null = & $PythonPath -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pkg_resources') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw (
        "Deep SORT requires pkg_resources, which is absent from this environment. " +
        "Install the pinned local dependencies with: python -m pip install " +
        "-r requirements.txt -r app\requirements.txt"
    )
}
$existingInclude = $env:INCLUDE
$includeParts = @($MsvcInclude, $RocmClangInclude)
if ($existingInclude) {
    $includeParts += $existingInclude
}

$processEnvironment = @{
    "DASHBOARD_ENABLE_LIVE" = "1"
    "DASHBOARD_TRUSTED_LOCAL" = "1"
    "DASHBOARD_DATASET_ROOT" = $DatasetRoot
    "DASHBOARD_SEQUENCE" = $Sequence
    "DASHBOARD_YOLO11_MODEL" = $Yolo11Model
    "DASHBOARD_YOLO26_MODEL" = $Yolo26Model
    "DASHBOARD_DEVICE" = $Device
    "DASHBOARD_EMBEDDER_GPU" = $EmbedderGpu.ToString()
    "INCLUDE" = $includeParts -join ";"
    "HIP_VISIBLE_DEVICES" = $VisibleGpu
    "TEMP" = $tempRoot
    "TMP" = $tempRoot
    "TORCH_HOME" = Join-Path $CacheRoot "torch"
    "YOLO_CONFIG_DIR" = Join-Path $CacheRoot "ultralytics"
    "PYTHONUTF8" = "1"
}
$originalEnvironment = @{}
foreach ($name in $processEnvironment.Keys) {
    $originalEnvironment[$name] =
        [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    foreach ($name in $processEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $processEnvironment[$name],
            "Process"
        )
    }

    $probeCode = @"
import json
import torch
print(json.dumps({
    'torch': torch.__version__,
    'hip': torch.version.hip,
    'available': torch.cuda.is_available(),
    'count': torch.cuda.device_count(),
    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
"@
    $probeOutput = & $PythonPath -c $probeCode
    if ($LASTEXITCODE -ne 0) {
        throw "The ROCm PyTorch preflight failed."
    }
    $probe = $probeOutput | ConvertFrom-Json
    if (-not $probe.hip) {
        throw "This Python environment is not a ROCm build of PyTorch (torch.version.hip is empty)."
    }
    if (-not $probe.available -or $probe.count -lt 1) {
        throw "PyTorch cannot access the selected AMD GPU through ROCm."
    }

    Write-Host "ROCm preflight passed: torch $($probe.torch), HIP $($probe.hip)"
    Write-Host "Visible accelerator: $($probe.device)"
    Write-Host "MIOpen C++ includes: $MsvcInclude"
    Write-Host "MIOpen Clang includes: $RocmClangInclude"
    Write-Host "Streamlit $($streamlitProbe.Trim())"
    Write-Host "KITTI sequence: $Sequence"
    Write-Host "Dashboard device: $Device"
    Write-Host "DeepSORT embedder GPU: $EmbedderGpu"
    Write-Host "Dashboard URL: http://localhost:$Port"

    $streamlitArguments = @(
        "-m", "streamlit", "run", $entryPoint,
        "--server.port", $Port,
        "--server.address", "127.0.0.1"
    )
    if ($ExtraStreamlitArguments) {
        $streamlitArguments += $ExtraStreamlitArguments
    }

    Push-Location $repoRoot
    try {
        & $PythonPath @streamlitArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Streamlit exited with code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    foreach ($name in $processEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $originalEnvironment[$name],
            "Process"
        )
    }
}
