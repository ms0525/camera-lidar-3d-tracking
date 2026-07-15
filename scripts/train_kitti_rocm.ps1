<#
.SYNOPSIS
Runs KITTI detector training with the verified native-Windows AMD ROCm setup.

.DESCRIPTION
ROCm uses PyTorch's torch.cuda API, so Ultralytics still receives --device 0.
This launcher keeps the selected AMD GPU visible, discovers the installed MSVC and ROCm
Clang headers needed by MIOpen HIPRTC, validates the backend, and then delegates
to train_kitti_detector.py. All environment changes are process-local.
#>
# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DatasetRoot,
    [string]$Project = "",
    [string]$CacheRoot = "",
    [ValidateNotNullOrEmpty()]
    [string]$Model = "yolo26s.pt",
    [ValidateRange(1, 100000)]
    [int]$Epochs = 100,
    [ValidateRange(1, 10000)]
    [int]$ImageSize = 640,
    [ValidateRange(1, 100000)]
    [int]$Batch = 2,
    [ValidateRange(0, 100000)]
    [int]$Workers = 0,
    [ValidateRange(0, 100000)]
    [int]$Patience = 50,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 0,
    [string]$Name = "",
    [string]$VisibleGpu = "0",
    [string]$Device = "0",
    [string]$MsvcInclude = "",
    [string]$RocmClangInclude = "",
    [switch]$AllowCustomDatasetSize,
    [switch]$Resume,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
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
if (-not $PythonPath) {
    $PythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not $Project) {
    $Project = Join-Path $repoRoot "runs"
}
if (-not $CacheRoot) {
    $CacheRoot = Get-DefaultCacheRoot
}
if (-not $Name) {
    $modelStem = [System.IO.Path]::GetFileNameWithoutExtension($Model)
    $Name = "kitti_${modelStem}_rocm"
}
$trainerPath = Resolve-RequiredFile `
    -Path (Join-Path $repoRoot "train_kitti_detector.py") `
    -Description "Training script"
$PythonPath = Resolve-RequiredFile -Path $PythonPath -Description "ROCm Python"
if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    throw "KITTI dataset root was not found: $DatasetRoot"
}
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
foreach ($relativePath in @("images\train", "images\val", "labels\train", "labels\val")) {
    $requiredDirectory = Join-Path $DatasetRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "KITTI dataset directory was not found: $requiredDirectory"
    }
}

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

$tempRoot = Join-Path $CacheRoot "temp"
foreach ($directory in @($Project, $CacheRoot, $tempRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
$Project = (Resolve-Path -LiteralPath $Project).Path
$CacheRoot = (Resolve-Path -LiteralPath $CacheRoot).Path
$tempRoot = (Resolve-Path -LiteralPath $tempRoot).Path

# Preserve Ultralytics model aliases while making an explicit local path stable
# across the later Push-Location into the repository root.
if (Test-Path -LiteralPath $Model -PathType Leaf) {
    $Model = (Resolve-Path -LiteralPath $Model).Path
}

$existingInclude = $env:INCLUDE
$includeParts = @($MsvcInclude, $RocmClangInclude)
if ($existingInclude) {
    $includeParts += $existingInclude
}

$environmentNames = @(
    "INCLUDE",
    "HIP_VISIBLE_DEVICES",
    "TEMP",
    "TMP",
    "TORCH_HOME",
    "YOLO_CONFIG_DIR"
)
$originalEnvironment = @{}
foreach ($environmentName in $environmentNames) {
    $originalEnvironment[$environmentName] =
        [Environment]::GetEnvironmentVariable($environmentName, "Process")
}

try {
$env:INCLUDE = $includeParts -join ";"
$env:HIP_VISIBLE_DEVICES = $VisibleGpu
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:YOLO_CONFIG_DIR = Join-Path $CacheRoot "ultralytics"

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

$trainingArguments = @(
    $trainerPath,
    "--model", $Model,
    "--dataset-root", $DatasetRoot,
    "--epochs", $Epochs,
    "--imgsz", $ImageSize,
    "--batch", $Batch,
    "--device", $Device,
    "--workers", $Workers,
    "--patience", $Patience,
    "--seed", $Seed,
    "--project", $Project,
    "--name", $Name
)
if ($AllowCustomDatasetSize) {
    $trainingArguments += "--allow-custom-dataset-size"
}
if ($Resume) {
    $trainingArguments += "--resume"
}
if ($DryRun) {
    $trainingArguments += "--dry-run"
}
if ($ExtraArguments) {
    $trainingArguments += $ExtraArguments
}

Push-Location $repoRoot
try {
    & $PythonPath @trainingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "KITTI detector training exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
}
finally {
    foreach ($environmentName in $environmentNames) {
        [Environment]::SetEnvironmentVariable(
            $environmentName,
            $originalEnvironment[$environmentName],
            "Process"
        )
    }
}
