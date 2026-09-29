# Package a built npurun into a self-contained Windows zip (used by the release workflow).
#   scripts\package.ps1 -Version v0.1.0
# Run from a VS 2022 x64 developer prompt (VCToolsRedistDir must be set) at the repo root.
param(
    [string]$BuildDir = "build",
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$OutDir = "dist"
)
$ErrorActionPreference = "Stop"

$name = "npurun-$Version-win-x64"
$stage = Join-Path $OutDir $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

$bin = Join-Path $BuildDir "bin"
Get-ChildItem $bin -Filter *.exe | Where-Object { $_.Name -notlike "test-*" } | Copy-Item -Destination $stage
Get-ChildItem $bin -Filter *.dll | Copy-Item -Destination $stage

# Visual C++ runtime, deployed app-locally so no redistributable install is needed.
$redist = $env:VCToolsRedistDir
if (-not $redist) { throw "VCToolsRedistDir is not set: run from a VS 2022 x64 developer prompt" }
$crt = Get-ChildItem (Join-Path $redist "x64") -Directory -Filter "Microsoft.VC*.CRT" | Select-Object -First 1
foreach ($dll in "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll") {
    Copy-Item (Join-Path $crt.FullName $dll) $stage
}

Copy-Item scripts\npurun-chat.cmd, scripts\npurun-server.cmd $stage
Copy-Item LICENSE (Join-Path $stage "LICENSE.txt")
Copy-Item third_party\llama.cpp\LICENSE (Join-Path $stage "LICENSE-llama.cpp.txt")
Copy-Item THIRD_PARTY_NOTICES.md $stage
Copy-Item scripts\README.txt $stage

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Set-Content -Encoding ascii (Join-Path $OutDir "$name.zip.sha256")
Write-Host "$zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB) sha256 $hash"
