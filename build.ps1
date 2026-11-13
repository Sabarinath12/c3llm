# Builds build\c3llm.exe. Pass -O3 for the fast build.
# Uses tools\c3\c3c.exe if present, otherwise c3c from PATH.
# Links with MSYS2 gcc so the Microsoft SDK is not needed.
# Targets AVX2 + FMA explicitly: c3c's --x86cpu=native only enabled AVX1 on a Zen 2 CPU.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$c3c = if (Test-Path tools\c3\c3c.exe) { 'tools\c3\c3c.exe' } else { 'c3c' }
& $c3c compile-only (Get-ChildItem src\*.c3).FullName --target mingw-x64 --single-module=yes --obj-out build\obj --x86cpu=avx2-v2 @args
if ($LASTEXITCODE) { exit $LASTEXITCODE }
gcc (Get-ChildItem build\obj\*.obj).FullName -o build\c3llm.exe -municode -ldbghelp
exit $LASTEXITCODE
