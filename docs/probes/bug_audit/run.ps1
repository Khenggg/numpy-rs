param([switch]$IncludeArtifacts)
$ErrorActionPreference = 'Stop'
$auditRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Push-Location $auditRoot
try {
    $auditOutput = Join-Path $auditRoot 'target/bug-audit'
    New-Item -ItemType Directory -Force -Path $auditOutput | Out-Null
    $exeExt = if ($IsWindows -or $env:OS -eq 'Windows_NT') { '.exe' } else { '' }
    $auditTypes = Get-ChildItem target/release/deps -Filter 'libnumpy_types*.rlib' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    if (!$auditTypes) { throw 'Build numpy-types in release mode first.' }
    foreach ($auditName in @('numeric_fixes','zero_shape','complex_order','histogram_boundary')) {
        $auditSource = Join-Path $PSScriptRoot ($auditName + '.rs')
        $auditExecutable = Join-Path $auditOutput ($auditName + $exeExt)
        & rustc --edition=2024 -Awarnings -O -C lto=thin -C overflow-checks=yes $auditSource `
            --extern "numpy_types=$auditTypes" -L dependency=target/release/deps -o $auditExecutable
        if ($LASTEXITCODE -ne 0) { throw "Failed compiling $auditName" }
        & $auditExecutable
        if ($LASTEXITCODE -ne 0) { throw "Failed probe $auditName" }
    }
    if ($IncludeArtifacts) {
        foreach ($auditProbe in @(@('io_polynomial','numpy_lib'),@('masked','numpy_extra'))) {
            $auditName = $auditProbe[0]; $auditCrate = $auditProbe[1]
            $auditLibrary = Get-ChildItem target/debug/deps -Filter ('lib' + $auditCrate + '*.rlib') |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
            if (!$auditLibrary) { throw "Missing debug artifact for $auditCrate" }
            $auditExecutable = Join-Path $auditOutput ($auditName + $exeExt)
            & rustc --edition=2024 -Awarnings (Join-Path $PSScriptRoot ($auditName + '.rs')) `
                --extern ($auditCrate + '=' + $auditLibrary) -L dependency=target/debug/deps -o $auditExecutable
            if ($LASTEXITCODE -ne 0) { throw "Failed compiling $auditName; rebuild owning crate first" }
            & $auditExecutable
            if ($LASTEXITCODE -ne 0) { throw "Failed probe $auditName" }
        }
    }
    $pythonExe = if ($env:NUMPY_RS_PYTHON) {
        $env:NUMPY_RS_PYTHON
    } elseif (Test-Path '.venv/Scripts/python.exe') {
        '.venv/Scripts/python.exe'
    } elseif (Test-Path '.venv/bin/python') {
        '.venv/bin/python'
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        'python3'
    } else {
        'python'
    }
    & $pythonExe (Join-Path $PSScriptRoot 'oracle.py')
    if ($LASTEXITCODE -ne 0) { throw 'Oracle check failed' }
} finally { Pop-Location }
