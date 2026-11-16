param([switch]$IncludeArtifacts)
$ErrorActionPreference = 'Stop'
$auditRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
Push-Location $auditRoot
try {
    $auditOutput = Join-Path $auditRoot 'target/bug-audit'
    New-Item -ItemType Directory -Force -Path $auditOutput | Out-Null
    $exeExt = if ($IsWindows -or $env:OS -eq 'Windows_NT') { '.exe' } else { '' }

    function Get-ProfileArtifacts {
        param([string]$ProfileName)
        $buildRoot = if ($env:CARGO_BUILD_BUILD_DIR) {
            $env:CARGO_BUILD_BUILD_DIR
        } elseif ($env:CARGO_TARGET_DIR) {
            $env:CARGO_TARGET_DIR
        } else {
            Join-Path $auditRoot 'target'
        }
        $profileDir = Join-Path $buildRoot $ProfileName
        $depDirs = [System.Collections.Generic.List[string]]::new()
        $rlibs = @{}
        if (-not (Test-Path $profileDir)) {
            return @{ DepDirs = $depDirs; Rlibs = $rlibs }
        }
        $skipDirs = @('.fingerprint', 'examples', 'fingerprint', 'incremental', 'run')
        $queue = [System.Collections.Generic.Queue[System.IO.DirectoryInfo]]::new()
        $rootInfo = Get-Item $profileDir
        foreach ($sub in $rootInfo.GetDirectories()) {
            if ($skipDirs -notcontains $sub.Name) { $queue.Enqueue($sub) }
        }
        $depExts = @('.rlib', '.rmeta', '.so', '.dll', '.dylib')
        while ($queue.Count -gt 0) {
            $dir = $queue.Dequeue()
            foreach ($sub in $dir.GetDirectories()) {
                if ($skipDirs -notcontains $sub.Name) { $queue.Enqueue($sub) }
            }
            $hasArtifact = $false
            foreach ($file in $dir.GetFiles()) {
                if ($depExts -contains $file.Extension) {
                    $hasArtifact = $true
                    if ($file.Extension -eq '.rlib' -and $file.Name.StartsWith('lib')) {
                        $stem = $file.BaseName.Substring(3)
                        $dash = $stem.LastIndexOf('-')
                        $crate = if ($dash -ge 0) { $stem.Substring(0, $dash) } else { $stem }
                        if (-not $rlibs.ContainsKey($crate)) {
                            $rlibs[$crate] = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
                        }
                        $rlibs[$crate].Add($file)
                    }
                }
            }
            if ($hasArtifact) {
                $depDirs.Add($dir.FullName)
            }
        }
        return @{ DepDirs = $depDirs; Rlibs = $rlibs }
    }

    function New-DepRspFlags {
        param([string]$ProfileName, [System.Collections.Generic.List[string]]$DepDirs)
        if ($DepDirs.Count -eq 0) { return @() }
        $rspPath = Join-Path $auditOutput ($ProfileName + '.deps.rsp')
        $lines = [System.Collections.Generic.List[string]]::new()
        foreach ($d in $DepDirs) {
            $lines.Add('-L')
            $lines.Add('dependency=' + $d)
        }
        [System.IO.File]::WriteAllLines($rspPath, $lines, [System.Text.UTF8Encoding]::new($false))
        return @('@' + $rspPath)
    }

    $relArtifacts = Get-ProfileArtifacts 'release'
    $relDepFlags = @(New-DepRspFlags 'release' $relArtifacts.DepDirs)
    $auditTypes = $null
    if ($relArtifacts.Rlibs.ContainsKey('numpy_types')) {
        $auditTypes = $relArtifacts.Rlibs['numpy_types'] |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (!$auditTypes) { throw 'Build numpy-types in release mode first.' }
    foreach ($auditName in @('numeric_fixes','zero_shape','complex_order','histogram_boundary')) {
        $auditSource = Join-Path $PSScriptRoot ($auditName + '.rs')
        $auditExecutable = Join-Path $auditOutput ($auditName + $exeExt)
        & rustc --edition=2024 -Awarnings -O -C lto=thin -C overflow-checks=yes $auditSource `
            --extern "numpy_types=$auditTypes" @relDepFlags -o $auditExecutable
        if ($LASTEXITCODE -ne 0) { throw "Failed compiling $auditName" }
        & $auditExecutable
        if ($LASTEXITCODE -ne 0) { throw "Failed probe $auditName" }
    }
    if ($IncludeArtifacts) {
        $dbgArtifacts = Get-ProfileArtifacts 'debug'
        $dbgDepFlags = @(New-DepRspFlags 'debug' $dbgArtifacts.DepDirs)
        foreach ($auditProbe in @(@('io_polynomial','numpy_lib'),@('masked','numpy_extra'))) {
            $auditName = $auditProbe[0]; $auditCrate = $auditProbe[1]
            $auditLibrary = $null
            if ($dbgArtifacts.Rlibs.ContainsKey($auditCrate)) {
                $auditLibrary = $dbgArtifacts.Rlibs[$auditCrate] |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
            }
            if (!$auditLibrary) { throw "Missing debug artifact for $auditCrate" }
            $auditExecutable = Join-Path $auditOutput ($auditName + $exeExt)
            & rustc --edition=2024 -Awarnings (Join-Path $PSScriptRoot ($auditName + '.rs')) `
                --extern ($auditCrate + '=' + $auditLibrary) @dbgDepFlags -o $auditExecutable
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
