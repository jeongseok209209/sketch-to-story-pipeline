param(
    [switch]$SkipInstall,
    [switch]$SkipPythonInstall
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Sketch to Story Pipeline - Windows setup"
Write-Host "Project root: $PSScriptRoot"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$supportedPythonVersions = @("3.12", "3.11", "3.10")

function Test-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$Label
    )

    try {
        $versionOutput = & $Exe @Args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|{sys.maxsize > 2**32}')"
        if ($LASTEXITCODE -ne 0 -or -not $versionOutput) {
            return $null
        }

        $parts = $versionOutput.Trim().Split("|")
        $version = $parts[0]
        $is64Bit = $parts.Count -gt 1 -and $parts[1] -eq "True"
        $versionParts = $version.Split(".")
        $majorMinor = "$($versionParts[0]).$($versionParts[1])"

        if (-not $is64Bit) {
            Write-Host "Found Python $version through $Label, but it is not 64-bit."
            return $null
        }

        if ($supportedPythonVersions -notcontains $majorMinor) {
            Write-Host "Found Python $version through $Label, but this project expects Python 3.10, 3.11, or 3.12."
            return $null
        }

        Write-Host "Using Python $version through $Label."
        return @($Exe) + $Args
    } catch {
        return $null
    }
}

function Find-PythonCommand {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($version in $supportedPythonVersions) {
            $candidate = Test-PythonCandidate -Exe "py" -Args @("-$version") -Label "py -$version"
            if ($candidate) {
                return $candidate
            }
        }
        Write-Host "Python launcher exists, but no supported Python version was found through py."
    }

    foreach ($command in @("python", "python3")) {
        if (Get-Command $command -ErrorAction SilentlyContinue) {
            $candidate = Test-PythonCandidate -Exe $command -Args @() -Label $command
            if ($candidate) {
                return $candidate
            }
        }
    }

    return $null
}

function Install-PythonWithWinget {
    if ($SkipPythonInstall) {
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        return
    }

    Write-Host "Supported Python was not found. Trying to install Python 3.12 x64 with winget ..."
    & winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "winget Python install did not complete successfully."
    }
}

function Get-PythonCommand {
    $candidate = Find-PythonCommand
    if ($candidate) {
        return $candidate
    }

    Install-PythonWithWinget
    $candidate = Find-PythonCommand
    if ($candidate) {
        return $candidate
    }

    throw @"
Python 3.10, 3.11, or 3.12 x64 was not found.

Install Python 3.12 x64, then run .\setup.bat again.

Recommended command:
  winget install --id Python.Python.3.12 --exact

Manual download:
  https://www.python.org/downloads/windows/

During installation, enable "Add python.exe to PATH" if the installer shows that option.
"@
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCommand = @(Get-PythonCommand)
    $pythonExe = $pythonCommand[0]
    $pythonArgs = @()
    if ($pythonCommand.Length -gt 1) {
        $pythonArgs = $pythonCommand[1..($pythonCommand.Length - 1)]
    }

    Write-Host "Creating .venv ..."
    & $pythonExe @pythonArgs -m venv .venv
} else {
    Write-Host ".venv already exists."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython"
}

& $venvPython --version

if (-not $SkipInstall) {
    Write-Host "Upgrading pip ..."
    & $venvPython -m pip install --upgrade pip

    Write-Host "Installing requirements.txt ..."
    & $venvPython -m pip install -r requirements.txt
} else {
    Write-Host "Skipping package install because -SkipInstall was provided."
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "NVIDIA check:"
    & nvidia-smi
} else {
    Write-Host "nvidia-smi was not found. That is OK; CPU mode is supported."
}

Write-Host "Running non-mutating setup check ..."
& $venvPython run.py check

Write-Host ""
Write-Host "Setup complete. Try a small smoke run:"
Write-Host ".\.venv\Scripts\python.exe run.py a --story 1 --image 1 --story-max-new-tokens 20 --output-dir outputs\smoke_A"
