param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Sketch to Story Pipeline - Windows setup"
Write-Host "Project root: $PSScriptRoot"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Get-PythonCommand {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            & py -3.12 -c "import sys; print(sys.version)"
            return @("py", "-3.12")
        } catch {
            Write-Host "Python launcher exists, but Python 3.12 was not found through py -3.12."
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        try {
            $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($version -eq "3.12") {
                return @("python")
            }
        } catch {
            Write-Host "A python command exists, but it could not run correctly."
        }
    }

    throw "Python 3.12 was not found. Install Python 3.12 x64, then run this script again."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCommand = Get-PythonCommand
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
