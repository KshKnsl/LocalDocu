Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Write-Succ($msg) { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "[error] $msg" -ForegroundColor Red }

$InstallDir = Join-Path $env:USERPROFILE ".localdocu-backend"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Info "install directory: $InstallDir"

function Test-Python {
    $candidates = @('python')
    foreach ($c in $candidates) {
        try {
            $out = & cmd /c "$c --version" 2>&1
            if ($LASTEXITCODE -eq 0) { return $c }
        } catch { }
    }
    return $null
}

function Try-Install-Python-Winget {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Attempting to install Python via winget (requires admin / consent)..."
        try {
            winget install --accept-package-agreements --accept-source-agreements -e --id Python.Python.3 -h
            return $true
        } catch {
            Write-Err "winget installation failed or was cancelled"
            return $false
        }
    }
    return $false
}

function Run-DirectBootstrap($pythonCmd) {
    Write-Info "Running bootstrap directly with: $pythonCmd"
    try {
        $installDir = $Env:USERPROFILE + '\\.localdocu-backend'
        New-Item -ItemType Directory -Force -Path $installDir | Out-Null

        & cmd /c "$pythonCmd -m pip install --upgrade pip setuptools wheel" 2>$null
        & cmd /c "$pythonCmd -m pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-ollama langchain-experimental flashrank pydantic python-dotenv" 2>$null

        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            Write-Info "Attempting to install Ollama (best-effort)"
            try {
                iex (iwr -useb https://ollama.com/install.ps1)
            } catch {
                Write-Err "Ollama installation failed or requires manual install"
            }
        }

        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            & ollama pull gemma3:1b 2>$null || Write-Info 'gemma3 pull failed (ok to ignore)'
            & ollama pull llava 2>$null || Write-Info 'llava pull failed (ok to ignore)'
        }

        $hUrl = 'https://raw.githubusercontent.com/KshKnsl/LocalDocu/main/ai-backend/Hindices.py'
        $hFile = Join-Path $installDir 'Hindices.py'
        Invoke-WebRequest -Uri $hUrl -OutFile $hFile -UseBasicParsing -TimeoutSec 60

        $shimDir = Join-Path $Env:USERPROFILE "bin"
        New-Item -ItemType Directory -Force -Path $shimDir | Out-Null
        $shimPath = Join-Path $shimDir "localdocu-run.cmd"
        $shimContent = "@echo off`r`npython `"$installDir\Hindices.py`" %*"
        Set-Content -Path $shimPath -Value $shimContent -Force

        $userPath = [Environment]::GetEnvironmentVariable("Path","User")
        if (-not ($userPath -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $shimDir })) {
            $newUserPath = ($userPath + ";" + $shimDir).TrimEnd(';')
            [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
            Write-Info "Added $shimDir to user PATH — restart your terminal to use 'localdocu-run'"
        }
        if (-not ($Env:PATH -split ';' | Where-Object { $_ -eq $shimDir })) {
            $Env:PATH = "$Env:PATH;$shimDir"
        }

        return $true
    } catch {
        Write-Err "Bootstrap failed: $_"
        return $false
    }
}
$py = Test-Python
if ($py) {
    Write-Info "Python detected: $py"
    if (Run-DirectBootstrap $py) {
        Write-Succ 'Setup completed (python present)'
        Write-Info "Start the backend manually: `n  python $InstallDir\Hindices.py`n  localdocu-run"
        exit 0
    } else {
        Write-Err 'Bootstrap run failed; will attempt other options'
    }
}

if (Try-Install-Python-Winget) {
    Start-Sleep -Seconds 2
    $py = Test-Python
    if ($py) {
        if (Run-DirectBootstrap $py) {

            Write-Succ 'Setup completed after installing Python'
            Write-Info "Start the backend manually: `n  python $InstallDir\Hindices.py`n  localdocu-run"
            exit 0
        } else {
            Write-Err 'Bootstrap after python install failed'
        }
    }
}

Write-Err "Automatic setup failed. Please install Python 3.10+ and rerun the installer or follow manual instructions in the README."
exit 1
