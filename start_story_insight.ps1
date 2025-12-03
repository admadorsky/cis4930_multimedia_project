param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 5000,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        & python @Args
    }
    else {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $py) {
            throw "Python was not found on PATH. Install Python 3.10+ and try again."
        }
        & py -3 @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($Args -join ' ')"
    }
}

Write-Host "Upgrading pip..."
Invoke-PythonCommand -Args @("-m", "pip", "install", "--upgrade", "pip")

Write-Host "Installing project dependencies..."
Invoke-PythonCommand -Args @("-m", "pip", "install", "--upgrade", "-r", "requirements.txt")

Write-Host "Ensuring spaCy 'en_core_web_sm' model is available..."
Invoke-PythonCommand -Args @("-m", "spacy", "download", "en_core_web_sm")

if (-not $NoBrowser) {
    $indexPath = Join-Path $scriptRoot "index.html"
    Write-Host "Opening $indexPath in your default browser..."
    Start-Process $indexPath
}

Write-Host "Launching Story Insight API at http://$ApiHost`:$Port (Ctrl+C to stop)..."
Invoke-PythonCommand -Args @((Join-Path $scriptRoot "main.py"), "--host", $ApiHost, "--port", $Port.ToString(), "--debug")

