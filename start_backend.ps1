# ─────────────────────────────────────────────────────────────
# MyopiaGuard — Backend Startup Script
# Run this from the repo root: .\start_backend.ps1
# ─────────────────────────────────────────────────────────────

$RepoRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy    = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$BackendDir = Join-Path $RepoRoot "backend"
$ApiScript  = Join-Path $BackendDir "api.py"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       MyopiaGuard — Backend Startup          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check venv exists ─────────────────────────────────────
if (-not (Test-Path $VenvPy)) {
    Write-Host "⚠  Virtual environment not found at .venv\" -ForegroundColor Yellow
    Write-Host "   Creating it now..." -ForegroundColor Yellow
    python -m venv "$RepoRoot\.venv"
    Write-Host "✓  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "✓  Virtual environment found." -ForegroundColor Green
}

# ── 2. Install / update requirements ─────────────────────────
Write-Host ""
Write-Host "→  Installing/verifying requirements..." -ForegroundColor Cyan
& $VenvPy -m pip install -q -r "$BackendDir\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗  pip install failed. Check requirements.txt." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✓  Requirements satisfied." -ForegroundColor Green

# ── 3. Set working dir to backend and launch Flask ───────────
Write-Host ""
Write-Host "→  Starting Flask API on http://localhost:5001 ..." -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host ""

Set-Location $BackendDir
& $VenvPy $ApiScript
