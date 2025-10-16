# -------------------------------------
# Interactive configuration
# -------------------------------------
$MODEL_ID = Read-Host "Enter model ID [default: ibm-granite/granite-3b-code-instruct-128k]"
if ([string]::IsNullOrWhiteSpace($MODEL_ID)) { $MODEL_ID = "ibm-granite/granite-3b-code-instruct-128k" }

$BITNESS = Read-Host "Enter quantization bitness (4bit / 8bit / 16bit) [default: 16bit]"
if ([string]::IsNullOrWhiteSpace($BITNESS)) { $BITNESS = "16bit" }

$PORT = Read-Host "Enter port [default: 8000]"
if ([string]::IsNullOrWhiteSpace($PORT)) { $PORT = 8000 }

$HOST_ADDRESS = Read-Host "Enter network address [default: 0.0.0.0]"
if ([string]::IsNullOrWhiteSpace($HOST)) { $HOST_ADDRESS = "0.0.0.0" }

$ACCESS_LOG = Read-Host "Enable access logging? [default: False]"
if ([string]::IsNullOrWhiteSpace($ACCESS_LOG)) { $ACCESS_LOG = $false }

# -------------------------------------
# Summary
# -------------------------------------
Write-Host ""
Write-Host "--------------------------------"
Write-Host "Starting Local LLM Server"
Write-Host "Model:            $MODEL_ID"
Write-Host "Host:             $HOST_ADDRESS"
Write-Host "Port:             $PORT"
Write-Host "Quantization:     $BITNESS"
Write-Host "Venv:             $PWD\.venv"
Write-Host "Access Log:       $ACCESS_LOG"
Write-Host "--------------------------------"
Write-Host ""

# -------------------------------------
# Export environment variables for Python
# -------------------------------------
$env:LLLM_MODEL_ID = $MODEL_ID
$env:LLLM_BITNESS = $BITNESS
$env:LLLM_PORT = $PORT
$env:LLLM_HOST = $HOST_ADDRESS
$env:LLLM_ACCESS_LOG = $ACCESS_LOG

# -------------------------------------
# Activate venv if exists
# -------------------------------------
$VenvActivate = Join-Path $PWD ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "Activating venv..."
    & $VenvActivate
}
else {
    Write-Warning ".venv not found. Make sure dependencies are installed."
}

# -------------------------------------
# Run the Python server
# -------------------------------------
python -m laurus_llm.server.app
