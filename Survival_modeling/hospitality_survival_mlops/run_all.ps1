# PowerShell equivalent of run_all.sh. Extra args are forwarded to the pipeline (e.g. -Quick).
param([switch]$Quick, [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$log = "output/training_logs/pipeline.log"
New-Item -ItemType Directory -Force output/training_logs, output/models, output/data_visuals, output/comparison_graphs, output/api_screenshots, data | Out-Null

function Step($msg) { "[run_all] $msg" | Tee-Object -FilePath $log -Append }
function Run($cmd) { & $Python @cmd 2>&1 | Tee-Object -FilePath $log -Append; if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" } }

Step "Installing dependencies ..."
Run @("-m", "pip", "install", "-q", "-r", "requirements.txt")
Run @("-m", "playwright", "install", "chromium")
Step "Training + evaluation pipeline ..."
$extra = @(); if ($Quick) { $extra += "--quick" }
Run (@("-m", "src.train_pipeline") + $extra)
Step "API tests ..."
Run @("-m", "pytest", "-q", "tests/test_api.py")
Step "Capturing API screenshots ..."
Run @("tests/capture_api_screenshots.py")
Step "Done."
