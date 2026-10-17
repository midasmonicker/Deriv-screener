# One-command run: quick sanity check, then the full 60-day edge report.
# Run from the deriv-vol-lab folder:  powershell -ExecutionPolicy Bypass -File scripts\run_edge_overnight.ps1
# Logs are plain text, git-ignored (*_log.txt). Read-only on the database.
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
$summary = "run_summary_log.txt"
$total = [System.Diagnostics.Stopwatch]::StartNew()
"started $(Get-Date -Format s)" | Set-Content -Encoding ascii $summary

# 1. quick sanity check (not valid evidence, only proves the script runs on the big database)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
cmd /c "$py scripts\flag_edge_report.py --quick > quick_out_log.txt 2> quick_err_log.txt"
$quickCode = $LASTEXITCODE
"quick: exit=$quickCode elapsed=$($sw.Elapsed)" | Add-Content -Encoding ascii $summary
if ($quickCode -ne 0) {
    "quick check failed; full run skipped" | Add-Content -Encoding ascii $summary
    exit $quickCode
}

# 2. full run (real-data analysis + 20 null replications, default parallel workers)
$sw.Restart()
cmd /c "$py scripts\flag_edge_report.py > edge_out_log.txt 2> edge_err_log.txt"
$fullCode = $LASTEXITCODE
"full: exit=$fullCode elapsed=$($sw.Elapsed)" | Add-Content -Encoding ascii $summary
"finished $(Get-Date -Format s), total $($total.Elapsed)" | Add-Content -Encoding ascii $summary
exit $fullCode
