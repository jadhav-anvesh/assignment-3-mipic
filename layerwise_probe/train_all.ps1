<#
.SYNOPSIS
    Train mrl, mrl_pic, mipic and mrl_sia one after another for one seed.

.DESCRIPTION
    Runs the unmodified run_experiments.py once per arm on a single GPU, with the
    Kaggle-matching defaults (5 epochs, lr 2e-5, max length 256, batch 16).
    Output goes to layerwise_probe/runs/seed<Seed>/.

    The SIA attention-KL term (Eq. 6) is pinned OFF with MIPIC_ATTENTION_KL=0,
    the configuration of every run under layerwise_probe/runs/. The repository default
    is ON (as in the paper), so the pin keeps those runs reproducible. Use
    -AttentionKL for the KL-on configuration.

    An arm is skipped if it already finished cleanly in that folder: best.pt
    exists AND its entry is in tweeteval_results.json. The exit code alone is not
    trusted, because run_experiments.py exits 0 even when evaluation fails.

    Anything not finished cleanly is restarted from scratch with MIPIC_FRESH=1 and
    never resumed: the best-epoch tracker is not saved in training_state.pt, so a
    resumed run could keep the wrong best.pt.

.PARAMETER Seed
    Training seed passed as MIPIC_SEED. Default 42.

.PARAMETER Python
    Python executable to use. Default: "python" from PATH.

.PARAMETER AttentionKL
    Train with attention KL ON (MIPIC_ATTENTION_KL=1). Only the arms that have
    SIA (mipic, mrl_sia) are trained, and output goes to
    layerwise_probe/runs_kl/seed<Seed>/ so it never mixes with the KL-off runs.

.PARAMETER DryRun
    Print what would run for each arm without training anything.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File layerwise_probe\train_all.ps1 -Seed 42
#>
param(
    [int]$Seed = 42,
    [string]$Python = "python",
    [switch]$AttentionKL,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($AttentionKL) {
    $Models = @("mipic", "mrl_sia")
    $RunsRel = "layerwise_probe\runs_kl"
    $KLValue = "1"
} else {
    $Models = @("mrl", "mrl_pic", "mipic", "mrl_sia")
    $RunsRel = "layerwise_probe\runs"
    $KLValue = "0"
}

# Any of these left in the environment would change the run away from the
# Kaggle-matching defaults, truncate it, or turn it into a smoke test.
$ClearVars = @(
    "MIPIC_MAX_BATCHES", "MIPIC_SMOKE_FULL_PIPELINE",
    "MIPIC_EVAL_TRAIN_SAMPLES", "MIPIC_EVAL_TEST_SAMPLES",
    "MIPIC_EPOCHS", "MIPIC_LEARNING_RATE", "MIPIC_MAX_LENGTH", "MIPIC_BATCH_SIZE",
    "MIPIC_CHECKPOINT", "MIPIC_FRESH", "MIPIC_ATTENTION_KL", "LOCAL_RANK", "RANK", "WORLD_SIZE"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
# Relative to the repo root, which is also the working directory for every run.
$OutRel = "$RunsRel\seed$Seed"
$OutDir = Join-Path $RepoRoot $OutRel
$LogDir = Join-Path $OutDir "logs"
$Summary = Join-Path $LogDir "summary.txt"
$ResultsJson = Join-Path $OutDir "tweeteval_results.json"
$PythonExe = (Get-Command $Python -ErrorAction Stop).Source

function Test-Finished([string]$Model) {
    $best = Join-Path $OutDir "checkpoints\$Model\best.pt"
    if (-not (Test-Path -LiteralPath $best)) { return $false }
    if (-not (Test-Path -LiteralPath $ResultsJson)) { return $false }
    try {
        $json = Get-Content -LiteralPath $ResultsJson -Raw | ConvertFrom-Json
    } catch {
        return $false
    }
    if ($null -eq $json.evaluations) { return $false }
    return ($json.evaluations.PSObject.Properties.Name -contains $Model)
}

function Format-Duration([TimeSpan]$Span) {
    return "{0:00}:{1:00}:{2:00}" -f [int][math]::Floor($Span.TotalHours), $Span.Minutes, $Span.Seconds
}

function Write-Utf8Line([string]$Path, [string]$Text) {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($Path, $Text + [Environment]::NewLine, $utf8)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Push-Location $RepoRoot
$results = @()
try {
    Write-Host "Seed $Seed -> $OutDir"
    Write-Host "Python: $PythonExe"
    Write-Host "Attention KL: MIPIC_ATTENTION_KL=$KLValue"
    if (-not $DryRun) {
        Write-Utf8Line $Summary ("===== train_all.ps1 seed {0} started {1} =====" -f $Seed, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    }

    foreach ($model in $Models) {
        if (Test-Finished $model) {
            Write-Host "[$model] already finished cleanly; skipping."
            $results += [pscustomobject]@{ Model = $model; Status = "skipped (done)"; Duration = "-"; ExitCode = "-" }
            continue
        }

        foreach ($name in $ClearVars) {
            Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
        }
        $env:MIPIC_SEED = "$Seed"
        $env:MIPIC_OUTPUT_DIR = $OutRel
        $env:MIPIC_FRESH = "1"
        $env:MIPIC_ATTENTION_KL = $KLValue
        # Python writing to a redirected file otherwise uses cp1252 and can crash
        # on non-ASCII log output.
        $env:PYTHONUTF8 = "1"

        $logRel = "$OutRel\logs\$model.log"
        $logPath = Join-Path $RepoRoot $logRel
        $cmdLine = "/s /c `"`"$PythonExe`" run_experiments.py --model $model >> `"$logRel`" 2>&1`""

        if ($DryRun) {
            Write-Host "[$model] would run fresh (MIPIC_ATTENTION_KL=$KLValue): cmd.exe $cmdLine"
            $results += [pscustomobject]@{ Model = $model; Status = "would train"; Duration = "-"; ExitCode = "-" }
            continue
        }

        # A previous failed attempt's best.pt must not survive into this run.
        $staleBest = Join-Path $OutDir "checkpoints\$model\best.pt"
        if (Test-Path -LiteralPath $staleBest) { Remove-Item -LiteralPath $staleBest -Force }

        $start = Get-Date
        Set-Content -LiteralPath $logPath -Value "" -NoNewline
        Write-Utf8Line $logPath ("# {0} seed {1} started {2} (MIPIC_FRESH=1, MIPIC_ATTENTION_KL={3})" -f $model, $Seed, $start.ToString("yyyy-MM-dd HH:mm:ss"), $KLValue)
        Write-Host ("[{0}] started {1}, log: {2}" -f $model, $start.ToString("HH:mm:ss"), $logRel)

        $exitCode = $null
        try {
            $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -NoNewWindow -PassThru
            $null = $proc.Handle  # cache the handle so ExitCode is available after the wait
            $proc.WaitForExit()
            $exitCode = $proc.ExitCode
        } catch {
            Write-Utf8Line $logPath ("# launcher error: {0}" -f $_.Exception.Message)
            $exitCode = -1
        }
        $end = Get-Date
        $duration = Format-Duration ($end - $start)

        if (Test-Finished $model) {
            $status = "ok"
        } elseif ($exitCode -eq 0) {
            $status = "FAILED (exit 0 but no result entry; check log for evaluation error)"
        } else {
            $status = "FAILED"
        }

        Write-Utf8Line $Summary ("{0,-8} start {1}  end {2}  duration {3}  exit {4}  status {5}" -f `
            $model, $start.ToString("yyyy-MM-dd HH:mm:ss"), $end.ToString("yyyy-MM-dd HH:mm:ss"), $duration, $exitCode, $status)
        Write-Host ("[{0}] {1} after {2} (exit {3})" -f $model, $status, $duration, $exitCode)
        $results += [pscustomobject]@{ Model = $model; Status = $status; Duration = $duration; ExitCode = $exitCode }
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Seed $Seed summary:"
$results | Format-Table -AutoSize Model, Status, Duration, ExitCode | Out-String | Write-Host
