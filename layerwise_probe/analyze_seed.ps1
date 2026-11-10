<#
.SYNOPSIS
    Run the layerwise_probe analysis pipeline on every finished model of one seed.

.DESCRIPTION
    Models are discovered from {RunsDir}/seed{Seed}/checkpoints/<model>/best.pt,
    so this works for layerwise_probe/runs (mrl, mrl_pic, mipic, mrl_sia) and
    layerwise_probe/runs_kl (mrl_sia_kl, mipic_kl) alike. A model is analysed only if
    its training finished cleanly: best.pt exists AND its entry is in
    tweeteval_results.json. That keeps a model that is still training (best.pt
    is written after epoch 1) out of the analysis.

    For each model, with feature name {model}_s{Seed}, in order:
      1. extract_cls.py   --name N --checkpoint best.pt
      2. probe_grid.py    --name N --check-readme
      3. sts_eval.py      --name N --checkpoint best.pt --drop-overlap
     3b. sts_eval.py      --name N --checkpoint best.pt   (all pairs, so the
                          clean-minus-all contamination check can be made)
      4. subset_probe.py  --name N --n-random 20

    A step is skipped when its output already exists, so reruns are safe:
      1. layerwise_probe/features/N/meta.json (written last by extract_cls.py), or
         steps 2 and 4 are both done (the features are no longer needed)
      2. layerwise_probe/results/N_grid.csv and N_preds.npz
      3. layerwise_probe/results/N_sts_clean.csv
     3b. layerwise_probe/results/N_sts.csv
      4. layerwise_probe/results/N_subsets.csv and N_subsets_summary.csv
    Steps 2 and 4 need the features, and step 4 also needs step 2's grid, so a
    failure upstream blocks them. Steps 3 and 3b read the checkpoint and always run.

    A step counts as done only if Python exits 0 AND its output files exist.

.PARAMETER Seed
    Seed folder to analyse. Default 42.

.PARAMETER RunsDir
    Folder holding seed<N>/. Default layerwise_probe/runs. Relative paths are resolved
    against the repository root.

.PARAMETER DryRun
    Print what would run or be skipped for each model, without running anything.

.PARAMETER DeleteFeatures
    After step 4 is done, delete layerwise_probe/features/N/. Results and prediction
    files (in layerwise_probe/results/) are kept.

.PARAMETER Python
    Python executable. Default: "python" from PATH.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File layerwise_probe\analyze_seed.ps1 -Seed 42
    powershell -ExecutionPolicy Bypass -File layerwise_probe\analyze_seed.ps1 -Seed 42 -RunsDir layerwise_probe\runs_kl -DeleteFeatures
#>
param(
    [int]$Seed = 42,
    [string]$RunsDir = "layerwise_probe/runs",
    [switch]$DryRun,
    [switch]$DeleteFeatures,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($RunsDir)) { $RunsDir = Join-Path $RepoRoot $RunsDir }
$RunsDir = [System.IO.Path]::GetFullPath($RunsDir)
$SeedDir = Join-Path $RunsDir "seed$Seed"
$CkptRoot = Join-Path $SeedDir "checkpoints"
$LogDir = Join-Path $SeedDir "logs"
$ResultsJson = Join-Path $SeedDir "tweeteval_results.json"
$FeaturesRoot = Join-Path $RepoRoot "layerwise_probe\features"
$ResultsRoot = Join-Path $RepoRoot "layerwise_probe\results"
$PythonExe = (Get-Command $Python -ErrorAction Stop).Source

function Write-Utf8Line([string]$Path, [string]$Text) {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($Path, $Text + [Environment]::NewLine, $utf8)
}

function Test-AllExist([string[]]$Paths) {
    foreach ($p in $Paths) { if (-not (Test-Path -LiteralPath $p)) { return $false } }
    return $true
}

function Get-FinishedModels {
    $evaluated = @()
    if (Test-Path -LiteralPath $ResultsJson) {
        try {
            $json = Get-Content -LiteralPath $ResultsJson -Raw | ConvertFrom-Json
            if ($null -ne $json.evaluations) { $evaluated = $json.evaluations.PSObject.Properties.Name }
        } catch {
            Write-Host "WARNING: could not read $ResultsJson ($($_.Exception.Message))"
        }
    }
    $found = @()
    if (Test-Path -LiteralPath $CkptRoot) {
        foreach ($dir in Get-ChildItem -LiteralPath $CkptRoot -Directory | Sort-Object Name) {
            if (-not (Test-Path -LiteralPath (Join-Path $dir.FullName "best.pt"))) { continue }
            if ($evaluated -contains $dir.Name) {
                $found += $dir.Name
            } else {
                Write-Host "[$($dir.Name)] best.pt exists but no result entry in tweeteval_results.json (still training or failed); not analysed."
            }
        }
    }
    return ,$found
}

# Runs one python script through cmd.exe so stdout and stderr go to the log
# unchanged and the exit code is reliable (same pattern as train_all.ps1).
function Invoke-Step([string]$Log, [string]$Label, [string]$ArgLine) {
    $cmdLine = "/s /c `"`"$PythonExe`" $ArgLine >> `"$Log`" 2>&1`""
    Write-Utf8Line $Log ("`n# ---- {0} started {1}: python {2}" -f $Label, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $ArgLine)
    try {
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -NoNewWindow -PassThru
        $null = $proc.Handle  # cache the handle so ExitCode is available after the wait
        $proc.WaitForExit()
        $code = $proc.ExitCode
    } catch {
        Write-Utf8Line $Log ("# launcher error: {0}" -f $_.Exception.Message)
        $code = -1
    }
    Write-Utf8Line $Log ("# ---- {0} finished {1} with exit code {2}" -f $Label, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $code)
    return $code
}

$env:PYTHONUTF8 = "1"
Push-Location $RepoRoot
$summary = @()
try {
    Write-Host "Seed $Seed in $SeedDir"
    Write-Host "Python: $PythonExe"
    if ($DryRun) { Write-Host "DRY RUN: nothing will be executed or deleted." }
    $models = Get-FinishedModels
    if ($models.Count -eq 0) { Write-Host "No finished models found under $CkptRoot." }
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

    foreach ($model in $models) {
        $name = "${model}_s$Seed"
        $ckpt = Join-Path $CkptRoot "$model\best.pt"
        $log = Join-Path $LogDir "analysis_$model.log"
        $featDir = Join-Path $FeaturesRoot $name
        # Step keys are strings so the all-pairs STS run can be "3b".
        $out = @{
            "1"  = @(Join-Path $featDir "meta.json")
            "2"  = @((Join-Path $ResultsRoot "${name}_grid.csv"), (Join-Path $ResultsRoot "${name}_preds.npz"))
            "3"  = @(Join-Path $ResultsRoot "${name}_sts_clean.csv")
            "3b" = @(Join-Path $ResultsRoot "${name}_sts.csv")
            "4"  = @((Join-Path $ResultsRoot "${name}_subsets.csv"), (Join-Path $ResultsRoot "${name}_subsets_summary.csv"))
        }
        $steps = @(
            @{ N = "1"; Label = "1 extract_cls"; Args = "layerwise_probe\extract_cls.py --name $name --checkpoint `"$ckpt`"" },
            @{ N = "2"; Label = "2 probe_grid"; Args = "layerwise_probe\probe_grid.py --name $name --check-readme" },
            @{ N = "3"; Label = "3 sts_eval (clean)"; Args = "layerwise_probe\sts_eval.py --name $name --checkpoint `"$ckpt`" --drop-overlap" },
            @{ N = "3b"; Label = "3b sts_eval (all pairs)"; Args = "layerwise_probe\sts_eval.py --name $name --checkpoint `"$ckpt`"" },
            @{ N = "4"; Label = "4 subset_probe"; Args = "layerwise_probe\subset_probe.py --name $name --n-random 20" }
        )
        $done = @{}
        foreach ($n in $out.Keys) { $done[$n] = Test-AllExist $out[$n] }
        $completed = @(); $skipped = @(); $failed = @()

        Write-Host ""
        Write-Host "[$model] as $name, checkpoint $ckpt"
        if (-not $DryRun) {
            Write-Utf8Line $log ("===== analyze_seed.ps1 {0} seed {1} started {2} =====" -f $model, $Seed, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        }

        foreach ($step in $steps) {
            $n = $step.N
            $reason = $null
            if ($done[$n]) {
                $reason = "output exists"
            } elseif ($n -eq "1" -and $done["2"] -and $done["4"]) {
                $reason = "steps 2 and 4 already done; features not needed"
                $done["1"] = $true
            } elseif (($n -eq "2" -or $n -eq "4") -and -not $done["1"]) {
                $reason = "BLOCKED: features missing (step 1 not done)"
            } elseif ($n -eq "4" -and -not $done["2"]) {
                $reason = "BLOCKED: grid missing (step 2 not done)"
            }

            if ($reason -and $reason.StartsWith("BLOCKED") -and -not $DryRun) {
                Write-Host ("  {0}: {1}" -f $step.Label, $reason)
                Write-Utf8Line $log ("# {0}: {1}" -f $step.Label, $reason)
                $failed += "$n(blocked)"
                continue
            }
            if ($reason -and -not $reason.StartsWith("BLOCKED")) {
                Write-Host ("  {0}: skip ({1})" -f $step.Label, $reason)
                if (-not $DryRun) { Write-Utf8Line $log ("# {0}: skipped ({1})" -f $step.Label, $reason) }
                $skipped += "$n"
                continue
            }
            if ($DryRun) {
                # In a dry run, earlier steps are assumed to succeed.
                Write-Host ("  {0}: would run: python {1}" -f $step.Label, $step.Args)
                $done[$n] = $true
                $completed += "$n"
                continue
            }

            Write-Host ("  {0}: running ({1})" -f $step.Label, (Get-Date -Format "HH:mm:ss"))
            $code = Invoke-Step $log $step.Label $step.Args
            if ($code -eq 0 -and (Test-AllExist $out[$n])) {
                $done[$n] = $true
                $completed += "$n"
                Write-Host ("  {0}: ok" -f $step.Label)
            } else {
                $why = if ($code -ne 0) { "exit $code" } else { "exit 0 but output missing" }
                $failed += "$n($why)"
                Write-Host ("  {0}: FAILED ({1}); see {2}" -f $step.Label, $why, $log)
                Write-Utf8Line $log ("# {0}: FAILED ({1})" -f $step.Label, $why)
            }
        }

        if ($DeleteFeatures) {
            if ($done["2"] -and $done["4"] -and (Test-Path -LiteralPath $featDir)) {
                if ($DryRun) {
                    Write-Host "  would delete features: $featDir"
                } else {
                    Remove-Item -LiteralPath $featDir -Recurse -Force
                    Write-Host "  deleted features: $featDir"
                    Write-Utf8Line $log "# deleted features folder $featDir"
                }
            } elseif (Test-Path -LiteralPath $featDir) {
                Write-Host "  features kept: steps 2 and 4 are not both done"
            }
        }

        $summary += [pscustomobject]@{
            Model     = $model
            Completed = if ($completed) { $completed -join "," } else { "-" }
            Skipped   = if ($skipped) { $skipped -join "," } else { "-" }
            Failures  = if ($failed) { $failed -join "," } else { "-" }
        }
    }
} finally {
    Pop-Location
}

Write-Host ""
if ($DryRun) { Write-Host "Seed $Seed analysis plan (dry run; 'Completed' = would run):" } else { Write-Host "Seed $Seed analysis summary:" }
$summary | Format-Table -AutoSize Model, Completed, Skipped, Failures | Out-String | Write-Host
