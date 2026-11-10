#!/usr/bin/env bash
# Run the layerwise_probe analysis pipeline on every finished model of one seed.
# Bash equivalent of analyze_seed.ps1, for Linux or Kaggle.
#
# Models are discovered from <RUNS_DIR>/seed<SEED>/checkpoints/<model>/best.pt and
# analysed only if their entry is in tweeteval_results.json (training finished).
# For each model, with feature name <model>_s<SEED>:
#   1. extract_cls.py   --name N --checkpoint best.pt
#   2. probe_grid.py    --name N --check-readme
#   3. sts_eval.py      --name N --checkpoint best.pt --drop-overlap
#  3b. sts_eval.py      --name N --checkpoint best.pt   (all pairs)
#   4. subset_probe.py  --name N --n-random 20
# A step is skipped when its output already exists, so reruns are safe; a step
# counts as done only if Python exits 0 AND its output files exist. Steps 2 and 4
# need the features (step 1); step 4 also needs step 2's grid.
#
# Usage (from anywhere):
#   bash layerwise_probe/analyze_seed.sh [--seed N] [--runs-dir layerwise_probe/runs] [--dry-run]
#                                  [--delete-features] [--python PYTHON]
set -euo pipefail

SEED=42
RUNS_DIR="layerwise_probe/runs"
DRY_RUN=0
DELETE_FEATURES=0
PYTHON=python
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --runs-dir) RUNS_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --delete-features) DELETE_FEATURES=1; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Absolute: /path on Linux, or C:/path or C:\path under Git Bash on Windows.
[[ "$RUNS_DIR" = /* || "$RUNS_DIR" =~ ^[A-Za-z]:[\\/] ]] || RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"
SEED_DIR="${RUNS_DIR}/seed${SEED}"
CKPT_ROOT="${SEED_DIR}/checkpoints"
LOG_DIR="${SEED_DIR}/logs"
RESULTS_JSON="${SEED_DIR}/tweeteval_results.json"
FEAT_ROOT="layerwise_probe/features"
RES_ROOT="layerwise_probe/results"
export PYTHONUTF8=1
command -v "$PYTHON" >/dev/null || { echo "python not found: $PYTHON" >&2; exit 1; }

all_exist() { for p in "$@"; do [[ -e "$p" ]] || return 1; done; }

# Sets STEP_ARGS to the python arguments of step $1, quoted safely (paths may contain spaces).
step_args() {
    case "$1" in
        1)  STEP_ARGS=(layerwise_probe/extract_cls.py --name "$name" --checkpoint "$ckpt") ;;
        2)  STEP_ARGS=(layerwise_probe/probe_grid.py --name "$name" --check-readme) ;;
        3)  STEP_ARGS=(layerwise_probe/sts_eval.py --name "$name" --checkpoint "$ckpt" --drop-overlap) ;;
        3b) STEP_ARGS=(layerwise_probe/sts_eval.py --name "$name" --checkpoint "$ckpt") ;;
        4)  STEP_ARGS=(layerwise_probe/subset_probe.py --name "$name" --n-random 20) ;;
    esac
}

evaluated_models() {
    [[ -f "$RESULTS_JSON" ]] || return 0
    "$PYTHON" -c 'import json,sys; print("\n".join((json.load(open(sys.argv[1],encoding="utf-8")).get("evaluations") or {}).keys()))' \
        "$RESULTS_JSON" 2>/dev/null || echo "WARNING: could not read $RESULTS_JSON" >&2
}

echo "Seed ${SEED} in ${SEED_DIR}"
echo "Python: $(command -v "$PYTHON")"
[[ $DRY_RUN -eq 1 ]] && echo "DRY RUN: nothing will be executed or deleted."
EVALUATED="$(evaluated_models)"
MODELS=()
if [[ -d "$CKPT_ROOT" ]]; then
    for dir in $(ls -1 "$CKPT_ROOT" | sort); do
        [[ -f "${CKPT_ROOT}/${dir}/best.pt" ]] || continue
        if grep -qx "$dir" <<< "$EVALUATED"; then MODELS+=("$dir")
        else echo "[$dir] best.pt exists but no result entry in tweeteval_results.json (still training or failed); not analysed."; fi
    done
fi
[[ ${#MODELS[@]} -gt 0 ]] || echo "No finished models found under ${CKPT_ROOT}."
[[ $DRY_RUN -eq 1 ]] || mkdir -p "$LOG_DIR"

declare -a SUMMARY=()
for model in "${MODELS[@]}"; do
    name="${model}_s${SEED}"
    ckpt="${CKPT_ROOT}/${model}/best.pt"
    log="${LOG_DIR}/analysis_${model}.log"
    feat="${FEAT_ROOT}/${name}"
    declare -A OUT=(
        [1]="${feat}/meta.json"
        [2]="${RES_ROOT}/${name}_grid.csv ${RES_ROOT}/${name}_preds.npz"
        [3]="${RES_ROOT}/${name}_sts_clean.csv"
        [3b]="${RES_ROOT}/${name}_sts.csv"
        [4]="${RES_ROOT}/${name}_subsets.csv ${RES_ROOT}/${name}_subsets_summary.csv"
    )
    declare -A LABEL=([1]="1 extract_cls" [2]="2 probe_grid" [3]="3 sts_eval (clean)" [3b]="3b sts_eval (all pairs)" [4]="4 subset_probe")
    declare -A DONE=()
    for n in 1 2 3 3b 4; do
        # shellcheck disable=SC2086
        if all_exist ${OUT[$n]}; then DONE[$n]=1; else DONE[$n]=0; fi
    done
    completed=(); skipped=(); failed=()

    echo
    echo "[$model] as $name, checkpoint $ckpt"
    [[ $DRY_RUN -eq 1 ]] || echo "===== analyze_seed.sh ${model} seed ${SEED} started $(date '+%F %T') =====" >> "$log"

    for n in 1 2 3 3b 4; do
        reason=""
        if [[ ${DONE[$n]} -eq 1 ]]; then reason="output exists"
        elif [[ $n == 1 && ${DONE[2]} -eq 1 && ${DONE[4]} -eq 1 ]]; then reason="steps 2 and 4 already done; features not needed"; DONE[1]=1
        elif [[ ($n == 2 || $n == 4) && ${DONE[1]} -eq 0 ]]; then reason="BLOCKED: features missing (step 1 not done)"
        elif [[ $n == 4 && ${DONE[2]} -eq 0 ]]; then reason="BLOCKED: grid missing (step 2 not done)"; fi

        if [[ $reason == BLOCKED* && $DRY_RUN -eq 0 ]]; then
            echo "  ${LABEL[$n]}: $reason"; echo "# ${LABEL[$n]}: $reason" >> "$log"; failed+=("$n(blocked)"); continue
        fi
        if [[ -n $reason && $reason != BLOCKED* ]]; then
            echo "  ${LABEL[$n]}: skip ($reason)"
            [[ $DRY_RUN -eq 1 ]] || echo "# ${LABEL[$n]}: skipped ($reason)" >> "$log"
            skipped+=("$n"); continue
        fi
        step_args "$n"
        if [[ $DRY_RUN -eq 1 ]]; then
            # In a dry run, earlier steps are assumed to succeed.
            echo "  ${LABEL[$n]}: would run: $PYTHON $(printf '%q ' "${STEP_ARGS[@]}")"; DONE[$n]=1; completed+=("$n"); continue
        fi

        echo "  ${LABEL[$n]}: running ($(date '+%T'))"
        printf '\n# ---- %s started %s: python %s\n' "${LABEL[$n]}" "$(date '+%F %T')" "$(printf '%q ' "${STEP_ARGS[@]}")" >> "$log"
        set +e
        "$PYTHON" "${STEP_ARGS[@]}" >> "$log" 2>&1
        code=$?
        set -e
        echo "# ---- ${LABEL[$n]} finished $(date '+%F %T') with exit code $code" >> "$log"
        # shellcheck disable=SC2086
        if [[ $code -eq 0 ]] && all_exist ${OUT[$n]}; then
            DONE[$n]=1; completed+=("$n"); echo "  ${LABEL[$n]}: ok"
        else
            why=$([[ $code -ne 0 ]] && echo "exit $code" || echo "exit 0 but output missing")
            failed+=("$n($why)"); echo "  ${LABEL[$n]}: FAILED ($why); see $log"; echo "# ${LABEL[$n]}: FAILED ($why)" >> "$log"
        fi
    done

    if [[ $DELETE_FEATURES -eq 1 && -d "$feat" ]]; then
        if [[ ${DONE[2]} -eq 1 && ${DONE[4]} -eq 1 ]]; then
            if [[ $DRY_RUN -eq 1 ]]; then echo "  would delete features: $feat"
            else rm -rf "$feat"; echo "  deleted features: $feat"; echo "# deleted features folder $feat" >> "$log"; fi
        else
            echo "  features kept: steps 2 and 4 are not both done"
        fi
    fi
    join() { local IFS=,; echo "${*:--}"; }
    SUMMARY+=("$model|$(join "${completed[@]}")|$(join "${skipped[@]}")|$(join "${failed[@]}")")
    unset OUT LABEL DONE
done

echo
if [[ $DRY_RUN -eq 1 ]]; then echo "Seed ${SEED} analysis plan (dry run; 'Completed' = would run):"; else echo "Seed ${SEED} analysis summary:"; fi
printf '%-12s  %-12s  %-12s  %s\n' Model Completed Skipped Failures
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r m c s f <<< "$row"
    printf '%-12s  %-12s  %-12s  %s\n' "$m" "$c" "$s" "$f"
done
