#!/usr/bin/env bash
# Train mrl, mrl_pic, mipic and mrl_sia one after another for one seed.
# Bash equivalent of train_all.ps1, for Linux or Kaggle.
#
# Runs the unmodified run_experiments.py once per arm on a single GPU, with the
# Kaggle-matching defaults (5 epochs, lr 2e-5, max length 256, batch 16).
# Output goes to h11_probe/runs/seed<SEED>/.
#
# The SIA attention-KL term (Eq. 6) is pinned OFF with MIPIC_ATTENTION_KL=0, the
# configuration of every run under h11_probe/runs/. The repository default is ON
# (as in the paper), so the pin keeps those runs reproducible. --attention-kl
# trains with KL ON, only the SIA arms (mipic, mrl_sia), into h11_probe/runs_kl/.
#
# An arm is skipped if it already finished cleanly: best.pt exists AND its entry
# is in tweeteval_results.json. The exit code alone is not trusted, because
# run_experiments.py exits 0 even when evaluation fails.
#
# Anything not finished cleanly is restarted from scratch with MIPIC_FRESH=1 and
# never resumed: the best-epoch tracker is not saved in training_state.pt, so a
# resumed run could keep the wrong best.pt.
#
# Usage (from anywhere):
#   bash h11_probe/train_all.sh [--seed N] [--python PYTHON] [--attention-kl] [--dry-run]
set -euo pipefail

SEED=42
PYTHON=python
DRY_RUN=0
ATTENTION_KL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --attention-kl) ATTENTION_KL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $ATTENTION_KL -eq 1 ]]; then
    MODELS=(mipic mrl_sia); RUNS_REL="h11_probe/runs_kl"
else
    MODELS=(mrl mrl_pic mipic mrl_sia); RUNS_REL="h11_probe/runs"
fi
# Any of these left in the environment would change the run away from the
# Kaggle-matching defaults, truncate it, or turn it into a smoke test.
CLEAR_VARS=(MIPIC_MAX_BATCHES MIPIC_SMOKE_FULL_PIPELINE MIPIC_EVAL_TRAIN_SAMPLES
            MIPIC_EVAL_TEST_SAMPLES MIPIC_EPOCHS MIPIC_LEARNING_RATE MIPIC_MAX_LENGTH
            MIPIC_BATCH_SIZE MIPIC_CHECKPOINT MIPIC_FRESH MIPIC_ATTENTION_KL LOCAL_RANK RANK WORLD_SIZE)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_REL="${RUNS_REL}/seed${SEED}"
LOG_DIR="${REPO_ROOT}/${OUT_REL}/logs"
SUMMARY="${LOG_DIR}/summary.txt"
RESULTS_JSON="${REPO_ROOT}/${OUT_REL}/tweeteval_results.json"
command -v "$PYTHON" >/dev/null || { echo "python not found: $PYTHON" >&2; exit 1; }

finished() {  # $1 = model
    [[ -f "${REPO_ROOT}/${OUT_REL}/checkpoints/$1/best.pt" && -f "$RESULTS_JSON" ]] || return 1
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if sys.argv[2] in (d.get("evaluations") or {}) else 1)' \
        "$RESULTS_JSON" "$1" 2>/dev/null
}

cd "$REPO_ROOT"
[[ $DRY_RUN -eq 1 ]] || mkdir -p "$LOG_DIR"
echo "Seed ${SEED} -> ${REPO_ROOT}/${OUT_REL}"
echo "Python: $(command -v "$PYTHON")"
echo "Attention KL: MIPIC_ATTENTION_KL=${ATTENTION_KL}"
[[ $DRY_RUN -eq 1 ]] || echo "===== train_all.sh seed ${SEED} started $(date '+%F %T') =====" >> "$SUMMARY"

declare -a ROWS=()
for model in "${MODELS[@]}"; do
    if finished "$model"; then
        echo "[$model] already finished cleanly; skipping."
        ROWS+=("$model|skipped (done)|-|-")
        continue
    fi
    log_rel="${OUT_REL}/logs/${model}.log"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[$model] would run fresh: MIPIC_SEED=${SEED} MIPIC_OUTPUT_DIR=${OUT_REL} MIPIC_FRESH=1 MIPIC_ATTENTION_KL=${ATTENTION_KL} $PYTHON run_experiments.py --model $model >> $log_rel 2>&1"
        ROWS+=("$model|would train|-|-")
        continue
    fi

    for v in "${CLEAR_VARS[@]}"; do unset "$v"; done
    export MIPIC_SEED="$SEED" MIPIC_OUTPUT_DIR="$OUT_REL" MIPIC_FRESH=1 MIPIC_ATTENTION_KL="$ATTENTION_KL" PYTHONUTF8=1
    # A previous failed attempt's best.pt must not survive into this run.
    rm -f "${OUT_REL}/checkpoints/${model}/best.pt"

    start=$(date +%s)
    echo "# ${model} seed ${SEED} started $(date '+%F %T') (MIPIC_FRESH=1, MIPIC_ATTENTION_KL=${ATTENTION_KL})" > "$log_rel"
    echo "[$model] started $(date '+%T'), log: $log_rel"
    set +e
    "$PYTHON" run_experiments.py --model "$model" >> "$log_rel" 2>&1
    code=$?
    set -e
    end=$(date +%s)
    dur=$(printf '%02d:%02d:%02d' $(((end - start) / 3600)) $((((end - start) % 3600) / 60)) $(((end - start) % 60)))

    if finished "$model"; then status="ok"
    elif [[ $code -eq 0 ]]; then status="FAILED (exit 0 but no result entry; check log for evaluation error)"
    else status="FAILED"; fi
    printf '%-8s start %s  end %s  duration %s  exit %s  status %s\n' "$model" \
        "$(date -d "@$start" '+%F %T' 2>/dev/null || echo "$start")" "$(date -d "@$end" '+%F %T' 2>/dev/null || echo "$end")" \
        "$dur" "$code" "$status" >> "$SUMMARY"
    echo "[$model] $status after $dur (exit $code)"
    ROWS+=("$model|$status|$dur|$code")
done

echo
echo "Seed ${SEED} summary:"
printf '%-8s  %-20s  %-9s  %s\n' Model Status Duration ExitCode
for row in "${ROWS[@]}"; do
    IFS='|' read -r m s d c <<< "$row"
    printf '%-8s  %-20s  %-9s  %s\n' "$m" "$s" "$d" "$c"
done
