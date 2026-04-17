#!/bin/bash
# LVEF loss weight sweep: runs 10 experiments with different lvef_loss_weight values.
# Each experiment: ~100 optimizer steps, resuming from champion ssn24jkm checkpoint.
# Tracks LVEF Pearson correlation, MAE, and AUROC (GT ≤40%).
# Results are saved to autoresearch/lvef/results.tsv
set -euo pipefail

cd /volume/ECG_tokenizer
source .venv/bin/activate

EXPERIMENT_YAML="autoresearch/lvef/experiment.yaml"
RESULTS_TSV="autoresearch/lvef/results.tsv"
LOG_DIR="autoresearch/lvef/logs"

mkdir -p "$LOG_DIR"

# Initialize results TSV
printf "lvef_loss_weight\tval_loss\trougeL\tbleu4\tmeteor\tbertscore_f1\tlvef_loss\tlvef_pearson\tlvef_mae\tlvef_auroc_le40\tlvef_n\tstatus\n" > "$RESULTS_TSV"

# Weights to sweep
WEIGHTS=(0.0 0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.7 1.0)

for weight in "${WEIGHTS[@]}"; do
    echo "============================================================"
    echo "LVEF SWEEP: lvef_loss_weight = $weight"
    echo "Started at: $(date)"
    echo "============================================================"

    # Kill any lingering GPU processes and wait for cleanup
    pkill -f "master_port=295" 2>/dev/null || true
    sleep 15
    # Force CUDA cleanup
    python3 -c "import torch; [torch.cuda.empty_cache() for _ in range(2) if torch.cuda.is_available()]" 2>/dev/null || true
    sleep 5

    # Update lvef_loss_weight in experiment.yaml
    sed -i "s/^lvef_loss_weight:.*/lvef_loss_weight: $weight/" "$EXPERIMENT_YAML"

    LOG_FILE="$LOG_DIR/lvef_weight_${weight}.log"

    # Run the experiment
    if bash autoresearch/lvef/run_experiment.sh > "$LOG_FILE" 2>&1; then
        STATUS="ok"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            STATUS="timeout"
        else
            STATUS="crash"
        fi
    fi

    echo "Experiment finished with status: $STATUS at $(date)"

    # Extract metrics from AUTORESEARCH_METRICS block
    VAL_LOSS="N/A"
    ROUGEL="N/A"
    BLEU4="N/A"
    METEOR="N/A"
    BERTSCORE_F1="N/A"
    LVEF_LOSS="N/A"
    LVEF_PEARSON="N/A"
    LVEF_MAE="N/A"
    LVEF_AUROC="N/A"
    LVEF_N="N/A"

    if [ "$STATUS" = "ok" ] || [ "$STATUS" = "timeout" ]; then
        # Try to extract metrics even on timeout (validation may have completed)
        METRICS_BLOCK=$(sed -n '/--- AUTORESEARCH_METRICS ---/,/--- END_AUTORESEARCH_METRICS ---/p' "$LOG_FILE" | tail -n +2 | head -n -1 || true)

        if [ -n "$METRICS_BLOCK" ]; then
            # Use the LAST metrics block (most recent validation)
            LAST_BLOCK=$(echo "$METRICS_BLOCK" | tac | sed '/--- AUTORESEARCH_METRICS ---/q' | tac)

            extract_metric() {
                local key="$1"
                local value
                value=$(echo "$LAST_BLOCK" | grep "^${key}:" | tail -1 | awk -F': ' '{print $2}' || true)
                if [ -n "$value" ]; then
                    echo "$value"
                else
                    echo "N/A"
                fi
            }

            VAL_LOSS=$(extract_metric "val_snapshot/loss")
            ROUGEL=$(extract_metric "val_snapshot/rougeL")
            BLEU4=$(extract_metric "val_snapshot/bleu4")
            METEOR=$(extract_metric "val_snapshot/meteor")
            BERTSCORE_F1=$(extract_metric "val_snapshot/bertscore_f1")
            LVEF_LOSS=$(extract_metric "val_snapshot/lvef_loss")
            LVEF_PEARSON=$(extract_metric "val_snapshot/lvef_pearson")
            LVEF_MAE=$(extract_metric "val_snapshot/lvef_mae")
            LVEF_AUROC=$(extract_metric "val_snapshot/lvef_auroc_le40")
            LVEF_N=$(extract_metric "val_snapshot/lvef_n_samples")
        fi

        # Fallback: try Overall Metrics if snapshot metrics not found
        if [ "$VAL_LOSS" = "N/A" ]; then
            VAL_LOSS=$(grep -oP 'loss:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "N/A")
        fi
        if [ "$ROUGEL" = "N/A" ]; then
            ROUGEL=$(grep -oP 'rougeL:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "N/A")
        fi
        # Try LVEF from printed line if not in metrics block
        if [ "$LVEF_PEARSON" = "N/A" ]; then
            LVEF_PEARSON=$(grep -oP 'Pearson=\K[\d.-]+' "$LOG_FILE" | tail -1 || echo "N/A")
        fi
        if [ "$LVEF_MAE" = "N/A" ]; then
            LVEF_MAE=$(grep -oP 'MAE=\K[\d.]+' "$LOG_FILE" | tail -1 || echo "N/A")
        fi
    fi

    # Append to results TSV
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$weight" "$VAL_LOSS" "$ROUGEL" "$BLEU4" "$METEOR" "$BERTSCORE_F1" "$LVEF_LOSS" \
        "$LVEF_PEARSON" "$LVEF_MAE" "$LVEF_AUROC" "$LVEF_N" "$STATUS" \
        >> "$RESULTS_TSV"

    echo "Results for lvef_loss_weight=$weight: loss=$VAL_LOSS rougeL=$ROUGEL bleu4=$BLEU4 meteor=$METEOR bertscore=$BERTSCORE_F1 lvef_loss=$LVEF_LOSS pearson=$LVEF_PEARSON mae=$LVEF_MAE auroc=$LVEF_AUROC n=$LVEF_N status=$STATUS"
    echo ""
done

echo "============================================================"
echo "LVEF SWEEP COMPLETE"
echo "Results saved to: $RESULTS_TSV"
echo "============================================================"
echo ""
column -t -s $'\t' "$RESULTS_TSV"
