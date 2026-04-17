#!/usr/bin/env python3
"""Extract metrics from autoresearch run.log and compute composite score."""

import re
import sys


def extract_metrics(log_path: str) -> dict[str, float]:
    """Parse AUTORESEARCH_METRICS block and Overall Metrics from run.log."""
    with open(log_path) as f:
        content = f.read()

    metrics: dict[str, float] = {}

    # Parse AUTORESEARCH_METRICS block (from validation snapshot)
    pattern = r"--- AUTORESEARCH_METRICS ---\n(.*?)\n--- END_AUTORESEARCH_METRICS ---"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        # Use the last block (most recent validation)
        block = matches[-1]
        for line in block.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    pass

    # Also try to extract from "Overall Metrics" blocks (end-of-epoch validation)
    overall_pattern = r"Overall Metrics.*?\n((?:\s+\w+.*\n)*)"
    overall_matches = re.findall(overall_pattern, content)
    if overall_matches:
        block = overall_matches[-1]
        for line in block.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                try:
                    metrics[f"overall/{key.strip()}"] = float(value.strip())
                except ValueError:
                    pass

    # Try to extract peak GPU memory from torch logs
    mem_pattern = r"peak_vram_mb:\s*([\d.]+)"
    mem_match = re.search(mem_pattern, content)
    if mem_match:
        metrics["peak_memory_gb"] = float(mem_match.group(1)) / 1024

    # Also try GPU memory from common PyTorch patterns
    gpu_mem_pattern = r"Max memory allocated:\s*([\d.]+)\s*GB"
    gpu_match = re.search(gpu_mem_pattern, content)
    if gpu_match:
        metrics["peak_memory_gb"] = float(gpu_match.group(1))

    return metrics


def compute_composite(metrics: dict[str, float]) -> float:
    """Compute weighted composite score from metrics."""
    # Try snapshot metrics first, then overall
    loss = metrics.get("val_snapshot/loss", metrics.get("overall/loss", None))
    rouge = metrics.get("val_snapshot/rougeL", metrics.get("overall/rougeL", None))
    bleu = metrics.get("val_snapshot/bleu4", metrics.get("overall/bleu4", None))
    meteor = metrics.get("val_snapshot/meteor", metrics.get("overall/meteor", None))
    bertscore = metrics.get("val_snapshot/bertscore_f1", metrics.get("overall/bertscore_f1", None))

    required = {"loss": loss, "rougeL": rouge, "bleu4": bleu, "meteor": meteor, "bertscore_f1": bertscore}
    missing = [k for k, v in required.items() if v is None]

    if missing:
        print(f"WARNING: Missing metrics: {missing}")
        # Try to compute with what we have
        composite = 0.0
        if loss is not None:
            composite += -0.3 * loss
        if rouge is not None:
            composite += 0.25 * rouge
        if meteor is not None:
            composite += 0.15 * meteor
        if bleu is not None:
            composite += 0.15 * bleu
        if bertscore is not None:
            composite += 0.15 * bertscore
        return composite

    return -0.3 * loss + 0.25 * rouge + 0.15 * meteor + 0.15 * bleu + 0.15 * bertscore


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run.log>")
        sys.exit(1)

    log_path = sys.argv[1]
    metrics = extract_metrics(log_path)

    if not metrics:
        print("ERROR: No metrics found in log file. Run may have crashed.")
        sys.exit(1)

    composite = compute_composite(metrics)

    # Print structured summary
    print("\n=== EXPERIMENT RESULTS ===")

    # Key metrics
    loss = metrics.get("val_snapshot/loss", metrics.get("overall/loss", "N/A"))
    rouge = metrics.get("val_snapshot/rougeL", metrics.get("overall/rougeL", "N/A"))
    bleu = metrics.get("val_snapshot/bleu4", metrics.get("overall/bleu4", "N/A"))
    meteor = metrics.get("val_snapshot/meteor", metrics.get("overall/meteor", "N/A"))
    bertscore = metrics.get("val_snapshot/bertscore_f1", metrics.get("overall/bertscore_f1", "N/A"))
    memory = metrics.get("peak_memory_gb", "N/A")

    print(f"composite_score: {composite:.6f}")
    print(f"val_loss: {loss}")
    print(f"rougeL: {rouge}")
    print(f"bleu4: {bleu}")
    print(f"meteor: {meteor}")
    print(f"bertscore_f1: {bertscore}")
    print(f"memory_gb: {memory}")
    print("=========================\n")

    # Also print all raw metrics for debugging
    print("All extracted metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
