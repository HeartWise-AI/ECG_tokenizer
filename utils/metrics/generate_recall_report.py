#!/usr/bin/env python3
"""
Generate comprehensive recall@K analysis report for SigLIP Phase 1.

This script produces detailed statistics, category breakdowns, and clinical
readiness assessment from recall metrics computed by siglip_recall.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recall-csv",
        required=True,
        type=Path,
        help="Recall CSV produced by siglip_recall.py",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="",
        help="Checkpoint name for report (e.g., 'tywat0ui_20251014-002732')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value used for recall computation",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load recall data
    df = pd.read_csv(args.recall_csv)
    lbl_df = df[df["label_type"] == "LBL"].copy()

    # Determine output path
    if args.output_md is None:
        output_md = args.recall_csv.parent / f"RECALL_AT{args.k}_REPORT.md"
    else:
        output_md = args.output_md

    # Generate report
    with open(output_md, "w") as f:
        f.write(f"# COMPREHENSIVE RECALL@{args.k} ANALYSIS\n")
        if args.checkpoint_name:
            f.write(f"## Checkpoint: {args.checkpoint_name}\n")
        f.write("\n---\n\n")

        # Overall statistics
        mean_recall = lbl_df["recall"].mean()
        median_recall = lbl_df["recall"].median()

        f.write("## OVERALL STATISTICS\n\n")
        f.write(f"**Total Labels:** {len(df)} ({len(lbl_df)} LBL + {len(df[df['label_type'] == 'QA'])} QA)\n\n")
        f.write(f"**Mean Recall@{args.k} (LBL only):** {mean_recall:.4f} ({mean_recall*100:.2f}%)\n\n")
        f.write(f"**Median Recall@{args.k} (LBL only):** {median_recall:.4f} ({median_recall*100:.2f}%)\n\n")
        f.write("---\n\n")

        # Category breakdown
        f.write("## CATEGORY BREAKDOWN (LBL only)\n\n")
        f.write("| Category | Mean Recall@{} | # Labels | Status |\n".format(args.k))
        f.write("|----------|----------------|----------|---------|\n")

        categories = lbl_df.groupby("category")["recall"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        for category, row in categories.iterrows():
            status = ""
            if row["mean"] >= 0.8:
                status = "Excellent"
            elif row["mean"] >= 0.6:
                status = "Good"
            elif row["mean"] >= 0.4:
                status = "Moderate"
            elif row["mean"] >= 0.2:
                status = "Poor"
            else:
                status = "Critical"

            f.write(f"| **{category}** | {row['mean']:.4f} ({row['mean']*100:.2f}%) | {int(row['count'])} | {status} |\n")

        f.write("\n---\n\n")

        # Top performers
        f.write(f"## TOP 20 BEST PERFORMING DIAGNOSES\n\n")
        for idx, (_, row) in enumerate(lbl_df.nlargest(20, "recall").iterrows(), 1):
            f.write(f"{idx}. **{row['text']}**\n")
            f.write(f"   - Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%)\n")
            f.write(f"   - Category: {row['category']}\n")
            f.write(f"   - Hits: {row['hits']}/{row['total']}\n\n")

        f.write("---\n\n")

        # Worst performers
        f.write(f"## TOP 20 WORST PERFORMING DIAGNOSES\n\n")
        for idx, (_, row) in enumerate(lbl_df.nsmallest(20, "recall").iterrows(), 1):
            warning = " ⚠️⚠️⚠️" if row['recall'] < 0.1 else " ⚠️⚠️" if row['recall'] < 0.2 else " ⚠️" if row['recall'] < 0.5 else ""
            f.write(f"{idx}. **{row['text']}**{warning}\n")
            f.write(f"   - Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%)\n")
            f.write(f"   - Category: {row['category']}\n")
            f.write(f"   - Hits: {row['hits']}/{row['total']}\n\n")

        f.write("---\n\n")

        # Performance distribution
        f.write(f"## RECALL@{args.k} DISTRIBUTION\n\n")
        f.write("| Recall Range | Count | Percentage |\n")
        f.write("|--------------|-------|-----------|\n")

        bins = [
            (0.0, 0.1, "Catastrophic"),
            (0.1, 0.2, "Poor"),
            (0.2, 0.3, "Below Average"),
            (0.3, 0.4, "Below Average"),
            (0.4, 0.5, "Moderate"),
            (0.5, 0.6, "Moderate"),
            (0.6, 0.7, "Good"),
            (0.7, 0.8, "Good"),
            (0.8, 0.9, "Excellent"),
            (0.9, 1.0, "Excellent"),
        ]

        for low, high, label in bins:
            count = len(lbl_df[(lbl_df["recall"] >= low) & (lbl_df["recall"] < high)])
            pct = count / len(lbl_df) * 100 if len(lbl_df) > 0 else 0
            f.write(f"| [{low:.1f}, {high:.1f}) - {label} | {count} | {pct:.1f}% |\n")

        # Perfect recall
        count_perfect = len(lbl_df[lbl_df["recall"] == 1.0])
        pct_perfect = count_perfect / len(lbl_df) * 100 if len(lbl_df) > 0 else 0
        f.write(f"| = 1.0 - Perfect | {count_perfect} | {pct_perfect:.1f}% |\n")

        f.write("\n---\n\n")

        # Clinical assessment
        f.write("## CLINICAL READINESS ASSESSMENT\n\n")

        # Critical life-threatening conditions
        critical_labels = {
            "LBL_ventricular_tachycardia": "Ventricular tachycardia",
            "LBL_third_degree_av_block": "3rd degree AV block",
            "LBL_st_elevation_anterior_v3_v4": "STEMI (anterior)",
            "LBL_st_elevation_inferior_ii_iii_avf": "STEMI (inferior)",
            "LBL_acute_mi": "Acute MI",
        }

        f.write("### Life-Threatening Conditions:\n\n")
        for text_id, name in critical_labels.items():
            row = lbl_df[lbl_df["text_id"] == text_id]
            if not row.empty:
                row = row.iloc[0]
                status = "🚫 UNSAFE" if row["recall"] < 0.9 else "✓ SAFE"
                f.write(f"- **{name}**: {row['recall']*100:.1f}% - {status}\n")

        f.write("\n")

        # Overall verdict
        mean_critical = 0.0
        critical_count = 0
        for text_id in critical_labels:
            row = lbl_df[lbl_df["text_id"] == text_id]
            if not row.empty:
                mean_critical += row.iloc[0]["recall"]
                critical_count += 1

        if critical_count > 0:
            mean_critical /= critical_count

        f.write("### Overall Clinical Verdict:\n\n")
        if mean_critical >= 0.9:
            f.write("**STATUS: READY FOR CLINICAL DEPLOYMENT** ✓\n\n")
            f.write("Model achieves >90% recall on life-threatening conditions.\n")
        elif mean_critical >= 0.7:
            f.write("**STATUS: ACCEPTABLE WITH CAUTION** ⚠️\n\n")
            f.write("Model achieves moderate performance on critical conditions. Recommend human oversight.\n")
        else:
            f.write("**STATUS: NOT READY FOR CLINICAL DEPLOYMENT** 🚫\n\n")
            f.write(f"Model achieves only {mean_critical*100:.1f}% mean recall on life-threatening conditions. ")
            f.write("This represents an UNACCEPTABLE SAFETY RISK.\n\n")
            f.write("**Recommended Use:**\n")
            f.write("- Research/development only\n")
            f.write("- Educational tools\n")
            f.write("- NOT for patient care decisions\n")

        f.write("\n---\n\n")

        # Recommendations
        f.write("## RECOMMENDATIONS\n\n")

        # Find categories with <50% mean recall
        weak_categories = categories[categories["mean"] < 0.5]
        if len(weak_categories) > 0:
            f.write("### Priority Improvements Needed:\n\n")
            for category, row in weak_categories.iterrows():
                f.write(f"1. **{category}**: {row['mean']*100:.1f}% → Target >70%\n")
                # Show worst labels in this category
                cat_labels = lbl_df[lbl_df["category"] == category].nsmallest(3, "recall")
                for _, label_row in cat_labels.iterrows():
                    f.write(f"   - {label_row['text']}: {label_row['recall']*100:.1f}%\n")
                f.write("\n")

        f.write("---\n\n")
        f.write(f"**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source:** {args.recall_csv.name}\n")

    print(f"Report saved to: {output_md}")


if __name__ == "__main__":
    main()
