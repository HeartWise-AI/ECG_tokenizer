"""
Script to visualize evaluation metrics from JSON output.
Creates bar plots for overall and per-category metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

def plot_metrics(metrics_file: str, output_dir: str = "metric_plots"):
    """
    Create visualizations for evaluation metrics.
    
    Args:
        metrics_file: Path to JSON file with metrics
        output_dir: Directory to save plots
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    overall_metrics = data.get('overall_metrics', {})
    per_category_metrics = data.get('per_category_metrics', {})
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Plot overall metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = {
        'ROUGE-1': overall_metrics.get('rouge1', 0),
        'ROUGE-2': overall_metrics.get('rouge2', 0),
        'ROUGE-L': overall_metrics.get('rougeL', 0),
        'BLEU-1': overall_metrics.get('bleu1', 0),
        'BLEU-4': overall_metrics.get('bleu4', 0),
        'METEOR': overall_metrics.get('meteor', 0),
        'BERTScore F1': overall_metrics.get('bertscore_f1', 0)
    }
    
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    
    # Color bars based on score
    colors = ['green' if v > 0.6 else 'orange' if v > 0.4 else 'red' 
              for v in metrics_to_plot.values()]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Overall Metrics (n={overall_metrics.get("num_samples", 0)} samples)', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Plot per-category comparison
    if per_category_metrics:
        # Prepare data for category plots
        categories = list(per_category_metrics.keys())
        
        # Sort by number of samples (largest first)
        categories = sorted(categories, 
                          key=lambda x: per_category_metrics[x].get('num_samples', 0),
                          reverse=True)
        
        # Limit to top categories if too many
        if len(categories) > 15:
            categories = categories[:15]
        
        # Create comparison matrix
        metrics_types = ['rouge1', 'rougeL', 'bleu1', 'bleu4', 'meteor', 'bertscore_f1']
        metric_labels = ['ROUGE-1', 'ROUGE-L', 'BLEU-1', 'BLEU-4', 'METEOR', 'BERTScore']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (metric_key, metric_label) in enumerate(zip(metrics_types, metric_labels)):
            ax = axes[idx]
            
            values = []
            cat_labels = []
            samples = []
            
            for cat in categories:
                if cat in per_category_metrics:
                    value = per_category_metrics[cat].get(metric_key, 0)
                    n_samples = per_category_metrics[cat].get('num_samples', 0)
                    values.append(value)
                    # Truncate long category names
                    cat_name = cat if len(cat) <= 20 else cat[:17] + '...'
                    cat_labels.append(f'{cat_name}\n(n={n_samples})')
                    samples.append(n_samples)
            
            # Create bars
            bars = ax.bar(range(len(values)), values)
            
            # Color based on performance
            colors = ['green' if v > 0.6 else 'orange' if v > 0.4 else 'red' for v in values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(cat_labels)))
            ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(f'{metric_label} by Category', fontsize=11, fontweight='bold')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Metrics by Category', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'category_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Create heatmap of all categories and metrics
        if len(per_category_metrics) > 0:
            # Prepare data for heatmap
            metrics_for_heatmap = ['rouge1', 'rouge2', 'rougeL', 'bleu1', 'bleu4', 'meteor']
            heatmap_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-1', 'BLEU-4', 'METEOR']
            
            # Get all categories sorted by total average score
            all_categories = []
            for cat, metrics in per_category_metrics.items():
                avg_score = np.mean([metrics.get(m, 0) for m in metrics_for_heatmap])
                all_categories.append((cat, avg_score, metrics.get('num_samples', 0)))
            
            all_categories.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to top 20 categories for readability
            top_categories = all_categories[:20]
            
            # Create matrix
            matrix = []
            cat_names = []
            
            for cat, avg_score, n_samples in top_categories:
                row = [per_category_metrics[cat].get(m, 0) for m in metrics_for_heatmap]
                matrix.append(row)
                cat_name = cat if len(cat) <= 25 else cat[:22] + '...'
                cat_names.append(f'{cat_name} (n={n_samples})')
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, max(8, len(cat_names) * 0.4)))
            
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks
            ax.set_xticks(np.arange(len(heatmap_labels)))
            ax.set_yticks(np.arange(len(cat_names)))
            ax.set_xticklabels(heatmap_labels)
            ax.set_yticklabels(cat_names)
            
            # Rotate the tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', rotation=270, labelpad=15)
            
            # Add text annotations
            for i in range(len(cat_names)):
                for j in range(len(heatmap_labels)):
                    value = matrix[i][j]
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha="center", va="center", 
                                 color="white" if value < 0.5 else "black",
                                 fontsize=8)
            
            ax.set_title('Metrics Heatmap by Category (Top 20)', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_path / 'category_metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Create summary statistics table
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Calculate summary stats across categories
        summary_stats = {}
        for metric in metrics_types:
            values = [per_category_metrics[cat].get(metric, 0) 
                     for cat in per_category_metrics if per_category_metrics[cat].get('num_samples', 0) > 0]
            if values:
                summary_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Print summary table
        print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 65)
        
        for metric, label in zip(metrics_types, metric_labels):
            if metric in summary_stats:
                stats = summary_stats[metric]
                print(f"{label:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                      f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}")
        
        # 5. Save summary to file
        summary_output = output_path / 'summary_statistics.txt'
        with open(summary_output, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-"*40 + "\n")
            for metric, value in metrics_to_plot.items():
                f.write(f"{metric:<20}: {value:.4f}\n")
            f.write(f"Total Samples: {overall_metrics.get('num_samples', 0)}\n")
            
            f.write("\n\nPer-Category Summary Statistics:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}\n")
            f.write("-" * 65 + "\n")
            
            for metric, label in zip(metrics_types, metric_labels):
                if metric in summary_stats:
                    stats = summary_stats[metric]
                    f.write(f"{label:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                          f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['median']:<10.4f}\n")
            
            f.write("\n\nTop 10 Categories by Average Score:\n")
            f.write("-"*40 + "\n")
            for i, (cat, avg_score, n_samples) in enumerate(all_categories[:10], 1):
                f.write(f"{i:2}. {cat:<30} (n={n_samples:<5}) Avg: {avg_score:.4f}\n")
        
        print(f"\nSummary saved to: {summary_output}")
        print(f"Plots saved to: {output_path}/")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot evaluation metrics")
    parser.add_argument('--metrics-file', type=str, required=True,
                       help='Path to JSON metrics file')
    parser.add_argument('--output-dir', type=str, default='metric_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_metrics(args.metrics_file, args.output_dir)