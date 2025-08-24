import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# Custom color scheme
COLORS = {
    'gpt4': "#1f77b4",      # Blue
    'gpt5': "#9467bd",      # Purple
    'correct': "#2ca02c",   # Green
    'hallucination': "#d62728",  # Red
    'background': "#f5f5f5"  # Light gray
}

# Create results directory structure
results_dir = Path('../results')
vis_dir = results_dir / 'visualizations'
csv_dir = results_dir / 'csv_data'

for directory in [results_dir, vis_dir, csv_dir]:
    directory.mkdir(exist_ok=True)

def safe_json_parse(text):
    """Safely parse a potential JSON string."""
    if pd.isna(text):
        return {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    
    if not isinstance(text, str):
        return {'entailment': 0, 'neutral': 0, 'contradiction': 0}
        
    try:
        # Replace single quotes with double quotes for JSON parsing
        return json.loads(text.replace("'", '"'))
    except:
        return {'entailment': 0, 'neutral': 0, 'contradiction': 0}

def load_and_preprocess_data():
    """Load and preprocess the datasets with error handling"""
    print("Loading and preprocessing data...")
    
    try:
        # Load datasets
        gpt4_data = pd.read_csv('../data/gpt4_responses_with_hallucinations.csv')
        gpt5_data = pd.read_csv('../data/gpt5_responses_with_hallucinations.csv')
        
        # Process each dataset
        for df, model in [(gpt4_data, 'gpt4'), (gpt5_data, 'gpt5')]:
            # Convert is_hallucination to boolean
            df['is_hallucination'] = df['is_hallucination'].astype(bool)
            
            # Process confidence scores
            df['confidence_scores_dict'] = df['confidence_scores'].apply(safe_json_parse)
            df['entailment_prob'] = df['confidence_scores_dict'].apply(lambda x: x.get('entailment', 0))
            df['neutral_prob'] = df['confidence_scores_dict'].apply(lambda x: x.get('neutral', 0))
            df['contradiction_prob'] = df['confidence_scores_dict'].apply(lambda x: x.get('contradiction', 0))
            
            # Define hallucination severity based on classification
            df['severity'] = 'None'
            df.loc[df['is_hallucination'] & (df['classification'] == 'neutral'), 'severity'] = 'Mild'
            df.loc[df['is_hallucination'] & (df['classification'] == 'contradiction'), 'severity'] = 'Severe'
        
        print(f"Loaded {len(gpt4_data)} GPT-4 responses and {len(gpt5_data)} GPT-5 responses")
        return gpt4_data, gpt5_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def analyze_sector_distribution(gpt4_data, gpt5_data):
    """
    Analyze and visualize hallucination distribution across biomedical sectors with
    improved visualizations for clarity and interpretability.
    
    Outputs:
    - sector_distribution.png: Enhanced horizontal bar chart with sample sizes and error bars
    - sector_heatmap.png: Heatmap showing hallucination rates across sectors
    - sector_ranking.csv: Ranked sectors by hallucination frequency
    - sector_improvement.png: Clearer comparison of GPT-4 vs GPT-5 improvements
    """
    print("\n1. Analyzing Hallucination Distribution Across Sectors...")
    
    # Compute hallucination rates by focus area with confidence intervals
    def get_sector_rates(df, model_name):
        sector_stats = df.groupby('focus_area').agg({
            'is_hallucination': ['count', 'mean', 'sum'],
        }).reset_index()
        
        # Calculate standard error and 95% confidence intervals
        sector_stats.columns = ['focus_area', 'sample_size', 'hallucination_rate', 'total_hallucinations']
        sector_stats['std_err'] = np.sqrt((sector_stats['hallucination_rate'] * (1 - sector_stats['hallucination_rate'])) 
                                         / sector_stats['sample_size'])
        sector_stats['ci_lower'] = sector_stats['hallucination_rate'] - 1.96 * sector_stats['std_err']
        sector_stats['ci_upper'] = sector_stats['hallucination_rate'] + 1.96 * sector_stats['std_err']
        sector_stats['model'] = model_name
        
        # Ensure confidence intervals are within [0, 1]
        sector_stats['ci_lower'] = sector_stats['ci_lower'].clip(0, 1)
        sector_stats['ci_upper'] = sector_stats['ci_upper'].clip(0, 1)
        
        return sector_stats
    
    gpt4_sectors = get_sector_rates(gpt4_data, 'GPT-4')
    gpt5_sectors = get_sector_rates(gpt5_data, 'GPT-5')
    
    # Combine datasets for comparison
    all_sectors = pd.concat([gpt4_sectors, gpt5_sectors])
    
    # Save sector ranking to CSV with additional statistics
    sector_ranking = pd.merge(
        gpt4_sectors[['focus_area', 'sample_size', 'hallucination_rate', 'total_hallucinations']],
        gpt5_sectors[['focus_area', 'sample_size', 'hallucination_rate', 'total_hallucinations']],
        on='focus_area',
        suffixes=('_GPT-4', '_GPT-5')
    )
    
    # Calculate difference, improvement, and statistical significance
    sector_ranking['difference'] = sector_ranking['hallucination_rate_GPT-4'] - sector_ranking['hallucination_rate_GPT-5']
    sector_ranking['improvement'] = sector_ranking['difference'] > 0
    
    # Calculate relative improvement percentage
    sector_ranking['improvement_percent'] = np.where(
        sector_ranking['hallucination_rate_GPT-4'] > 0,
        (sector_ranking['difference'] / sector_ranking['hallucination_rate_GPT-4']) * 100,
        0
    )
    
    # Determine statistical significance using z-test for proportions
    def z_test_proportions(p1, n1, p2, n2):
        # Pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        # Z-statistic
        if se == 0:  # Avoid division by zero
            return 0
        z = (p1 - p2) / se
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_value
    
    # Apply z-test to each row
    for i, row in sector_ranking.iterrows():
        p_value = z_test_proportions(
            row['hallucination_rate_GPT-4'], row['sample_size_GPT-4'],
            row['hallucination_rate_GPT-5'], row['sample_size_GPT-5']
        )
        sector_ranking.loc[i, 'p_value'] = p_value
    
    # Add significance indicator
    sector_ranking['significant'] = sector_ranking['p_value'] < 0.05
    
    # Sort by GPT-4 hallucination rate (descending)
    sector_ranking = sector_ranking.sort_values('hallucination_rate_GPT-4', ascending=False)
    sector_ranking.to_csv(csv_dir / '1_sector_ranking.csv', index=False)
    
    # Filter for sectors with at least 5 samples in each model
    filtered_ranking = sector_ranking[
        (sector_ranking['sample_size_GPT-4'] >= 5) & 
        (sector_ranking['sample_size_GPT-5'] >= 5)
    ]
    
    # VISUALIZATION 1: Enhanced horizontal bar chart with top sectors
    plt.figure(figsize=(14, 10))
    
    # Take top 15 sectors by GPT-4 hallucination rate
    top_sectors = filtered_ranking.head(15)
    
    # Set up plot positions
    y_pos = np.arange(len(top_sectors))
    width = 0.4
    
    # Create horizontal bars
    ax = plt.subplot(111)
    
    # Plot GPT-4 bars
    gpt4_bars = ax.barh(
        y_pos - width/2, 
        top_sectors['hallucination_rate_GPT-4'], 
        width,
        color=COLORS['gpt4'],
        alpha=0.8,
        label='GPT-4'
    )
    
    # Calculate error bar widths for GPT-4 with proper clipping to [0, 1]
    gpt4_stderr = np.sqrt((top_sectors['hallucination_rate_GPT-4'] * (1 - top_sectors['hallucination_rate_GPT-4'])) / top_sectors['sample_size_GPT-4'])
    gpt4_lower_err = np.minimum(top_sectors['hallucination_rate_GPT-4'], 1.96 * gpt4_stderr)
    gpt4_upper_err = np.minimum(1 - top_sectors['hallucination_rate_GPT-4'], 1.96 * gpt4_stderr)
    
    # Add error bars for GPT-4
    ax.errorbar(
        top_sectors['hallucination_rate_GPT-4'],
        y_pos - width/2,
        xerr=np.vstack([gpt4_lower_err, gpt4_upper_err]),
        fmt='none',
        color='black',
        capsize=3
    )
    
    # Plot GPT-5 bars
    gpt5_bars = ax.barh(
        y_pos + width/2, 
        top_sectors['hallucination_rate_GPT-5'], 
        width,
        color=COLORS['gpt5'],
        alpha=0.8,
        label='GPT-5'
    )
    
    # Calculate error bar widths for GPT-5 with proper clipping to [0, 1]
    gpt5_stderr = np.sqrt((top_sectors['hallucination_rate_GPT-5'] * (1 - top_sectors['hallucination_rate_GPT-5'])) / top_sectors['sample_size_GPT-5'])
    gpt5_lower_err = np.minimum(top_sectors['hallucination_rate_GPT-5'], 1.96 * gpt5_stderr)
    gpt5_upper_err = np.minimum(1 - top_sectors['hallucination_rate_GPT-5'], 1.96 * gpt5_stderr)
    
    # Add error bars for GPT-5
    ax.errorbar(
        top_sectors['hallucination_rate_GPT-5'],
        y_pos + width/2,
        xerr=np.vstack([gpt5_lower_err, gpt5_upper_err]),
        fmt='none',
        color='black',
        capsize=3
    )
    
    # Add sample size annotations
    for i, (_, row) in enumerate(top_sectors.iterrows()):
        # GPT-4 sample size
        ax.text(
            0.01, 
            y_pos[i] - width/2, 
            f"n={int(row['sample_size_GPT-4'])}",
            va='center',
            color='white',
            fontweight='bold',
            fontsize=9
        )
        
        # GPT-5 sample size
        ax.text(
            0.01, 
            y_pos[i] + width/2, 
            f"n={int(row['sample_size_GPT-5'])}",
            va='center',
            color='white',
            fontweight='bold',
            fontsize=9
        )
        
        # Add significance indicators for statistically significant improvements
        if row['significant'] and row['improvement']:
            ax.text(
                max(row['hallucination_rate_GPT-4'], row['hallucination_rate_GPT-5']) + 0.02,
                y_pos[i],
                f"Δ = {row['difference']:.2f} (*)",
                va='center',
                fontweight='bold',
                color='green'
            )
        elif row['improvement']:
            ax.text(
                max(row['hallucination_rate_GPT-4'], row['hallucination_rate_GPT-5']) + 0.02,
                y_pos[i],
                f"Δ = {row['difference']:.2f}",
                va='center',
                color='green'
            )
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_sectors['focus_area'])
    ax.set_xlabel('Hallucination Rate', fontsize=12)
    ax.set_title('Top 15 Medical Sectors by Hallucination Rate', fontsize=14)
    ax.set_xlim(0, max(filtered_ranking['hallucination_rate_GPT-4'].max(), 
                       filtered_ranking['hallucination_rate_GPT-5'].max()) * 1.2)
    
    # Add a grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add annotation explaining the significance
    plt.annotate(
        "* indicates statistically significant difference (p < 0.05)",
        xy=(0.01, -0.07),
        xycoords='axes fraction',
        fontsize=10,
        style='italic'
    )
    
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(vis_dir / '1_sector_distribution.png', dpi=300)
    plt.close()
    
    # VISUALIZATION 2: Improvement visualization (lollipop chart)
    plt.figure(figsize=(14, 10))
    
    # Select sectors with significant improvements
    improved_sectors = filtered_ranking[
        (filtered_ranking['improvement']) & 
        (filtered_ranking['significant'])
    ].sort_values('improvement_percent', ascending=False).head(15)
    
    if len(improved_sectors) > 0:
        # Create positions
        y_pos = np.arange(len(improved_sectors))
        
        # Plot horizontal lines
        plt.hlines(
            y=y_pos, 
            xmin=0, 
            xmax=improved_sectors['improvement_percent'],
            color='gray', 
            alpha=0.5
        )
        
        # Plot dots at each end
        plt.scatter(
            improved_sectors['improvement_percent'],
            y_pos,
            s=120,
            color='green',
            alpha=0.7,
            label='Improvement'
        )
        
        # Add percentage labels
        for i, (_, row) in enumerate(improved_sectors.iterrows()):
            plt.text(
                row['improvement_percent'] + 1, 
                y_pos[i],
                f"{row['improvement_percent']:.1f}%",
                va='center'
            )
            
            # Add sample sizes
            plt.text(
                1, 
                y_pos[i] - 0.25,
                f"GPT-4: {row['hallucination_rate_GPT-4']:.2f} (n={int(row['sample_size_GPT-4'])})",
                va='center',
                fontsize=8,
                color='darkblue'
            )
            
            plt.text(
                1, 
                y_pos[i] + 0.25,
                f"GPT-5: {row['hallucination_rate_GPT-5']:.2f} (n={int(row['sample_size_GPT-5'])})",
                va='center',
                fontsize=8,
                color='darkred'
            )
        
        # Customize plot
        plt.yticks(y_pos, improved_sectors['focus_area'])
        plt.xlabel('Improvement Percentage (%)', fontsize=12)
        plt.title('Sectors with Statistically Significant Improvements (GPT-4 to GPT-5)', fontsize=14)
        
        # Add grid for readability
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
    else:
        # If no significant improvements, add a note
        plt.text(0.5, 0.5, "No statistically significant improvements found", 
                ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    
    plt.savefig(vis_dir / '1_sector_improvement.png', dpi=300)
    plt.close()
    
    # VISUALIZATION 3: Heatmap of hallucination rates by sector and model
    # Filter for sectors with reasonable sample sizes
    heatmap_sectors = filtered_ranking.sort_values('hallucination_rate_GPT-4', ascending=False).head(20)
    
    # Create a pivot table for the heatmap
    heatmap_data = pd.DataFrame({
        'focus_area': heatmap_sectors['focus_area'],
        'GPT-4': heatmap_sectors['hallucination_rate_GPT-4'],
        'GPT-5': heatmap_sectors['hallucination_rate_GPT-5']
    })
    
    # Create a separate DataFrame for annotations (sample sizes)
    annot_data = pd.DataFrame({
        'focus_area': heatmap_sectors['focus_area'],
        'GPT-4': [f"n={n}" for n in heatmap_sectors['sample_size_GPT-4'].astype(int)],
        'GPT-5': [f"n={n}" for n in heatmap_sectors['sample_size_GPT-5'].astype(int)]
    })
    
    # Melt the dataframes for seaborn
    heatmap_melted = pd.melt(
        heatmap_data, 
        id_vars=['focus_area'], 
        var_name='model', 
        value_name='hallucination_rate'
    )
    
    annot_melted = pd.melt(
        annot_data,
        id_vars=['focus_area'],
        var_name='model',
        value_name='annotation'
    )
    
    # Create pivoted data for the heatmap
    pivot_data = heatmap_melted.pivot(index='focus_area', columns='model', values='hallucination_rate')
    annot_pivot = annot_melted.pivot(index='focus_area', columns='model', values='annotation')
    
    # Create the heatmap
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(
        pivot_data,
        annot=True,
        cmap='YlOrRd',
        vmin=0,
        vmax=max(filtered_ranking['hallucination_rate_GPT-4'].max(), 
                filtered_ranking['hallucination_rate_GPT-5'].max()),
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Hallucination Rate'}
    )
    
    # Add sample size annotations
    for i, focus_area in enumerate(pivot_data.index):
        for j, model in enumerate(pivot_data.columns):
            # Get the sample size annotation
            sample_text = annot_pivot.loc[focus_area, model]
            # Add text below the hallucination rate
            ax.text(
                j + 0.5, 
                i + 0.75, 
                sample_text, 
                ha='center', 
                va='center', 
                fontsize=8,
                color='gray'
            )
    
    # Customize plot
    plt.title('Hallucination Rate Heatmap by Medical Sector', fontsize=14)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(vis_dir / '1_sector_heatmap.png', dpi=300)
    plt.close()
    
    print("  - Generated sector_distribution.png (enhanced horizontal bar chart)")
    print("  - Generated sector_improvement.png (improvement visualization)")
    print("  - Generated sector_heatmap.png (sector comparison heatmap)")
    print("  - Saved sector_ranking.csv (with statistical significance)")
    
    # NEW VISUALIZATION: Top 15 focus areas by total number of questions
    # This plot mirrors the exact format of 1_sector_distribution.png but is
    # focused on the focus areas with the most questions (combined across GPT-4 and GPT-5).
    try:
        # Compute top 15 focus areas by total questions across both datasets
        combined_counts = pd.concat([gpt4_data[['focus_area']], gpt5_data[['focus_area']]])
        top_counts = combined_counts['focus_area'].value_counts().head(15).reset_index()
        top_counts.columns = ['focus_area', 'total_questions']
        top_focus = top_counts['focus_area'].tolist()

        # Select rows from sector_ranking for these focus areas and preserve the order
        top_sectors = sector_ranking[sector_ranking['focus_area'].isin(top_focus)].copy()
        # Reorder according to top_focus
        top_sectors['order'] = top_sectors['focus_area'].apply(lambda x: top_focus.index(x) if x in top_focus else -1)
        top_sectors = top_sectors.sort_values('order')

        # Create horizontal bar chart in the same style as 1_sector_distribution.png
        plt.figure(figsize=(14, 10))
        y_pos = np.arange(len(top_sectors))
        width = 0.4

        # GPT-4 bars (use columns from sector_ranking)
        gpt4_vals = top_sectors['hallucination_rate_GPT-4']
        gpt4_bars = plt.barh(
            y_pos - width/2,
            gpt4_vals,
            width,
            color=COLORS['gpt4'],
            alpha=0.8,
            label='GPT-4'
        )

        # GPT-4 error bars
        gpt4_stderr = np.sqrt((gpt4_vals * (1 - gpt4_vals)) / top_sectors['sample_size_GPT-4'])
        gpt4_lower_err = np.minimum(gpt4_vals, 1.96 * gpt4_stderr)
        gpt4_upper_err = np.minimum(1 - gpt4_vals, 1.96 * gpt4_stderr)
        plt.errorbar(
            gpt4_vals,
            y_pos - width/2,
            xerr=np.vstack([gpt4_lower_err, gpt4_upper_err]),
            fmt='none',
            color='black',
            capsize=3
        )

        # GPT-5 bars
        gpt5_vals = top_sectors['hallucination_rate_GPT-5']
        gpt5_bars = plt.barh(
            y_pos + width/2,
            gpt5_vals,
            width,
            color=COLORS['gpt5'],
            alpha=0.8,
            label='GPT-5'
        )

        # GPT-5 error bars
        gpt5_stderr = np.sqrt((gpt5_vals * (1 - gpt5_vals)) / top_sectors['sample_size_GPT-5'])
        gpt5_lower_err = np.minimum(gpt5_vals, 1.96 * gpt5_stderr)
        gpt5_upper_err = np.minimum(1 - gpt5_vals, 1.96 * gpt5_stderr)
        plt.errorbar(
            gpt5_vals,
            y_pos + width/2,
            xerr=np.vstack([gpt5_lower_err, gpt5_upper_err]),
            fmt='none',
            color='black',
            capsize=3
        )

        # Add sample size annotations and significance markers (same logic as original)
        for i, (_, row) in enumerate(top_sectors.iterrows()):
            # GPT-4 sample size
            plt.text(
                0.01,
                y_pos[i] - width/2,
                f"n={int(row['sample_size_GPT-4'])}",
                va='center',
                color='white' if row['hallucination_rate_GPT-4'] > 0.2 else 'black',
                fontweight='bold',
                fontsize=9
            )

            # GPT-5 sample size
            plt.text(
                0.01,
                y_pos[i] + width/2,
                f"n={int(row['sample_size_GPT-5'])}",
                va='center',
                color='white' if row['hallucination_rate_GPT-5'] > 0.2 else 'black',
                fontweight='bold',
                fontsize=9
            )

            # Significance/annotation similar to original
            if row.get('p_value', 1) < 0.05 and (row['hallucination_rate_GPT-4'] - row['hallucination_rate_GPT-5']) > 0:
                plt.text(
                    max(row['hallucination_rate_GPT-4'], row['hallucination_rate_GPT-5']) + 0.02,
                    y_pos[i],
                    f"Δ = {row['hallucination_rate_GPT-4'] - row['hallucination_rate_GPT-5']:.2f} (*)",
                    va='center',
                    fontweight='bold',
                    color='green'
                )
            elif (row['hallucination_rate_GPT-4'] - row['hallucination_rate_GPT-5']) > 0:
                plt.text(
                    max(row['hallucination_rate_GPT-4'], row['hallucination_rate_GPT-5']) + 0.02,
                    y_pos[i],
                    f"Δ = {row['hallucination_rate_GPT-4'] - row['hallucination_rate_GPT-5']:.2f}",
                    va='center',
                    color='green'
                )

        # Final formatting to match 1_sector_distribution.png
        plt.yticks(y_pos, top_sectors['focus_area'])
        plt.xlabel('Hallucination Rate', fontsize=12)
        plt.title('Top 15 Focus Areas by Number of Questions (Hallucination Rates)', fontsize=14)
        plt.xlim(0, max(top_sectors['hallucination_rate_GPT-4'].max(), top_sectors['hallucination_rate_GPT-5'].max()) * 1.2)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(vis_dir / '1_top15_by_questions.png', dpi=300)
        plt.close()

        print("  - Generated 1_top15_by_questions.png (top 15 by total questions)")
    except Exception as e:
        print(f"  - Skipped generating top-questions visualization due to error: {e}")
    
    return sector_ranking

def analyze_model_comparison(gpt4_data, gpt5_data):
    """
    Compare GPT-4 vs GPT-5 model performance.
    
    Outputs:
    - overall_performance.png: Bar chart of overall hallucination rates
    - classification_distribution.png: Distribution by classification type
    - agreement_matrix.png: Confusion matrix of model agreement
    - model_metrics.csv: Detailed performance metrics
    """
    print("\n2. Analyzing Model Comparison...")
    
    # Merge datasets for comparison
    merged_data = pd.merge(
        gpt4_data[['question', 'is_hallucination', 'classification']],
        gpt5_data[['question', 'is_hallucination', 'classification']],
        on='question',
        suffixes=('_gpt4', '_gpt5')
    )
    
    # Calculate overall metrics
    metrics = {
        'GPT-4': {
            'Hallucination Rate': gpt4_data['is_hallucination'].mean(),
            'Sample Size': len(gpt4_data)
        },
        'GPT-5': {
            'Hallucination Rate': gpt5_data['is_hallucination'].mean(),
            'Sample Size': len(gpt5_data)
        }
    }
    
    # Classification distribution
    gpt4_class_dist = gpt4_data['classification'].value_counts(normalize=True)
    gpt5_class_dist = gpt5_data['classification'].value_counts(normalize=True)
    
    class_dist = pd.DataFrame({
        'GPT-4': gpt4_class_dist,
        'GPT-5': gpt5_class_dist
    }).fillna(0).reset_index()
    class_dist.columns = ['Classification', 'GPT-4', 'GPT-5']
    
    # Agreement metrics
    agreement_counts = {
        'Both Correct': ((~merged_data['is_hallucination_gpt4']) & 
                        (~merged_data['is_hallucination_gpt5'])).sum(),
        'Only GPT-4 Hallucinated': (merged_data['is_hallucination_gpt4'] & 
                                  (~merged_data['is_hallucination_gpt5'])).sum(),
        'Only GPT-5 Hallucinated': ((~merged_data['is_hallucination_gpt4']) & 
                                  merged_data['is_hallucination_gpt5']).sum(),
        'Both Hallucinated': (merged_data['is_hallucination_gpt4'] & 
                             merged_data['is_hallucination_gpt5']).sum()
    }
    
    # Calculate improvement and regression
    improvement_rate = agreement_counts['Only GPT-4 Hallucinated'] / len(merged_data)
    regression_rate = agreement_counts['Only GPT-5 Hallucinated'] / len(merged_data)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'GPT-4 Hallucination Rate': metrics['GPT-4']['Hallucination Rate'],
        'GPT-5 Hallucination Rate': metrics['GPT-5']['Hallucination Rate'],
        'Improvement Rate': improvement_rate,
        'Regression Rate': regression_rate,
        'Net Improvement': improvement_rate - regression_rate,
        'Both Correct': agreement_counts['Both Correct'] / len(merged_data),
        'Both Hallucinated': agreement_counts['Both Hallucinated'] / len(merged_data)
    }])
    
    metrics_df.to_csv(csv_dir / '2_model_metrics.csv', index=False)
    
    # VISUALIZATION 1: Overall Performance
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with error bars (using bootstrap CI)
    models = ['GPT-4', 'GPT-5']
    hall_rates = [metrics['GPT-4']['Hallucination Rate'], metrics['GPT-5']['Hallucination Rate']]
    
    # Calculate confidence intervals using bootstrap
    def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        lower = np.percentile(bootstrap_means, (1-ci)/2 * 100)
        upper = np.percentile(bootstrap_means, (1+ci)/2 * 100)
        return np.mean(data) - lower, upper - np.mean(data)
    
    gpt4_error = bootstrap_ci(gpt4_data['is_hallucination'])
    gpt5_error = bootstrap_ci(gpt5_data['is_hallucination'])
    errors = [[gpt4_error[0], gpt5_error[0]], [gpt4_error[1], gpt5_error[1]]]
    
    # Create bar chart
    plt.bar(models, hall_rates, color=[COLORS['gpt4'], COLORS['gpt5']])
    plt.errorbar(models, hall_rates, yerr=errors, fmt='o', color='black', capsize=5)
    
    # Add value labels
    for i, v in enumerate(hall_rates):
        plt.text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    # Add improvement annotation
    improvement = hall_rates[0] - hall_rates[1]
    if improvement > 0:
        plt.annotate(f'Improvement: {improvement:.1%}', 
                    xy=(0.5, max(hall_rates) + 0.05),
                    ha='center', fontsize=12, color='green')
    
    # Customize plot
    plt.title('Overall Hallucination Rate Comparison', fontsize=14)
    plt.ylabel('Hallucination Rate', fontsize=12)
    plt.ylim(0, max(hall_rates) * 1.25)  # Add some space at top
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '2_overall_performance.png')
    plt.close()
    
    # VISUALIZATION 2: Classification Distribution
    plt.figure(figsize=(10, 6))
    
    # Reshape data for grouped bar chart
    class_data = pd.melt(class_dist, id_vars=['Classification'], 
                        value_vars=['GPT-4', 'GPT-5'],
                        var_name='Model', value_name='Proportion')
    
    # Create the plot
    sns.barplot(x='Classification', y='Proportion', hue='Model', data=class_data,
               palette=[COLORS['gpt4'], COLORS['gpt5']])
    
    # Customize plot
    plt.title('Distribution by Classification Type', fontsize=14)
    plt.ylabel('Proportion', fontsize=12)
    plt.xlabel('Classification', fontsize=12)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '2_classification_distribution.png')
    plt.close()
    
    # VISUALIZATION 3: Agreement Matrix
    plt.figure(figsize=(8, 6))
    
    # Create agreement matrix
    agreement_matrix = np.array([
        [agreement_counts['Both Correct'], agreement_counts['Only GPT-5 Hallucinated']],
        [agreement_counts['Only GPT-4 Hallucinated'], agreement_counts['Both Hallucinated']]
    ])
    
    # Convert to percentages
    agreement_pct = agreement_matrix / agreement_matrix.sum() * 100
    
    # Create heatmap
    sns.heatmap(agreement_pct, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=['Correct', 'Hallucination'],
               yticklabels=['Correct', 'Hallucination'])
    
    # Customize plot
    plt.title('Model Agreement Matrix (% of samples)', fontsize=14)
    plt.xlabel('GPT-5', fontsize=12)
    plt.ylabel('GPT-4', fontsize=12)
    
    # Add annotations for counts
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.8, f'n={agreement_matrix[i,j]}', 
                    ha='center', va='center', color='black', fontsize=9)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '2_agreement_matrix.png')
    plt.close()
    
    print("  - Generated overall_performance.png")
    print("  - Generated classification_distribution.png")
    print("  - Generated agreement_matrix.png")
    print("  - Saved model_metrics.csv")
    
    return metrics_df

def analyze_confidence_hallucinations(gpt4_data, gpt5_data):
    """
    Analyze the relationship between model confidence and hallucinations.
    
    Outputs:
    - confidence_distribution.png: Box plot of confidence by hallucination status
    - confidence_statistics.csv: Statistical tests of confidence differences
    - confidence_calibration.png: Calibration curves for both models
    - sector_confidence_breakdown.csv: Detailed sector-level confidence by hallucination status
    - confidence_extremes.png: Analysis of lowest vs highest confidence prompts
    - confidence_calibration_detailed.png: Enhanced calibration analysis
    """
    print("\n3. Analyzing Confidence vs Hallucinations...")
    
    # Prepare data for statistical tests
    stats_results = {}
    
    for model_name, df, conf_col in [('GPT-4', gpt4_data, 'gpt4_confidence'), 
                                   ('GPT-5', gpt5_data, 'gpt5_confidence')]:
        # Extract confidence values by hallucination status
        hall_conf = df[df['is_hallucination']][conf_col]
        no_hall_conf = df[~df['is_hallucination']][conf_col]
        
        # Calculate basic statistics
        stats_results[f'{model_name} Mean Confidence (Hallucinated)'] = hall_conf.mean()
        stats_results[f'{model_name} Mean Confidence (Correct)'] = no_hall_conf.mean()
        stats_results[f'{model_name} Std Confidence (Hallucinated)'] = hall_conf.std()
        stats_results[f'{model_name} Std Confidence (Correct)'] = no_hall_conf.std()
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(hall_conf, no_hall_conf, equal_var=False)
        stats_results[f'{model_name} t-statistic'] = t_stat
        stats_results[f'{model_name} p-value'] = p_val
        
        # Perform Mann-Whitney test
        mw_stat, mw_p = stats.mannwhitneyu(hall_conf, no_hall_conf)
        stats_results[f'{model_name} Mann-Whitney U'] = mw_stat
        stats_results[f'{model_name} Mann-Whitney p-value'] = mw_p
        
        # Calculate effect size (Cohen's d)
        mean_diff = no_hall_conf.mean() - hall_conf.mean()
        pooled_std = np.sqrt((no_hall_conf.std()**2 + hall_conf.std()**2) / 2)
        effect_size = mean_diff / pooled_std
        stats_results[f'{model_name} Effect Size (Cohen\'s d)'] = effect_size
    
    # Save statistics to CSV
    pd.DataFrame([stats_results]).to_csv(csv_dir / '3_confidence_statistics.csv', index=False)
    
    # VISUALIZATION 1: Confidence distribution by hallucination status
    plt.figure(figsize=(12, 6))
    
    # Prepare data for visualization
    conf_data = []
    
    for model_name, df, conf_col in [('GPT-4', gpt4_data, 'gpt4_confidence'), 
                                   ('GPT-5', gpt5_data, 'gpt5_confidence')]:
        for hall_status, hall_value in [('Correct', False), ('Hallucinated', True)]:
            filtered_data = df[df['is_hallucination'] == hall_value][conf_col]
            for value in filtered_data:
                conf_data.append({
                    'Model': model_name,
                    'Hallucination': hall_status,
                    'Confidence': value
                })
    
    conf_df = pd.DataFrame(conf_data)
    
    # Create violin plots
    ax = sns.violinplot(x='Model', y='Confidence', hue='Hallucination', 
                       data=conf_df, split=True, inner='quart',
                       palette={'Correct': COLORS['correct'], 
                               'Hallucinated': COLORS['hallucination']})
    
    # Add statistical significance annotations
    for i, model in enumerate(['GPT-4', 'GPT-5']):
        p_val = stats_results[f'{model} p-value']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        plt.annotate(f'{sig}', xy=(i, 0.95), ha='center', fontsize=12)
    
    # Add legend and labels
    plt.title('Confidence Distribution by Hallucination Status', fontsize=14)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.xlabel('')
    
    # Add statistical details in the bottom
    plt.annotate(
        f"GPT-4: p={stats_results['GPT-4 p-value']:.4f}, d={stats_results.get('GPT-4 Effect Size (Cohen\'s d)', 0):.2f}\n" + 
        f"GPT-5: p={stats_results['GPT-5 p-value']:.4f}, d={stats_results.get('GPT-5 Effect Size (Cohen\'s d)', 0):.2f}",
        xy=(0.5, 0.01), xycoords='figure fraction', ha='center', fontsize=9
    )
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '3_confidence_distribution.png')
    plt.close()
    
    # VISUALIZATION 2: Calibration curves
    plt.figure(figsize=(10, 6))
    
    # Create calibration curves
    for model_name, df, conf_col, color in [
        ('GPT-4', gpt4_data, 'gpt4_confidence', COLORS['gpt4']),
        ('GPT-5', gpt5_data, 'gpt5_confidence', COLORS['gpt5'])
    ]:
        # Create confidence bins
        bins = np.linspace(0, 1, 11)  # 0.0-0.1, 0.1-0.2, etc.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        conf_means = []
        sizes = []
        
        # Calculate accuracy in each bin
        for i in range(len(bins) - 1):
            mask = (df[conf_col] >= bins[i]) & (df[conf_col] < bins[i+1])
            if mask.sum() > 5:  # Only include bins with sufficient samples
                accuracies.append(1 - df[mask]['is_hallucination'].mean())
                conf_means.append(df[mask][conf_col].mean())
                sizes.append(mask.sum())
        
        # Plot calibration curve
        plt.plot(conf_means, accuracies, 'o-', label=model_name, color=color)
        
        # Add sample size annotations (only for a few points to avoid clutter)
        for i in range(0, len(sizes), 2):
            plt.annotate(f'n={sizes[i]}', 
                        xy=(conf_means[i], accuracies[i]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8)
    
    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    
    # Customize plot
    plt.title('Model Calibration: Confidence vs. Accuracy', fontsize=14)
    plt.xlabel('Predicted Confidence', fontsize=12)
    plt.ylabel('Observed Accuracy (1 - Hallucination Rate)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '3_confidence_calibration.png')
    plt.close()
    
    # NEW FEATURE 1: Generate comprehensive sector-level confidence breakdown CSV
    print("  - Generating sector confidence breakdown CSV...")
    
    # Create comprehensive dataframe with confidence metrics by sector
    # Filter out NaN values and ensure all focus areas are strings before sorting
    gpt4_sectors = [str(x) for x in gpt4_data['focus_area'].unique() if pd.notna(x)]
    gpt5_sectors = [str(x) for x in gpt5_data['focus_area'].unique() if pd.notna(x)]
    all_sectors = sorted(list(set(gpt4_sectors) | set(gpt5_sectors)))
    sector_breakdown = []
    
    for sector in all_sectors:
        sector_row = {'focus_area': sector}
        
        # Process each model
        for model_name, df, conf_col in [('GPT-4', gpt4_data, 'gpt4_confidence'), 
                                       ('GPT-5', gpt5_data, 'gpt5_confidence')]:
            # Convert focus_area to string for consistent comparison
            sector_df = df[df['focus_area'].astype(str) == sector]
            sample_size = len(sector_df)
            
            if sample_size > 0:  # Only process if we have samples
                # Overall metrics
                sector_row[f'{model_name}_sample_size'] = sample_size
                sector_row[f'{model_name}_avg_confidence'] = sector_df[conf_col].mean()
                sector_row[f'{model_name}_hallucination_rate'] = sector_df['is_hallucination'].mean()
                
                # Split by hallucination status
                hall_df = sector_df[sector_df['is_hallucination']]
                no_hall_df = sector_df[~sector_df['is_hallucination']]
                
                # Confidence metrics for hallucinated responses
                if len(hall_df) > 0:
                    sector_row[f'{model_name}_hall_count'] = len(hall_df)
                    sector_row[f'{model_name}_hall_conf_avg'] = hall_df[conf_col].mean()
                    sector_row[f'{model_name}_hall_conf_std'] = hall_df[conf_col].std()
                    sector_row[f'{model_name}_hall_conf_min'] = hall_df[conf_col].min()
                    sector_row[f'{model_name}_hall_conf_max'] = hall_df[conf_col].max()
                else:
                    sector_row[f'{model_name}_hall_count'] = 0
                    sector_row[f'{model_name}_hall_conf_avg'] = None
                    sector_row[f'{model_name}_hall_conf_std'] = None
                    sector_row[f'{model_name}_hall_conf_min'] = None
                    sector_row[f'{model_name}_hall_conf_max'] = None
                
                # Confidence metrics for correct responses
                if len(no_hall_df) > 0:
                    sector_row[f'{model_name}_correct_count'] = len(no_hall_df)
                    sector_row[f'{model_name}_correct_conf_avg'] = no_hall_df[conf_col].mean()
                    sector_row[f'{model_name}_correct_conf_std'] = no_hall_df[conf_col].std()
                    sector_row[f'{model_name}_correct_conf_min'] = no_hall_df[conf_col].min()
                    sector_row[f'{model_name}_correct_conf_max'] = no_hall_df[conf_col].max()
                else:
                    sector_row[f'{model_name}_correct_count'] = 0
                    sector_row[f'{model_name}_correct_conf_avg'] = None
                    sector_row[f'{model_name}_correct_conf_std'] = None
                    sector_row[f'{model_name}_correct_conf_min'] = None
                    sector_row[f'{model_name}_correct_conf_max'] = None
                
                # Confidence gap between correct and hallucinated responses
                if len(hall_df) > 0 and len(no_hall_df) > 0:
                    sector_row[f'{model_name}_conf_gap'] = no_hall_df[conf_col].mean() - hall_df[conf_col].mean()
                else:
                    sector_row[f'{model_name}_conf_gap'] = None
            else:
                # Fill with None if no data for this sector and model
                sector_row[f'{model_name}_sample_size'] = 0
                sector_row[f'{model_name}_avg_confidence'] = None
                sector_row[f'{model_name}_hallucination_rate'] = None
                sector_row[f'{model_name}_hall_count'] = 0
                sector_row[f'{model_name}_hall_conf_avg'] = None
                sector_row[f'{model_name}_correct_count'] = 0
                sector_row[f'{model_name}_correct_conf_avg'] = None
                sector_row[f'{model_name}_conf_gap'] = None
        
        sector_breakdown.append(sector_row)
    
    # Convert to dataframe and save as CSV
    sector_confidence_df = pd.DataFrame(sector_breakdown)
    
    # Sort by sample size (descending) and filter out sectors with very small sample sizes
    sector_confidence_df['total_samples'] = sector_confidence_df['GPT-4_sample_size'] + sector_confidence_df['GPT-5_sample_size']
    sector_confidence_df = sector_confidence_df[sector_confidence_df['total_samples'] >= 5]
    sector_confidence_df = sector_confidence_df.sort_values('total_samples', ascending=False)
    sector_confidence_df = sector_confidence_df.drop(columns=['total_samples'])
    
    # Save to CSV
    sector_confidence_df.to_csv(csv_dir / '3_sector_confidence_breakdown.csv', index=False)
    
    # NEW FEATURE 2: Enhanced calibration analysis
    print("  - Creating enhanced calibration visualization...")
    
    # Define confidence bins more intuitively - focus more bins on high confidence range
    custom_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0]
    # Create clearer bin labels with consistent formatting
    bin_labels = []
    for i in range(len(custom_bins)-1):
        # Format ranges like "0.50-0.60" for consistency
        start = f"{custom_bins[i]:.2f}"
        end = f"{custom_bins[i+1]:.2f}"
        bin_labels.append(f"{start}-{end}")
    
    # Create a figure with two subplots (one for each model)
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
    
    # Store calibration error info for summary table
    error_summary = "Bin | Hall Rate | Expected | Error\n" + "-" * 40 + "\n"
    
    # Process each model
    for ax_idx, (model_name, df, conf_col, color) in enumerate([
        ('GPT-4', gpt4_data, 'gpt4_confidence', COLORS['gpt4']),
        ('GPT-5', gpt5_data, 'gpt5_confidence', COLORS['gpt5'])
    ]):
        ax = axes[ax_idx]
        error_text = f"\n{model_name}:\n" + "-" * 40 + "\n"
        
        # Create confidence bins
        df['conf_bin'] = pd.cut(
            df[conf_col], 
            bins=custom_bins, 
            labels=bin_labels, 
            include_lowest=True
        )
        
        # Group by bin and calculate metrics
        bin_stats = df.groupby('conf_bin').agg({
            'is_hallucination': ['count', 'mean', 'sum'],
            conf_col: 'mean'
        }).reset_index()
        
        # Fix column names
        bin_stats.columns = ['conf_bin', 'count', 'hallucination_rate', 'hall_count', 'avg_confidence']
        
        # Filter out bins with too few samples
        bin_stats = bin_stats[bin_stats['count'] >= 10]
        
        if len(bin_stats) == 0:
            ax.text(0.5, 0.5, f"Not enough data for {model_name} to create bins", 
                    ha='center', va='center', fontsize=14)
            continue
            
        # Calculate standard error for hallucination rate
        bin_stats['se'] = np.sqrt(
            (bin_stats['hallucination_rate'] * (1 - bin_stats['hallucination_rate'])) / 
            bin_stats['count']
        )
        
        # Create positions for bars
        bar_positions = np.arange(len(bin_stats))
        width = 0.8
        
        # Calculate expected hallucination rate (1 - confidence)
        # Add a line showing the expected hallucination rate (perfect calibration)
        ax.plot(
            bar_positions,
            1 - bin_stats['avg_confidence'],
            'o-',
            color='black',
            linewidth=2,
            markersize=8,
            label='Expected hallucination rate (1 - confidence)'
        )
        
        # Create the bar chart
        bars = ax.bar(
            bar_positions,
            bin_stats['hallucination_rate'],
            width=width,
            yerr=1.96 * bin_stats['se'],  # 95% confidence interval
            color=color,
            alpha=0.7,
            error_kw={'ecolor': 'black', 'elinewidth': 1, 'capsize': 5},
            label='Observed hallucination rate'
        )
        
        # Bin numbers are now added with confidence ranges below in the formatted labels section
        
        # Add annotations for each bar
        for i, row in enumerate(bin_stats.itertuples()):
            # Add sample size annotation
            ax.annotate(
                f"n={row.count}",
                xy=(bar_positions[i], row.hallucination_rate + 0.02),
                ha='center',
                va='bottom',
                fontsize=10
            )
            
            # Add hallucination count inside bar if bar is tall enough
            if row.hallucination_rate > 0.08:
                ax.annotate(
                    f"{int(row.hall_count)} hall.",
                    xy=(bar_positions[i], row.hallucination_rate / 2),
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9,
                    fontweight='bold'
                )
            
            # Calculate and show calibration error
            expected_hall = 1 - row.avg_confidence
            cal_error = row.hallucination_rate - expected_hall
            
            # Add error annotation with color coding
            error_color = "🟥" if cal_error > 0.05 else "🟩" if cal_error < -0.05 else "⬜"
            error_text += f"{row.conf_bin} | {row.hallucination_rate:.2f} | {expected_hall:.2f} | {cal_error:+.2f} {error_color}\n"
            
            if abs(cal_error) > 0.05:  # Only annotate significant errors
                color_cal = 'red' if cal_error > 0 else 'green'
                ax.annotate(
                    f"{cal_error:+.2f}",
                    xy=(bar_positions[i], max(row.hallucination_rate, expected_hall) + 0.08),
                    ha='center',
                    color=color_cal,
                    fontweight='bold',
                    fontsize=9
                )
        
        # Add the error text to summary
        error_summary += error_text
        
        # Set axis labels and title
        ax.set_title(f'{model_name}: Observed vs Expected Hallucination Rate by Confidence Bin', fontsize=14)
        ax.set_ylabel('Hallucination Rate', fontsize=12)
        
        # Ensure x-ticks are properly labeled with bin ranges
        ax.set_xticks(bar_positions)
        
        # Create better formatted labels
        formatted_labels = []
        for bin_name in bin_stats['conf_bin']:
            # Make sure each label is clearly readable
            if isinstance(bin_name, str):
                formatted_labels.append(bin_name)
            else:
                # In case it's a pandas interval object
                start = f"{bin_name.left:.2f}"
                end = f"{bin_name.right:.2f}"
                formatted_labels.append(f"{start}-{end}")
        
        # Add bin numbers and confidence ranges to the bottom of each bar
        for i, pos in enumerate(bar_positions):
            # Add bin number
            ax.text(pos, -0.04, f"Bin {i+1}", ha='center', va='top', 
                   fontsize=9, fontweight='bold', color='darkblue')
            
            # Add confidence range below bin number
            if i < len(formatted_labels):
                ax.text(pos, -0.08, formatted_labels[i], ha='center', va='top',
                       fontsize=8, color='black')
        
        # Set x-tick labels with better formatting - empty strings since we added labels manually
        ax.set_xticklabels(
            [""] * len(bar_positions),  # Empty strings instead of labels
            rotation=45, 
            ha='right',
            fontsize=10,
            fontweight='bold'  # Make labels bolder
        )
        
        # Add gridlines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend with better positioning
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Instead of adding bin info at the top, we'll add it between plots later
        
        # Ensure y-axis has reasonable limits
        max_y = max(bin_stats['hallucination_rate'].max(), 
                    bin_stats['avg_confidence'].max()) + 0.15
        ax.set_ylim(0, min(1.0, max_y))
    
    # Add x-label only to bottom subplot
    axes[1].set_xlabel('Confidence Bin', fontsize=12)
    
    # Add explanation text at the bottom
    plt.figtext(
        0.5, 0.02, 
        "Calibration error: difference between observed hallucination rate and expected rate (1 - confidence)\n" +
        "🟥 Red: Overconfident (more hallucinations than expected)  |  🟩 Green: Underconfident (fewer hallucinations than expected)",
        ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
    )
    
    # Adjust layout to prevent overlap and make room for bin numbers and confidence ranges
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.15)  # Increased bottom to make room for bin labels with confidence ranges
    
    # Adjust y-limits to make room for bin numbers and confidence ranges
    for ax in axes:
        current_ylim = ax.get_ylim()
        ax.set_ylim(bottom=-0.12, top=current_ylim[1])  # Increased negative space for the confidence range labels
    
    # Save the visualization
    plt.savefig(vis_dir / '3_confidence_calibration_detailed.png', dpi=300)
    plt.close()
    
    # NEW FEATURE 3: Analysis of extreme confidence cases
    print("  - Creating confidence extremes analysis...")
    
    # Set up figure for extreme confidence analysis
    plt.figure(figsize=(14, 10))
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Process each model (GPT-4 and GPT-5)
    for model_idx, (model_name, df, conf_col) in enumerate([
        ('GPT-4', gpt4_data, 'gpt4_confidence'),
        ('GPT-5', gpt5_data, 'gpt5_confidence')
    ]):
        # First subplot: Lowest confidence examples
        low_conf_ax = axes[model_idx, 0]
        
        # Get the 10 lowest confidence examples
        lowest_conf = df.sort_values(conf_col).head(10)
        
        # Calculate hallucination rate for lowest confidence
        low_hall_rate = lowest_conf['is_hallucination'].mean()
        
        # Create bar chart
        low_conf_ax.bar(
            np.arange(len(lowest_conf)),
            lowest_conf[conf_col],
            color=[COLORS['hallucination'] if h else COLORS['correct'] for h in lowest_conf['is_hallucination']],
            alpha=0.8
        )
        
        # Add annotations for hallucination status
        for i, is_hall in enumerate(lowest_conf['is_hallucination']):
            status = "Hall." if is_hall else "Correct"
            low_conf_ax.annotate(
                status,
                xy=(i, lowest_conf.iloc[i][conf_col] + 0.02),
                ha='center',
                fontsize=8,
                rotation=90 if status == "Correct" else 0
            )
        
        # Add title and formatting
        low_conf_ax.set_title(f'{model_name}: 10 Lowest Confidence Responses\nHallucination Rate: {low_hall_rate:.2f}', fontsize=12)
        low_conf_ax.set_ylabel('Confidence Score', fontsize=10)
        low_conf_ax.set_ylim(0, 1)
        low_conf_ax.set_xticks([])
        low_conf_ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Second subplot: High/Perfect confidence examples (>= 0.98)
        high_conf_ax = axes[model_idx, 1]
        
        # Get examples with very high confidence (>= 0.98)
        perfect_conf = df[df[conf_col] >= 0.98]
        
        # If no examples with high confidence, try a lower threshold
        if len(perfect_conf) < 5 and model_name == "GPT-5":
            # Try a lower threshold for GPT-5 if needed
            perfect_conf = df.sort_values(by=conf_col, ascending=False).head(10)
            
        # If too many, take a random sample of 50
        if len(perfect_conf) > 50:
            perfect_conf = perfect_conf.sample(50, random_state=42)
        
        # Calculate hallucination rate for high confidence
        high_hall_rate = perfect_conf['is_hallucination'].mean() if len(perfect_conf) > 0 else 0
        high_hall_count = perfect_conf['is_hallucination'].sum() if len(perfect_conf) > 0 else 0
        
        # Create a pie chart of hallucination distribution in perfect confidence cases
        hall_counts = perfect_conf['is_hallucination'].value_counts()
        labels = [f"Correct\n{hall_counts.get(False, 0)} responses", f"Hallucination\n{hall_counts.get(True, 0)} responses"]
        colors = [COLORS['correct'], COLORS['hallucination']]
        
        # Only include labels that exist in the data
        existing_labels = []
        existing_colors = []
        sizes = []
        
        if False in hall_counts:
            existing_labels.append(f"Correct\n{hall_counts.get(False)} responses")
            existing_colors.append(COLORS['correct'])
            sizes.append(hall_counts.get(False))
            
        if True in hall_counts:
            existing_labels.append(f"Hallucination\n{hall_counts.get(True)} responses")
            existing_colors.append(COLORS['hallucination'])
            sizes.append(hall_counts.get(True))
        
        # Create pie chart
        high_conf_ax.pie(
            sizes,
            labels=existing_labels,
            colors=existing_colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(existing_labels)
        )
        
        # Add title
        conf_threshold = "≥ 0.98" if len(df[df[conf_col] >= 0.98]) >= 5 else "Top 10"
        high_conf_ax.set_title(
            f'{model_name}: Responses with High Confidence ({conf_threshold})\n'
            f'Total: {len(perfect_conf)} responses, Hallucination Rate: {high_hall_rate:.2f}',
            fontsize=12
        )
    
    # Add overall title
    fig.suptitle('Analysis of Extreme Confidence Cases', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(vis_dir / '3_confidence_extremes.png', dpi=300)
    plt.close()
    
    print("  - Generated confidence_distribution.png")
    print("  - Generated confidence_calibration.png")
    print("  - Generated confidence_calibration_detailed.png (enhanced)")
    print("  - Generated confidence_extremes.png (new)")
    print("  - Saved confidence_statistics.csv")
    print("  - Saved sector_confidence_breakdown.csv (new)")
    
    return stats_results

def analyze_confidence_nli(gpt4_data, gpt5_data):
    """
    Analyze the relationship between model confidence and entailment probabilities.
    
    Outputs:
    - confidence_nli_scatter.png: Enhanced scatter plot of confidence vs entailment
    - confidence_nli_correlation.csv: Detailed correlation statistics
    """
    print("\n4. Analyzing Confidence vs Entailment Probabilities...")
    
    # Calculate correlations
    correlations = {}
    
    # Fill any NaN values in entailment_prob with 0 to avoid missing data issues
    gpt4_data['entailment_prob'] = gpt4_data['entailment_prob'].fillna(0)
    gpt5_data['entailment_prob'] = gpt5_data['entailment_prob'].fillna(0)
    
    for model_name, df, conf_col in [('GPT-4', gpt4_data, 'gpt4_confidence'),
                                   ('GPT-5', gpt5_data, 'gpt5_confidence')]:
        # Clean data - remove NaN values only from confidence column
        clean_df = df.dropna(subset=[conf_col])
        
        if len(clean_df) < 10:
            print(f"  - Warning: Not enough clean data points for {model_name}. Skipping correlation.")
            correlations[f'{model_name} Pearson r'] = np.nan
            correlations[f'{model_name} p-value'] = np.nan
            continue
        
        # Calculate Pearson correlation
        pearson_r, p_value = stats.pearsonr(clean_df[conf_col], clean_df['entailment_prob'])
        correlations[f'{model_name} Pearson r'] = pearson_r
        correlations[f'{model_name} p-value'] = p_value
        
        # Calculate Spearman rank correlation (less affected by outliers)
        spearman_r, spearman_p = stats.spearmanr(clean_df[conf_col], clean_df['entailment_prob'])
        correlations[f'{model_name} Spearman r'] = spearman_r
        correlations[f'{model_name} Spearman p-value'] = spearman_p
        
        # Calculate for hallucinated and correct responses separately
        for hall_status, hall_value in [('Hallucinated', True), ('Correct', False)]:
            filtered = clean_df[clean_df['is_hallucination'] == hall_value]
            if len(filtered) > 10:  # Only if enough samples
                pearson_r, p_value = stats.pearsonr(filtered[conf_col], filtered['entailment_prob'])
                correlations[f'{model_name} {hall_status} Pearson r'] = pearson_r
                correlations[f'{model_name} {hall_status} p-value'] = p_value
    
    # Save correlations to CSV
    pd.DataFrame([correlations]).to_csv(csv_dir / '4_confidence_nli_correlation.csv', index=False)
    
    # Create an improved visualization
    fig = plt.figure(figsize=(15, 8))  # Reduced height slightly
    
    # Set up a 2x2 grid for more detailed analysis
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Create two main axes for the scatter plots
    axes_top = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    
    # Create two bottom axes for the marginal distributions
    axes_bottom = [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]
    
    # Process each model
    for i, (model_name, df, conf_col, color) in enumerate([
        ('GPT-4', gpt4_data, 'gpt4_confidence', COLORS['gpt4']),
        ('GPT-5', gpt5_data, 'gpt5_confidence', COLORS['gpt5'])
    ]):
        # Clean data by removing NaNs only from confidence column
        # We already filled NaN entailment values with 0
        clean_df = df.dropna(subset=[conf_col])
        
        if len(clean_df) < 10:
            # Show an error message on the plot if not enough data
            axes_top[i].text(0.5, 0.5, f"Insufficient data for {model_name}", 
                        ha='center', va='center', transform=axes_top[i].transAxes, 
                        fontsize=12, color='red')
            continue
        
        # Split data by hallucination status for better visualization
        hall_df = clean_df[clean_df['is_hallucination'] == True]
        correct_df = clean_df[clean_df['is_hallucination'] == False]
        
        # Create better scatter plot
        # For hallucinations - red triangles
        axes_top[i].scatter(
            hall_df[conf_col], 
            hall_df['entailment_prob'],
            color='red',
            marker='^',
            s=30,
            alpha=0.5,
            label='Hallucinations'
        )
        
        # For correct answers - blue circles
        axes_top[i].scatter(
            correct_df[conf_col], 
            correct_df['entailment_prob'],
            color='blue',
            marker='o',
            s=30,
            alpha=0.5,
            label='Correct Responses'
        )
        
        # Add regression line for all data
        x = clean_df[conf_col]
        y = clean_df['entailment_prob']
        
        try:
            m, b = np.polyfit(x, y, 1)
            x_line = np.array([min(x), max(x)])
            axes_top[i].plot(x_line, m*x_line + b, '--', color='black', linewidth=2, 
                         label=f'Trend (r={correlations[f"{model_name} Pearson r"]:.2f})')
        except Exception as e:
            print(f"  - Warning: Could not generate regression line for {model_name}: {e}")
        
        # Add quadrant labels to make interpretation clearer
        # Top right - High confidence, high entailment (good)
        axes_top[i].text(0.85, 0.85, "Justified\nConfidence", 
                     transform=axes_top[i].transAxes, ha='center', va='center',
                     bbox={'facecolor': 'lightgreen', 'alpha': 0.3, 'boxstyle': 'round'})
        
        # Bottom right - High confidence, low entailment (bad)
        axes_top[i].text(0.85, 0.15, "Overconfident\nHallucinations", 
                     transform=axes_top[i].transAxes, ha='center', va='center',
                     bbox={'facecolor': 'salmon', 'alpha': 0.3, 'boxstyle': 'round'})
        
        # Add histograms on the bottom to show confidence distribution
        axes_bottom[i].hist(hall_df[conf_col], bins=20, alpha=0.5, color='red', label='Hallucinations')
        axes_bottom[i].hist(correct_df[conf_col], bins=20, alpha=0.5, color='blue', label='Correct')
        
        # Add correlation statistics as text
        stats_text = (
            f"Pearson r = {correlations.get(f'{model_name} Pearson r', 'N/A'):.3f}"
            f" (p = {correlations.get(f'{model_name} p-value', 'N/A'):.3g})\n"
            f"Spearman ρ = {correlations.get(f'{model_name} Spearman r', 'N/A'):.3f}"
            f" (p = {correlations.get(f'{model_name} Spearman p-value', 'N/A'):.3g})"
        )
        
        axes_top[i].text(
            0.05, 0.05, 
            stats_text,
            transform=axes_top[i].transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8}
        )
        
        # Customize subplots
        axes_top[i].set_title(f'{model_name}: Confidence vs. Entailment', fontsize=14)
        axes_top[i].set_xlabel('Model Confidence', fontsize=12)
        axes_top[i].set_ylabel('Entailment Probability', fontsize=12)
        axes_top[i].grid(True, alpha=0.3)
        axes_top[i].legend(loc='upper left')
        axes_top[i].set_xlim(0, 1)
        axes_top[i].set_ylim(0, 1)
        
        # Format bottom histogram
        axes_bottom[i].set_xlabel('Model Confidence', fontsize=12)
        axes_bottom[i].set_ylabel('Count', fontsize=10)
        axes_bottom[i].set_xlim(0, 1)
        
        if i == 0:  # Only add legend to the first histogram to avoid duplication
            axes_bottom[i].legend(loc='upper right')
    
    # Add overall title
    fig.suptitle('Relationship Between Model Confidence and Entailment', 
                fontsize=16, y=0.98)
    
    # Save the visualization with tighter layout
    plt.tight_layout()
    plt.savefig(vis_dir / '4_confidence_nli_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  - Generated confidence_nli_scatter.png (improved)")
    print("  - Saved confidence_nli_correlation.csv (with correlation statistics)")
    
    return correlations

def analyze_cross_model_overlap(gpt4_data, gpt5_data):
    """
    Analyze the overlap and divergence in hallucinations between models.
    
    Outputs:
    - cross_model_overlap.png: Bar chart of overlap categories
    - cross_model_improvement.png: Visual of improvement vs regression
    - cross_model_analysis.csv: Detailed cross-model metrics
    """
    print("\n5. Analyzing Cross-Model Hallucination Overlap...")
    
    # Merge datasets for comparison
    merged_data = pd.merge(
        gpt4_data[['question', 'is_hallucination', 'gpt4_confidence', 'focus_area']],
        gpt5_data[['question', 'is_hallucination', 'gpt5_confidence']],
        on='question',
        suffixes=('_gpt4', '_gpt5')
    )
    
    # Calculate overlap categories
    categories = {
        'Both Correct': ((~merged_data['is_hallucination_gpt4']) & 
                       (~merged_data['is_hallucination_gpt5'])),
        'Only GPT-4 Hallucinated': (merged_data['is_hallucination_gpt4'] & 
                                  (~merged_data['is_hallucination_gpt5'])),
        'Only GPT-5 Hallucinated': ((~merged_data['is_hallucination_gpt4']) & 
                                  merged_data['is_hallucination_gpt5']),
        'Both Hallucinated': (merged_data['is_hallucination_gpt4'] & 
                            merged_data['is_hallucination_gpt5'])
    }
    
    # Calculate metrics
    overlap_metrics = {
        name: condition.mean() for name, condition in categories.items()
    }
    
    # Add counts
    for name, condition in categories.items():
        overlap_metrics[f'{name} Count'] = condition.sum()
    
    # Calculate improvement metrics
    overlap_metrics['Net Improvement'] = (
        overlap_metrics['Only GPT-4 Hallucinated'] - 
        overlap_metrics['Only GPT-5 Hallucinated']
    )
    
    # Save metrics to CSV
    pd.DataFrame([overlap_metrics]).to_csv(csv_dir / '5_cross_model_analysis.csv', index=False)
    
    # VISUALIZATION 1: Overlap categories
    plt.figure(figsize=(10, 6))
    
    # Prepare data for visualization
    category_names = list(categories.keys())
    values = [overlap_metrics[name] for name in category_names]
    
    # Create bar chart
    colors = [COLORS['correct'], COLORS['gpt4'], COLORS['gpt5'], COLORS['hallucination']]
    plt.bar(category_names, values, color=colors)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.1%}', ha='center')
        plt.text(i, v/2, f'n={overlap_metrics[category_names[i]+" Count"]}', 
                ha='center', color='white' if i==3 else 'black')
    
    # Customize plot
    plt.title('Cross-Model Hallucination Overlap', fontsize=14)
    plt.ylabel('Proportion of Samples', fontsize=12)
    plt.ylim(0, max(values) * 1.2)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '5_cross_model_overlap.png')
    plt.close()
    
    # VISUALIZATION 2: Improvement analysis
    plt.figure(figsize=(10, 6))
    
    # Prepare data for visualization
    improvement_data = {
        'Improved\n(GPT-4 → GPT-5)': overlap_metrics['Only GPT-4 Hallucinated'],
        'Regressed\n(GPT-5 worse)': overlap_metrics['Only GPT-5 Hallucinated'],
        'Net\nImprovement': overlap_metrics['Net Improvement']
    }
    
    # Create bar chart
    colors = ['green', 'red', 'blue' if improvement_data['Net\nImprovement'] > 0 else 'red']
    plt.bar(improvement_data.keys(), improvement_data.values(), color=colors)
    
    # Add value labels
    for i, (k, v) in enumerate(improvement_data.items()):
        plt.text(i, abs(v) + 0.01, f'{v:.1%}', ha='center')
    
    # Customize plot
    plt.title('GPT-5 Improvement vs. Regression', fontsize=14)
    plt.ylabel('Proportion of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '5_cross_model_improvement.png')
    plt.close()
    
    # VISUALIZATION 3: Sector-level improvement analysis
    # Calculate improvement by sector
    sector_improvement = merged_data.groupby('focus_area').apply(
        lambda x: (x['is_hallucination_gpt4'] & ~x['is_hallucination_gpt5']).sum() - 
                 (~x['is_hallucination_gpt4'] & x['is_hallucination_gpt5']).sum()
    ).reset_index()
    sector_improvement.columns = ['focus_area', 'net_improvement']
    
    # Add total count for percentage calculation
    sector_counts = merged_data.groupby('focus_area').size().reset_index()
    sector_counts.columns = ['focus_area', 'total']
    
    sector_improvement = pd.merge(sector_improvement, sector_counts, on='focus_area')
    sector_improvement['improvement_rate'] = sector_improvement['net_improvement'] / sector_improvement['total']
    
    # Sort and get top improving/regressing sectors
    sector_improvement = sector_improvement.sort_values('improvement_rate')
    top_sectors = pd.concat([
        sector_improvement.head(5),  # Most regressing
        sector_improvement.tail(5)   # Most improving
    ])
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    colors = ['red' if x < 0 else 'green' for x in top_sectors['improvement_rate']]
    plt.bar(top_sectors['focus_area'], top_sectors['improvement_rate'], color=colors)
    
    # Add value labels and sample counts
    for i, row in enumerate(top_sectors.itertuples()):
        plt.text(i, row.improvement_rate + (0.01 if row.improvement_rate >= 0 else -0.03),
                f'{row.improvement_rate:.1%}', ha='center')
        plt.text(i, 0, f'n={row.total}', ha='center', va='center', fontsize=8)
    
    # Customize plot
    plt.title('Sectors with Most Improvement/Regression', fontsize=14)
    plt.ylabel('Net Improvement Rate (GPT-5 vs GPT-4)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '5_sector_improvement.png')
    plt.close()
    
    print("  - Generated cross_model_overlap.png")
    print("  - Generated cross_model_improvement.png")
    print("  - Generated sector_improvement.png")
    print("  - Saved cross_model_analysis.csv")
    
    return overlap_metrics

def analyze_hallucination_severity(gpt4_data, gpt5_data):
    """
    Analyze the severity of hallucinations between models.
    
    Outputs:
    - hallucination_severity.png: Bar chart of severity categories
    - severity_by_focus.png: Heatmap of severity by focus area
    - severity_metrics.csv: Detailed severity metrics
    """
    print("\n6. Analyzing Hallucination Severity...")
    
    # Calculate severity metrics for each model
    severity_metrics = {}
    
    for model_name, df in [('GPT-4', gpt4_data), ('GPT-5', gpt5_data)]:
        # Calculate overall counts
        total = len(df)
        total_hall = df['is_hallucination'].sum()
        
        # Calculate counts and rates for each severity
        for severity in ['None', 'Mild', 'Severe']:
            count = (df['severity'] == severity).sum()
            severity_metrics[f'{model_name} {severity} Count'] = count
            severity_metrics[f'{model_name} {severity} Rate'] = count / total
        
        # Calculate severity distribution among hallucinations
        if total_hall > 0:
            severity_metrics[f'{model_name} Mild Among Hall'] = (
                (df['severity'] == 'Mild').sum() / total_hall
            )
            severity_metrics[f'{model_name} Severe Among Hall'] = (
                (df['severity'] == 'Severe').sum() / total_hall
            )
    
    # Save metrics to CSV
    pd.DataFrame([severity_metrics]).to_csv(csv_dir / '6_severity_metrics.csv', index=False)
    
    # VISUALIZATION 1: Severity distribution
    plt.figure(figsize=(10, 6))
    
    # Prepare data for visualization
    severity_types = ['None', 'Mild', 'Severe']
    gpt4_rates = [severity_metrics[f'GPT-4 {s} Rate'] for s in severity_types]
    gpt5_rates = [severity_metrics[f'GPT-5 {s} Rate'] for s in severity_types]
    
    # Create grouped bar chart
    x = np.arange(len(severity_types))
    width = 0.35
    
    plt.bar(x - width/2, gpt4_rates, width, label='GPT-4', color=COLORS['gpt4'])
    plt.bar(x + width/2, gpt5_rates, width, label='GPT-5', color=COLORS['gpt5'])
    
    # Add value labels
    for i, v in enumerate(gpt4_rates):
        plt.text(i - width/2, v + 0.01, f'{v:.1%}', ha='center')
        plt.text(i - width/2, v/2, f"n={severity_metrics[f'GPT-4 {severity_types[i]} Count']}", 
                ha='center', va='center', fontsize=8)
    
    for i, v in enumerate(gpt5_rates):
        plt.text(i + width/2, v + 0.01, f'{v:.1%}', ha='center')
        plt.text(i + width/2, v/2, f"n={severity_metrics[f'GPT-5 {severity_types[i]} Count']}", 
                ha='center', va='center', fontsize=8)
    
    # Customize plot
    plt.title('Hallucination Severity Distribution', fontsize=14)
    plt.ylabel('Proportion of Total Responses', fontsize=12)
    plt.xticks(x, severity_types)
    plt.legend()
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '6_hallucination_severity.png')
    plt.close()
    
    # VISUALIZATION 2: Severity among hallucinations
    plt.figure(figsize=(8, 5))
    
    # Prepare data for visualization
    hall_severity = ['Mild', 'Severe']
    gpt4_hall_rates = [severity_metrics[f'GPT-4 {s} Among Hall'] for s in hall_severity]
    gpt5_hall_rates = [severity_metrics[f'GPT-5 {s} Among Hall'] for s in hall_severity]
    
    # Create grouped bar chart
    x = np.arange(len(hall_severity))
    width = 0.35
    
    plt.bar(x - width/2, gpt4_hall_rates, width, label='GPT-4', color=COLORS['gpt4'])
    plt.bar(x + width/2, gpt5_hall_rates, width, label='GPT-5', color=COLORS['gpt5'])
    
    # Add value labels
    for i, v in enumerate(gpt4_hall_rates):
        plt.text(i - width/2, v + 0.01, f'{v:.1%}', ha='center')
    
    for i, v in enumerate(gpt5_hall_rates):
        plt.text(i + width/2, v + 0.01, f'{v:.1%}', ha='center')
    
    # Customize plot
    plt.title('Severity Distribution Among Hallucinations', fontsize=14)
    plt.ylabel('Proportion of Hallucinations', fontsize=12)
    plt.xticks(x, hall_severity)
    plt.legend()
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '6_severity_among_hallucinations.png')
    plt.close()
    
    # VISUALIZATION 3: Severity by focus area
    # Get top focus areas by sample size
    top_areas = gpt4_data['focus_area'].value_counts().head(8).index
    
    # Calculate severe hallucination rate by focus area
    focus_severity = []
    
    for model_name, df in [('GPT-4', gpt4_data), ('GPT-5', gpt5_data)]:
        for area in top_areas:
            area_data = df[df['focus_area'] == area]
            if len(area_data) > 0:
                severe_rate = (area_data['severity'] == 'Severe').mean()
                mild_rate = (area_data['severity'] == 'Mild').mean()
                focus_severity.append({
                    'Model': model_name,
                    'Focus Area': area,
                    'Severe Rate': severe_rate,
                    'Mild Rate': mild_rate,
                    'Sample Size': len(area_data)
                })
    
    focus_severity_df = pd.DataFrame(focus_severity)
    
    # Create pivot table for heatmap
    severity_pivot = focus_severity_df.pivot_table(
        index='Focus Area', 
        columns='Model', 
        values='Severe Rate'
    )
    
    plt.figure(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(severity_pivot, annot=True, fmt='.1%', cmap='YlOrRd')
    
    # Customize plot
    plt.title('Severe Hallucination Rate by Focus Area', fontsize=14)
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(vis_dir / '6_severity_by_focus.png')
    plt.close()
    
    print("  - Generated hallucination_severity.png")
    print("  - Generated severity_among_hallucinations.png")
    print("  - Generated severity_by_focus.png")
    print("  - Saved severity_metrics.csv")
    
    return severity_metrics

def analyze_information_source_comparison(gpt4_data, gpt5_data):
    """
    Analyze the reliability of different knowledge sources.
    
    Outputs:
    - source_reliability.png: Bar chart of hallucination rates by source
    - source_metrics.csv: Detailed information source comparison metrics
    """
    print("\n7. Analyzing Source-Level Reliability...")
    
    # Calculate information source comparison metrics
    def get_source_metrics(df, model_name):
        source_stats = df.groupby('source').agg({
            'is_hallucination': ['count', 'mean', 'std'],
        }).reset_index()
        
        source_stats.columns = ['source', 'sample_size', 'hallucination_rate', 'std']
        source_stats['model'] = model_name
        
        # Add severity metrics
        severity_stats = df.groupby('source').apply(
            lambda x: pd.Series({
                'severe_rate': (x['severity'] == 'Severe').mean(),
                'mild_rate': (x['severity'] == 'Mild').mean()
            })
        ).reset_index()
        
        source_stats = pd.merge(source_stats, severity_stats, on='source')
        return source_stats
    
    gpt4_sources = get_source_metrics(gpt4_data, 'GPT-4')
    gpt5_sources = get_source_metrics(gpt5_data, 'GPT-5')
    
    # Combine datasets
    all_sources = pd.concat([gpt4_sources, gpt5_sources])
    
    # Create pivot table for easier comparison
    source_comparison = all_sources.pivot(
        index='source', 
        columns='model',
        values=['hallucination_rate', 'sample_size', 'severe_rate']
    )
    
    # Flatten multi-level column index and replace hyphens with underscores to avoid attribute access issues
    source_comparison.columns = [f'{col[1]}_{col[0]}'.replace('-', '_') for col in source_comparison.columns]
    source_comparison = source_comparison.reset_index()
    
    # Calculate improvement
    source_comparison['improvement'] = (
        source_comparison['GPT_4_hallucination_rate'] - 
        source_comparison['GPT_5_hallucination_rate']
    )
    
    # Sort by GPT-4 hallucination rate
    source_comparison = source_comparison.sort_values('GPT_4_hallucination_rate', ascending=False)
    
    # Save to CSV
    source_comparison.to_csv(csv_dir / '7_source_metrics.csv', index=False)
    
    # VISUALIZATION 1: Completely redesigned information source comparison
    plt.figure(figsize=(14, 8))
    
    # Sort sources by GPT-4 hallucination rate for better visual comparison
    source_comparison_sorted = source_comparison.sort_values('GPT_4_hallucination_rate', ascending=False)
    
    # Set up horizontal bar chart (easier to read source names)
    sources = source_comparison_sorted['source']
    y_pos = np.arange(len(sources))
    bar_height = 0.35
    
    # Create horizontal bars for each model
    plt.barh(y_pos + bar_height/2, source_comparison_sorted['GPT_5_hallucination_rate'], 
            height=bar_height, label='GPT-5', color=COLORS['gpt5'], alpha=0.8)
    plt.barh(y_pos - bar_height/2, source_comparison_sorted['GPT_4_hallucination_rate'], 
            height=bar_height, label='GPT-4', color=COLORS['gpt4'], alpha=0.8)
    
    # Add percentage and sample size labels inside or next to each bar
    for i, row in enumerate(source_comparison_sorted.itertuples()):
        # GPT-4 percentage and sample size
        gpt4_value = row.GPT_4_hallucination_rate
        gpt4_label_pos = min(gpt4_value / 2, gpt4_value - 0.1) if gpt4_value > 0.2 else gpt4_value + 0.02
        plt.text(gpt4_label_pos, y_pos[i] - bar_height/2, 
                f"{gpt4_value:.1%} (n={row.GPT_4_sample_size})", 
                va='center', 
                ha='center' if gpt4_value > 0.2 else 'left',
                color='white' if gpt4_value > 0.2 else 'black',
                fontweight='bold', fontsize=9)
        
        # GPT-5 percentage and sample size
        gpt5_value = row.GPT_5_hallucination_rate
        gpt5_label_pos = min(gpt5_value / 2, gpt5_value - 0.1) if gpt5_value > 0.2 else gpt5_value + 0.02
        plt.text(gpt5_label_pos, y_pos[i] + bar_height/2, 
                f"{gpt5_value:.1%} (n={row.GPT_5_sample_size})", 
                va='center', 
                ha='center' if gpt5_value > 0.2 else 'left',
                color='white' if gpt5_value > 0.2 else 'black',
                fontweight='bold', fontsize=9)
        
        # Add improvement indicator
        if row.improvement > 0.05:  # Only mark significant improvements
            plt.text(max(row.GPT_4_hallucination_rate, row.GPT_5_hallucination_rate) + 0.03, 
                    y_pos[i], f"↓ {row.improvement:.1%}", ha='left', va='center', 
                    color='green', fontweight='bold')
    
    # Customize plot
    plt.title('Hallucination Rates by Information Source', fontsize=16)
    plt.xlabel('Hallucination Rate', fontsize=12)
    plt.yticks(y_pos, sources)
    plt.xlim(0, max(source_comparison['GPT_4_hallucination_rate'].max(), 
                   source_comparison['GPT_5_hallucination_rate'].max()) * 1.2)
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend with clear positioning
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='center', ncol=2, fontsize=10)
    
    # Add caption that explains the visualization clearly
    plt.figtext(0.5, 0.01, 
               "Lower hallucination rates are better. Green percentage shows improvement from GPT-4 to GPT-5.",
               ha='center', fontsize=10, style='italic')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '7_source_reliability.png')
    plt.close()
    
    # VISUALIZATION 2: Source improvement
    plt.figure(figsize=(10, 6))
    
    # Sort by improvement for this visualization
    improvement_df = source_comparison.sort_values('improvement')
    
    # Create bar chart of improvements
    colors = ['green' if x > 0 else 'red' for x in improvement_df['improvement']]
    plt.bar(improvement_df['source'], improvement_df['improvement'], color=colors)
    
    # Add value labels with better positioning
    for i, row in enumerate(improvement_df.itertuples()):
        # Use larger offset for very large positive values to prevent overlap with graph boundaries
        if row.improvement > 0.4:  # For values greater than 40%
            offset = 0.03  # Use larger offset
        else:
            offset = 0.01 if row.improvement >= 0 else -0.03
        
        plt.text(i, row.improvement + offset,
                f'{row.improvement:.1%}', ha='center', fontweight='bold')
    
    # Customize plot
    plt.title('Improvement by Information Source (GPT-4 to GPT-5)', fontsize=14)
    plt.ylabel('Improvement in Hallucination Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(vis_dir / '7_source_improvement.png')
    plt.close()
    
    print("  - Generated source_reliability.png")
    print("  - Generated source_improvement.png")
    print("  - Saved 7_source_metrics.csv")
    
    return source_comparison

def main():
    """Main function to run all analyses"""
    print("=== Hallucination Analysis Suite ===")
    
    # Load and preprocess data
    gpt4_data, gpt5_data = load_and_preprocess_data()
    
    # Run all analyses
    sector_data = analyze_sector_distribution(gpt4_data, gpt5_data)
    model_metrics = analyze_model_comparison(gpt4_data, gpt5_data)
    confidence_stats = analyze_confidence_hallucinations(gpt4_data, gpt5_data)
    nli_correlations = analyze_confidence_nli(gpt4_data, gpt5_data)
    overlap_metrics = analyze_cross_model_overlap(gpt4_data, gpt5_data)
    severity_metrics = analyze_hallucination_severity(gpt4_data, gpt5_data)
    source_data = analyze_information_source_comparison(gpt4_data, gpt5_data)
    
    # Create executive summary
    summary = {
        'Total Samples': len(gpt4_data),
        'GPT-4 Hallucination Rate': gpt4_data['is_hallucination'].mean(),
        'GPT-5 Hallucination Rate': gpt5_data['is_hallucination'].mean(),
        'Overall Improvement': (gpt4_data['is_hallucination'].mean() - 
                               gpt5_data['is_hallucination'].mean()),
        'Most Problematic Sector': sector_data.iloc[0]['focus_area'],
        'Most Improved Sector': source_data.sort_values('improvement', ascending=False).iloc[0]['source'],
        'Severe Hallucination Rate GPT-4': (gpt4_data['severity'] == 'Severe').mean(),
        'Severe Hallucination Rate GPT-5': (gpt5_data['severity'] == 'Severe').mean(),
        'Confidence-Hallucination Correlation GPT-4': confidence_stats['GPT-4 Effect Size (Cohen\'s d)'],
        'Confidence-Hallucination Correlation GPT-5': confidence_stats['GPT-5 Effect Size (Cohen\'s d)'],
        'Cross-Model Agreement Rate': overlap_metrics['Both Correct'] + overlap_metrics['Both Hallucinated']
    }
    
    pd.DataFrame([summary]).to_csv(csv_dir / 'executive_summary.csv', index=False)
    
    print("\nAnalysis Complete!")
    print(f"Results saved to {results_dir}")
    print(f"- {len(list(vis_dir.glob('*.png')))} visualizations in {vis_dir}")
    print(f"- {len(list(csv_dir.glob('*.csv')))} CSV data files in {csv_dir}")
    
    # Print key findings
    print("\nKey Findings:")
    print(f"- Overall hallucination rate: GPT-4 {summary['GPT-4 Hallucination Rate']:.1%} vs GPT-5 {summary['GPT-5 Hallucination Rate']:.1%}")
    print(f"- Net improvement: {summary['Overall Improvement']:.1%}")
    print(f"- Most problematic sector: {summary['Most Problematic Sector']}")
    print(f"- Most improved source: {summary['Most Improved Sector']}")

if __name__ == "__main__":
    main()
