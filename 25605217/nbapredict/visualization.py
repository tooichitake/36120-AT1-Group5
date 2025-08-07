"""Visualization utilities for data analysis and model evaluation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc

def set_visualization_style():
    """Set consistent visualization style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def plot_target_distribution(y: pd.Series, title: str = "Target Distribution"):
    """Plot target variable distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    counts = y.value_counts()
    colors = sns.color_palette("husl", n_colors=len(counts))
    
    ax1.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Drafted Status')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Count')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Not Drafted', 'Drafted'])
    
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    ax2.pie(counts.values, labels=['Not Drafted', 'Drafted'], 
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} - Percentage')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    colors = sns.color_palette("husl", n_colors=len(top_features))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Drafted', 'Drafted'],
                yticklabels=['Not Drafted', 'Drafted'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_correlation(df: pd.DataFrame, features: List[str], figsize: Tuple[int, int] = (12, 10)):
    """Plot correlation heatmap for selected features."""
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = df[features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    return fig

def plot_feature_distributions(df: pd.DataFrame, features: List[str], target_col: str = 'drafted'):
    """Plot distributions of features split by target variable."""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        for target_val in df[target_col].unique():
            data = df[df[target_col] == target_val][feature]
            label = 'Drafted' if target_val == 1 else 'Not Drafted'
            ax.hist(data, alpha=0.5, label=label, bins=30, edgecolor='black')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results_dict: dict):
    """Plot comparison of different models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[model].get(metric, 0) for model in models]
        colors = sns.color_palette("husl", n_colors=len(models))
        
        bars = ax.bar(models, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim([0, 1])
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def plot_draft_rate_by_conference(df: pd.DataFrame):
    """Plot draft rate by conference."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Conference stats
    conf_stats = df.groupby('conf').agg({
        'drafted': ['sum', 'count', 'mean']
    }).round(3)
    conf_stats.columns = ['Drafted_Count', 'Total_Players', 'Draft_Rate']
    conf_stats = conf_stats.sort_values('Draft_Rate', ascending=False).head(20)
    
    # Bar plot
    ax1 = axes[0]
    bars = ax1.barh(range(len(conf_stats)), conf_stats['Draft_Rate'].values)
    ax1.set_yticks(range(len(conf_stats)))
    ax1.set_yticklabels(conf_stats.index)
    ax1.set_xlabel('Draft Rate')
    ax1.set_title('Top 20 Conferences by Draft Rate')
    ax1.invert_yaxis()
    
    # Color bars by draft rate
    colors = plt.cm.RdYlGn(conf_stats['Draft_Rate'].values / conf_stats['Draft_Rate'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Power vs Non-Power comparison
    ax2 = axes[1]
    power_conferences = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']
    df['power_conf'] = df['conf'].isin(power_conferences)
    
    power_data = df.groupby('power_conf')['drafted'].value_counts().unstack()
    power_data.index = ['Non-Power', 'Power']
    
    power_data.plot(kind='bar', stacked=True, ax=ax2, color=['#FF6B6B', '#4ECDC4'])
    ax2.set_xlabel('Conference Type')
    ax2.set_ylabel('Number of Players')
    ax2.set_title('Draft Distribution: Power vs Non-Power Conferences')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.legend(['Not Drafted', 'Drafted'])
    
    plt.tight_layout()
    return fig

def plot_height_analysis(df: pd.DataFrame):
    """Plot height distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert height to inches if not already done
    if 'height_inches' not in df.columns:
        from src.dataset import convert_height_to_inches
        df = convert_height_to_inches(df)
    
    # 1. Violin plot
    ax1 = axes[0, 0]
    drafted_heights = df[df['drafted']==1]['height_inches'].dropna()
    undrafted_heights = df[df['drafted']==0]['height_inches'].dropna()
    
    parts = ax1.violinplot([undrafted_heights, drafted_heights],
                           positions=[0, 1], showmeans=True, showmedians=True)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Not Drafted', 'Drafted'])
    ax1.set_ylabel('Height (inches)')
    ax1.set_title('Height Distribution by Draft Status')
    
    # 2. KDE plot
    ax2 = axes[0, 1]
    for status, color, label in [(0, '#FF6B6B', 'Not Drafted'), 
                                  (1, '#4ECDC4', 'Drafted')]:
        data = df[df['drafted']==status]['height_inches'].dropna()
        data.plot(kind='kde', ax=ax2, color=color, label=label, linewidth=2)
    
    ax2.set_xlabel('Height (inches)')
    ax2.set_ylabel('Density')
    ax2.set_title('Height Density Distribution')
    ax2.legend()
    
    # 3. Box plot by position
    ax3 = axes[1, 0]
    if 'pos' in df.columns:
        positions = df['pos'].value_counts().head(5).index
        data_by_pos = [df[df['pos']==pos]['height_inches'].dropna() for pos in positions]
        
        bp = ax3.boxplot(data_by_pos, labels=positions, patch_artist=True)
        colors = sns.color_palette("husl", n_colors=len(positions))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Height (inches)')
        ax3.set_title('Height Distribution by Position')
    
    # 4. Height vs Draft Rate
    ax4 = axes[1, 1]
    height_bins = pd.cut(df['height_inches'], bins=10)
    height_draft_rate = df.groupby(height_bins)['drafted'].mean()
    
    ax4.bar(range(len(height_draft_rate)), height_draft_rate.values, 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Height Range')
    ax4.set_ylabel('Draft Rate')
    ax4.set_title('Draft Rate by Height Range')
    ax4.set_xticks(range(len(height_draft_rate)))
    ax4.set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in height_draft_rate.index], 
                        rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def visualize_power_conference(df: pd.DataFrame, target_col: str = 'drafted'):
    """Visualize power conference feature and its relationship with draft status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing power_conference and target column
    target_col : str
        Name of the target column (default: 'drafted')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Draft rate comparison between Power and Non-Power conferences
    ax1 = axes[0, 0]
    power_stats = df.groupby('power_conference')[target_col].agg(['mean', 'count'])
    power_stats.index = ['Non-Power', 'Power']
    
    bars = ax1.bar(power_stats.index, power_stats['mean'], color=['#FF6B6B', '#4ECDC4'], 
                   edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Draft Rate')
    ax1.set_title('Draft Rate: Power vs Non-Power Conferences')
    ax1.set_ylim([0, power_stats['mean'].max() * 1.2])
    
    # Add value labels on bars
    for bar, val in zip(bars, power_stats['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add count labels below
    for i, (idx, row) in enumerate(power_stats.iterrows()):
        ax1.text(i, -0.001, f'n={row["count"]:,}', ha='center', va='top', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Distribution of drafted vs not drafted by conference type
    ax2 = axes[0, 1]
    conf_data = df.groupby(['power_conference', target_col]).size().unstack(fill_value=0)
    conf_data.index = ['Non-Power', 'Power']
    
    x = np.arange(len(conf_data.index))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, conf_data[0], width, label='Not Drafted', 
                    color='#FF6B6B', alpha=0.7)
    bars2 = ax2.bar(x + width/2, conf_data[1], width, label='Drafted', 
                    color='#4ECDC4', alpha=0.7)
    
    ax2.set_xlabel('Conference Type')
    ax2.set_ylabel('Number of Players')
    ax2.set_title('Player Distribution by Conference Type and Draft Status')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conf_data.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Top conferences by draft count
    ax3 = axes[1, 0]
    if 'conf' in df.columns:
        conf_draft_stats = df.groupby('conf')[target_col].agg(['sum', 'count', 'mean'])
        conf_draft_stats = conf_draft_stats[conf_draft_stats['sum'] > 0].sort_values('sum', ascending=False).head(15)
        
        bars = ax3.barh(range(len(conf_draft_stats)), conf_draft_stats['sum'], 
                        color='steelblue', edgecolor='black', alpha=0.7)
        ax3.set_yticks(range(len(conf_draft_stats)))
        ax3.set_yticklabels(conf_draft_stats.index)
        ax3.set_xlabel('Number of Drafted Players')
        ax3.set_title('Top 15 Conferences by Number of Drafted Players')
        ax3.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, conf_draft_stats['sum']):
            ax3.text(val, bar.get_y() + bar.get_height()/2., f'{int(val)}', 
                    ha='left', va='center', fontsize=9)
        
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Statistical comparison of features by conference type
    ax4 = axes[1, 1]
    
    # Select key performance metrics
    metrics = ['Min_per', 'bpm', 'AST_per', 'TS_per']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        power_metrics = df[df['power_conference'] == 1][available_metrics].mean()
        non_power_metrics = df[df['power_conference'] == 0][available_metrics].mean()
        
        x = np.arange(len(available_metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, non_power_metrics, width, label='Non-Power', 
                       color='#FF6B6B', alpha=0.7)
        bars2 = ax4.bar(x + width/2, power_metrics, width, label='Power', 
                       color='#4ECDC4', alpha=0.7)
        
        ax4.set_xlabel('Performance Metrics')
        ax4.set_ylabel('Average Value')
        ax4.set_title('Average Performance Metrics: Power vs Non-Power')
        ax4.set_xticks(x)
        ax4.set_xticklabels(available_metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Power Conference Feature Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("POWER CONFERENCE SUMMARY STATISTICS")
    print("="*60)
    
    # Power conference distribution
    power_dist = df['power_conference'].value_counts()
    print(f"\nConference Distribution:")
    print(f"  Non-Power Conferences: {power_dist.get(0, 0):,} players ({power_dist.get(0, 0)/len(df):.1%})")
    print(f"  Power Conferences: {power_dist.get(1, 0):,} players ({power_dist.get(1, 0)/len(df):.1%})")
    
    # Draft rates
    draft_by_power = df.groupby('power_conference')[target_col].mean()
    print(f"\nDraft Rates:")
    print(f"  Non-Power: {draft_by_power.get(0, 0):.2%}")
    print(f"  Power: {draft_by_power.get(1, 0):.2%}")
    print(f"  Ratio: {draft_by_power.get(1, 0)/draft_by_power.get(0, 0.001):.1f}x higher in Power conferences")
    
    # Top power conferences
    if 'conf' in df.columns:
        power_confs = ['B10', 'B12', 'ACC', 'SEC', 'P12', 'BE']
        print(f"\nPower Conferences: {', '.join(power_confs)}")
        
        # Draft counts by power conference
        power_conf_drafts = df[df['conf'].isin(power_confs)].groupby('conf')[target_col].agg(['sum', 'mean'])
        power_conf_drafts = power_conf_drafts.sort_values('sum', ascending=False)
        print(f"\nDrafted Players by Power Conference:")
        for conf, row in power_conf_drafts.iterrows():
            print(f"  {conf}: {int(row['sum'])} drafted ({row['mean']:.2%} draft rate)")
    
    print("="*60)
    
    return fig

def visualize_usage_efficiency(df: pd.DataFrame, target_col: str = 'drafted'):
    """Visualize usage efficiency feature distribution and relationship with draft status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing usage_efficiency and target column
    target_col : str
        Name of the target column (default: 'drafted')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution by draft status (Violin plot)
    ax1 = axes[0, 0]
    drafted_data = df[df[target_col] == 1]['usage_efficiency'].dropna()
    undrafted_data = df[df[target_col] == 0]['usage_efficiency'].dropna()
    
    parts = ax1.violinplot([undrafted_data, drafted_data], 
                           positions=[0, 1], showmeans=True, showmedians=True)
    
    # Color the violin plots
    colors = ['#FF6B6B', '#4ECDC4']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Not Drafted', 'Drafted'])
    ax1.set_ylabel('Usage Efficiency (usg × TS%)')
    ax1.set_title('Usage Efficiency Distribution by Draft Status')
    ax1.grid(True, alpha=0.3)
    
    # Add mean values as text
    ax1.text(0, undrafted_data.mean(), f'Mean: {undrafted_data.mean():.1f}', 
             ha='center', va='bottom', fontweight='bold')
    ax1.text(1, drafted_data.mean(), f'Mean: {drafted_data.mean():.1f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # 2. Histogram with overlaid distributions
    ax2 = axes[0, 1]
    ax2.hist(undrafted_data, bins=50, alpha=0.5, label='Not Drafted', 
             color='#FF6B6B', edgecolor='black', density=True)
    ax2.hist(drafted_data, bins=30, alpha=0.5, label='Drafted', 
             color='#4ECDC4', edgecolor='black', density=True)
    ax2.set_xlabel('Usage Efficiency')
    ax2.set_ylabel('Density')
    ax2.set_title('Usage Efficiency Density Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[0, 2]
    box_data = [undrafted_data, drafted_data]
    bp = ax3.boxplot(box_data, labels=['Not Drafted', 'Drafted'], 
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Usage Efficiency')
    ax3.set_title('Usage Efficiency Box Plot Comparison')
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: Usage vs True Shooting with efficiency as size
    ax4 = axes[1, 0]
    if 'usg' in df.columns and 'TS_per' in df.columns:
        for status, color, label in [(0, '#FF6B6B', 'Not Drafted'), 
                                     (1, '#4ECDC4', 'Drafted')]:
            mask = df[target_col] == status
            ax4.scatter(df[mask]['usg'], df[mask]['TS_per'], 
                       s=df[mask]['usage_efficiency']/10, 
                       alpha=0.5, color=color, label=label, edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('Usage Rate (%)')
        ax4.set_ylabel('True Shooting (%)')
        ax4.set_title('Usage vs True Shooting (Size = Usage Efficiency)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Percentile analysis
    ax5 = axes[1, 1]
    percentiles = [25, 50, 75, 90, 95, 99]
    drafted_percentiles = [np.percentile(drafted_data, p) for p in percentiles]
    undrafted_percentiles = [np.percentile(undrafted_data, p) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, undrafted_percentiles, width, 
                    label='Not Drafted', color='#FF6B6B', alpha=0.7)
    bars2 = ax5.bar(x + width/2, drafted_percentiles, width, 
                    label='Drafted', color='#4ECDC4', alpha=0.7)
    
    ax5.set_xlabel('Percentile')
    ax5.set_ylabel('Usage Efficiency Value')
    ax5.set_title('Usage Efficiency by Percentile')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{p}th' for p in percentiles])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Draft rate by usage efficiency bins
    ax6 = axes[1, 2]
    usage_eff_bins = pd.qcut(df['usage_efficiency'], q=10, duplicates='drop')
    draft_rate_by_bin = df.groupby(usage_eff_bins)[target_col].mean()
    
    bars = ax6.bar(range(len(draft_rate_by_bin)), draft_rate_by_bin.values, 
                   color='steelblue', edgecolor='black', alpha=0.7)
    
    # Color bars based on draft rate
    colors_bar = plt.cm.RdYlGn(draft_rate_by_bin.values / draft_rate_by_bin.max())
    for bar, color in zip(bars, colors_bar):
        bar.set_color(color)
    
    ax6.set_xlabel('Usage Efficiency Decile')
    ax6.set_ylabel('Draft Rate')
    ax6.set_title('Draft Rate by Usage Efficiency Decile')
    ax6.set_xticks(range(len(draft_rate_by_bin)))
    ax6.set_xticklabels([f'D{i+1}' for i in range(len(draft_rate_by_bin))], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, draft_rate_by_bin.values)):
        ax6.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Usage Efficiency Feature Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("USAGE EFFICIENCY SUMMARY STATISTICS")
    print("="*60)
    print(f"\nNot Drafted Players:")
    print(f"  Mean: {undrafted_data.mean():.2f}")
    print(f"  Median: {undrafted_data.median():.2f}")
    print(f"  Std Dev: {undrafted_data.std():.2f}")
    print(f"  Min: {undrafted_data.min():.2f}")
    print(f"  Max: {undrafted_data.max():.2f}")
    
    print(f"\nDrafted Players:")
    print(f"  Mean: {drafted_data.mean():.2f}")
    print(f"  Median: {drafted_data.median():.2f}")
    print(f"  Std Dev: {drafted_data.std():.2f}")
    print(f"  Min: {drafted_data.min():.2f}")
    print(f"  Max: {drafted_data.max():.2f}")
    
    print(f"\nDifference (Drafted - Not Drafted):")
    print(f"  Mean difference: {drafted_data.mean() - undrafted_data.mean():.2f}")
    print(f"  Median difference: {drafted_data.median() - undrafted_data.median():.2f}")
    
    # Statistical test
    from scipy import stats
    statistic, p_value = stats.mannwhitneyu(undrafted_data, drafted_data, alternative='two-sided')
    print(f"\nMann-Whitney U Test:")
    print(f"  Statistic: {statistic:.2f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    print("="*60)
    
    return fig

def evaluate_model_with_visualizations(y_true, y_pred_proba, y_pred_binary=None, model_name="Model"):
    """Evaluate model with comprehensive visualizations."""
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Drafted', 'Drafted'],
                yticklabels=['Not Drafted', 'Drafted'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'{model_name} - Confusion Matrix')
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')
    
    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc_score:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} - ROC Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    ax3 = axes[1, 0]
    ax3.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, 
            label='Not Drafted', color='#FF6B6B', edgecolor='black')
    ax3.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, 
            label='Drafted', color='#4ECDC4', edgecolor='black')
    ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{model_name} - Prediction Distribution')
    ax3.legend()
    
    # 4. Calibration Plot
    ax4 = axes[1, 1]
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    
    ax4.plot(mean_pred, fraction_pos, marker='o', linewidth=2, 
            label=model_name, color='steelblue')
    ax4.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title(f'{model_name} - Calibration Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} Performance Evaluation', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred_binary, 
                               target_names=['Not Drafted', 'Drafted']))
    
    return fig

def analyze_feature_by_draft_status(df: pd.DataFrame, feature: str, feature_name: str = None, 
                                   target_col: str = 'drafted'):
    """Analyze any feature by draft status with comprehensive visualizations."""
    if feature not in df.columns:
        print(f"Feature {feature} not found in dataframe")
        return None
    
    if feature_name is None:
        feature_name = feature.replace('_', ' ').title()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. Violin plot
    ax1 = axes[0, 0]
    violin_parts = ax1.violinplot(
        [df[df[target_col]==0][feature].dropna(),
         df[df[target_col]==1][feature].dropna()],
        positions=[0, 1], showmeans=True, showmedians=True
    )
    
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Not Drafted', 'Drafted'])
    ax1.set_ylabel(feature_name, fontsize=12)
    ax1.set_title(f'{feature_name} Distribution by Draft Status', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. KDE plot
    ax2 = axes[0, 1]
    for status, color, label in [(0, '#FF6B6B', 'Not Drafted'), (1, '#4ECDC4', 'Drafted')]:
        data = df[df[target_col]==status][feature].dropna()
        if len(data) > 0:
            data.plot.kde(ax=ax2, label=label, color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel(feature_name, fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'{feature_name} Density Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot with stats
    ax3 = axes[1, 0]
    bp = ax3.boxplot(
        [df[df[target_col]==0][feature].dropna(),
         df[df[target_col]==1][feature].dropna()],
        labels=['Not Drafted', 'Drafted'],
        patch_artist=True,
        notch=True
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel(feature_name, fontsize=12)
    ax3.set_title(f'{feature_name} Box Plot Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histogram comparison
    ax4 = axes[1, 1]
    for status, color, label in [(0, '#FF6B6B', 'Not Drafted'), (1, '#4ECDC4', 'Drafted')]:
        data = df[df[target_col]==status][feature].dropna()
        ax4.hist(data, bins=30, alpha=0.5, label=label, color=color, edgecolor='black')
    
    ax4.set_xlabel(feature_name, fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'{feature_name} Histogram Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    
    plt.suptitle(f'{feature_name} Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{feature_name} Statistics by Draft Status:")
    print("="*50)
    print(df.groupby(target_col)[feature].describe().round(2))
    
    return fig

def plot_feature_correlations(df: pd.DataFrame, target_col: str = 'drafted', top_n: int = 30):
    """Plot feature correlations with target variable."""
    # Get numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-feature columns
    exclude_cols = [target_col, 'player_id', 'height_inches', 'power_conf', 'year_numeric']
    numeric_features = [f for f in numeric_features if f not in exclude_cols]
    
    # Calculate correlations with target
    correlations = pd.DataFrame({
        'feature': numeric_features,
        'correlation': [df[feature].corr(df[target_col]) for feature in numeric_features]
    })
    
    # Sort by absolute correlation
    correlations['abs_correlation'] = correlations['correlation'].abs()
    correlations = correlations.sort_values('abs_correlation', ascending=False).head(top_n)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Bar plot of correlations
    ax1 = axes[0]
    colors_list = ['green' if x > 0 else 'red' for x in correlations['correlation'].values]
    bars = ax1.barh(range(len(correlations)), correlations['correlation'].values, color=colors_list, alpha=0.7)
    ax1.set_yticks(range(len(correlations)))
    ax1.set_yticklabels(correlations['feature'].values)
    ax1.set_xlabel('Correlation with Draft Status', fontsize=12)
    ax1.set_title(f'Top {top_n} Feature Correlations', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, correlations['correlation'].values):
        ax1.text(val + 0.005 if val > 0 else val - 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left' if val > 0 else 'right', va='center', fontsize=9)
    
    # 2. Heatmap of top features correlation matrix
    ax2 = axes[1]
    top_features = correlations['feature'].head(15).tolist()
    corr_matrix = df[top_features].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, ax=ax2,
                cbar_kws={'label': 'Correlation'})
    ax2.set_title('Top 15 Features Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    print(f"\nTop {top_n} Features by Correlation with Draft Status:")
    print("="*50)
    print(correlations[['feature', 'correlation']].to_string(index=False))
    
    return fig, correlations

def visualize_all_around_score(df: pd.DataFrame, target_col: str = 'drafted'):
    """Visualize all-around score feature with comprehensive analysis."""
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. All-around score distribution
    ax1 = axes[0, 0]
    if 'all_around_score' in df.columns:
        sns.kdeplot(data=df[df[target_col]==0]['all_around_score'].dropna(), 
                    ax=ax1, color='#FF6B6B', fill=True, alpha=0.6, label='Not Drafted', linewidth=2)
        sns.kdeplot(data=df[df[target_col]==1]['all_around_score'].dropna(), 
                    ax=ax1, color='#4ECDC4', fill=True, alpha=0.6, label='Drafted', linewidth=2)
        
        ax1.set_xlabel('All-Around Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('All-Around Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
    
    # 2. Component breakdown
    ax2 = axes[0, 1]
    components = ['AST_per', 'rebound_total', 'defensive_stats']
    available_components = [c for c in components if c in df.columns]
    
    if available_components:
        drafted_means = [df[df[target_col]==1][comp].mean() for comp in available_components]
        not_drafted_means = [df[df[target_col]==0][comp].mean() for comp in available_components]
        
        x = np.arange(len(available_components))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, not_drafted_means, width, label='Not Drafted', 
                       color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax2.bar(x + width/2, drafted_means, width, label='Drafted',
                       color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Assists %', 'Total Rebounds %', 'Defensive Stats %'][:len(available_components)])
        ax2.set_ylabel('Average Value', fontsize=12)
        ax2.set_title('All-Around Score Components Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Radar chart comparison
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    
    radar_features = ['AST_per', 'ORB_per', 'DRB_per', 'stl_per', 'blk_per', 'TS_per']
    available_radar = [f for f in radar_features if f in df.columns]
    
    if available_radar:
        labels = ['Assists', 'Off Reb', 'Def Reb', 'Steals', 'Blocks', 'True Shooting'][:len(available_radar)]
        
        drafted_values = [df[df[target_col]==1][feat].mean() for feat in available_radar]
        not_drafted_values = [df[df[target_col]==0][feat].mean() for feat in available_radar]
        
        # Normalize to 0-1 scale
        max_values = [max(d, nd) if max(d, nd) > 0 else 1 for d, nd in zip(drafted_values, not_drafted_values)]
        drafted_norm = [d/m for d, m in zip(drafted_values, max_values)]
        not_drafted_norm = [nd/m for nd, m in zip(not_drafted_values, max_values)]
        
        # Plot
        angles = np.linspace(0, 2 * np.pi, len(available_radar), endpoint=False).tolist()
        drafted_norm += drafted_norm[:1]
        not_drafted_norm += not_drafted_norm[:1]
        angles += angles[:1]
        
        ax3.plot(angles, drafted_norm, 'o-', linewidth=2, color='#4ECDC4', label='Drafted')
        ax3.fill(angles, drafted_norm, alpha=0.25, color='#4ECDC4')
        ax3.plot(angles, not_drafted_norm, 'o-', linewidth=2, color='#FF6B6B', label='Not Drafted')
        ax3.fill(angles, not_drafted_norm, alpha=0.25, color='#FF6B6B')
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(labels)
        ax3.set_title('Player Profile Comparison', fontsize=14, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # 4. Feature relationship analysis
    ax4 = axes[1, 1]
    feature_pairs = ['all_around_score', 'usage_efficiency', 'minutes_impact']
    available_pairs = [f for f in feature_pairs if f in df.columns]
    
    if len(available_pairs) >= 2:
        subset_df = df[available_pairs + [target_col]].dropna()
        
        # Create custom scatter plot
        for status, color, marker in [(0, '#FF6B6B', 'o'), (1, '#4ECDC4', '^')]:
            data = subset_df[subset_df[target_col]==status]
            if len(data) > 0:
                ax4.scatter(data[available_pairs[0]], data[available_pairs[1]], 
                          s=30, c=color, marker=marker, alpha=0.5, 
                          edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel(available_pairs[0].replace('_', ' ').title(), fontsize=12)
        ax4.set_ylabel(available_pairs[1].replace('_', ' ').title(), fontsize=12)
        ax4.set_title('Feature Relationship Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                                 markersize=8, label='Not Drafted'),
                           Line2D([0], [0], marker='^', color='w', markerfacecolor='#4ECDC4', 
                                 markersize=8, label='Drafted')]
        ax4.legend(handles=legend_elements)
    
    plt.suptitle('Comprehensive Player Value Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Print statistics
    if 'all_around_score' in df.columns:
        print("All-Around Score by draft status:")
        print("="*50)
        print(df.groupby(target_col)['all_around_score'].describe().round(1))
    
    return fig

def visualize_rare_skills(df: pd.DataFrame, target_col: str = 'drafted'):
    """Visualize rare skill indicators with comprehensive analysis."""
    # Create visualization for rare skills
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Big man shooter analysis
    ax1 = axes[0, 0]
    if 'big_man_shooter' in df.columns:
        big_man_stats = df.groupby('big_man_shooter')[target_col].agg(['mean', 'count'])
        colors = ['#95A5A6', '#E74C3C']
        
        bars = ax1.bar(big_man_stats.index, big_man_stats['mean'], 
                      color=colors, edgecolor='black', linewidth=2, alpha=0.8)
        
        # Add value labels
        for bar, (rate, count) in zip(bars, zip(big_man_stats['mean'], big_man_stats['count'])):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.1%}\n(n={count})', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Regular Players', 'Big Man Shooters'])
        ax1.set_ylabel('Draft Rate', fontsize=12)
        ax1.set_title('Draft Success: Big Men Who Can Shoot (>6\'8", >35% 3PT)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        if big_man_stats['mean'].max() > 0:
            ax1.set_ylim(0, big_man_stats['mean'].max() * 1.3)
    
    # 2. Playmaking big analysis
    ax2 = axes[0, 1]
    if 'playmaking_big' in df.columns:
        playmaking_stats = df.groupby('playmaking_big')[target_col].agg(['mean', 'count'])
        
        bars = ax2.bar(playmaking_stats.index, playmaking_stats['mean'], 
                      color=['#95A5A6', '#3498DB'], edgecolor='black', linewidth=2, alpha=0.8)
        
        for bar, (rate, count) in zip(bars, zip(playmaking_stats['mean'], playmaking_stats['count'])):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.1%}\n(n={count})', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Regular Players', 'Playmaking Bigs'])
        ax2.set_ylabel('Draft Rate', fontsize=12)
        ax2.set_title('Draft Success: Playmaking Big Men (>6\'8", >20 AST%)', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        if playmaking_stats['mean'].max() > 0:
            ax2.set_ylim(0, playmaking_stats['mean'].max() * 1.3)
    
    # 3. Height vs 3PT% scatter with rare skills highlighted
    ax3 = axes[1, 0]
    if 'height_inches' in df.columns and 'TP_per' in df.columns:
        # Regular players
        if 'big_man_shooter' in df.columns and 'playmaking_big' in df.columns:
            regular = df[(df['big_man_shooter']==0) & (df['playmaking_big']==0)]
            ax3.scatter(regular['height_inches'], regular['TP_per'], 
                       c='gray', alpha=0.3, s=20, label='Regular')
            
            # Big man shooters
            bms = df[df['big_man_shooter']==1]
            if len(bms) > 0:
                ax3.scatter(bms['height_inches'], bms['TP_per'], 
                           c='#E74C3C', s=100, marker='^', edgecolors='black', 
                           linewidth=1, label='Big Man Shooters', alpha=0.8)
            
            # Playmaking bigs
            pmb = df[df['playmaking_big']==1]
            if len(pmb) > 0 and 'AST_per' in df.columns:
                ax3.scatter(pmb['height_inches'], pmb['AST_per'], 
                           c='#3498DB', s=100, marker='s', edgecolors='black', 
                           linewidth=1, label='Playmaking Bigs', alpha=0.8)
        
        # Add reference lines
        ax3.axhline(35, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(80, color='black', linestyle='--', alpha=0.5)
        ax3.text(81, 36, "Big Man Shooter\nThreshold", fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Height (inches)', fontsize=12)
        ax3.set_ylabel('3-Point % / Assist %', fontsize=12)
        ax3.set_title('Identifying Rare NBA Skills', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(68, 88)
    
    # 4. Recruit rank availability impact
    ax4 = axes[1, 1]
    if 'has_recruit_rank' in df.columns:
        recruit_stats = df.groupby('has_recruit_rank')[target_col].agg(['mean', 'count'])
        
        # Create stacked bar chart
        draft_by_recruit = df.groupby(['has_recruit_rank', target_col]).size().unstack(fill_value=0)
        draft_by_recruit.index = ['No Rank', 'Has Rank']
        
        if not draft_by_recruit.empty:
            draft_by_recruit.plot(kind='bar', stacked=True, ax=ax4, 
                                  color=['#FF6B6B', '#4ECDC4'],
                                  edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Add percentage labels
            for i, (idx, row) in enumerate(draft_by_recruit.iterrows()):
                total = row.sum()
                draft_pct = row[1] / total * 100 if 1 in row and total > 0 else 0
                ax4.text(i, total + 20, f'{draft_pct:.1f}%\n({int(total)} players)', 
                        ha='center', fontweight='bold')
            
            ax4.set_xlabel('Recruit Rank Status', fontsize=12)
            ax4.set_ylabel('Number of Players', fontsize=12)
            ax4.set_title('Impact of Having Recruit Ranking Data', fontsize=14, fontweight='bold')
            ax4.legend(['Not Drafted', 'Drafted'], fontsize=11)
            ax4.set_xticklabels(['No Rank', 'Has Rank'], rotation=0)
            ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Rare Skills and Special Indicators Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Summary statistics
    print("Rare Skill Draft Rates:")
    print("="*50)
    if 'big_man_shooter' in df.columns and 'playmaking_big' in df.columns:
        regular_rate = df[(df['big_man_shooter']==0) & (df['playmaking_big']==0)][target_col].mean()
        big_shooter_rate = df[df['big_man_shooter']==1][target_col].mean()
        playmaking_rate = df[df['playmaking_big']==1][target_col].mean()
        
        print(f"Regular players: {regular_rate:.1%}")
        print(f"Big man shooters: {big_shooter_rate:.1%}")
        print(f"Playmaking bigs: {playmaking_rate:.1%}")
    
    if 'has_recruit_rank' in df.columns:
        with_rank_rate = df[df['has_recruit_rank']==1][target_col].mean()
        without_rank_rate = df[df['has_recruit_rank']==0][target_col].mean()
        
        print(f"\nPlayers with recruit rank: {with_rank_rate:.1%}")
        print(f"Players without recruit rank: {without_rank_rate:.1%}")
    
    return fig

def evaluate_model(y_true, y_pred_proba, y_pred_binary=None, title="Model Performance"):
    """Comprehensive model evaluation with beautiful visualizations."""
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
    
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    print(f"\n{title}")
    print("="*50)
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=['Not Drafted', 'Drafted']))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Not Drafted', 'Drafted'],
                yticklabels=['Not Drafted', 'Drafted'],
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Add percentage annotations
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            axes[0].text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                        ha='center', va='center', fontsize=10, color='gray')
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    axes[1].plot(fpr, tpr, color='#2E86AB', linewidth=3, 
                 label=f'ROC curve (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Probability distribution
    axes[2].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, 
                 label='Not Drafted', color='#FF6B6B', density=True, 
                 edgecolor='black', linewidth=1)
    axes[2].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, 
                 label='Drafted', color='#4ECDC4', density=True,
                 edgecolor='black', linewidth=1)
    axes[2].axvline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('Predicted Probability', fontsize=12)
    axes[2].set_ylabel('Density', fontsize=12)
    axes[2].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - AUC: {auc_score:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return auc_score