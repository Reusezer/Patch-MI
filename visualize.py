#!/usr/bin/env python
# demo_read_patch_out.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termplotlib import figure
import textwrap
import sys
from io import StringIO

def text_visualize(patch_dir: Path):
    """Create text-based visualizations for console output"""
    prompt_df = pd.read_csv(patch_dir/"prompt_results.csv")
    layer_df = pd.read_csv(patch_dir/"layer_stats.csv")
    neuron_df = pd.read_csv(patch_dir/"neuron_stats.csv")

    # 1. Layer Distribution ASCII plot
    print("\n📊 Layer Activity Distribution:")
    fig = figure()
    fig.barh(layer_df['count'].values, labels=[f"L{i}" for i in layer_df['layer']])
    fig.show()

    # 2. Top Neurons Summary
    print("\n🔥 Most Active Neurons:")
    top_neurons = neuron_df.nlargest(5, 'count')
    for _, row in top_neurons.iterrows():
        print(f"Layer {row['layer']:2d} Neuron {row['neuron_idx']:4d}: {'█' * int(row['count']/5)} ({row['count']})")

    # 3. KL Divergence Stats
    if 'kl_patched' in prompt_df.columns:
        print("\n📉 KL Divergence Summary:")
        print(f"Mean: {prompt_df['kl_patched'].mean():.4f}")
        print(f"Std:  {prompt_df['kl_patched'].std():.4f}")
        print(f"Max:  {prompt_df['kl_patched'].max():.4f}")
        print(f"Min:  {prompt_df['kl_patched'].min():.4f}")

    # 4. Pair Gap Text Summary
    gap_files = list(patch_dir.glob("pairgap_*.csv"))
    if gap_files:
        print("\n🔍 Pair Gap Analysis:")
        for gf in gap_files:
            try:
                df_gap = pd.read_csv(gf, index_col=0)  # use the first column as index
            except Exception as e:
                print(f"❌  Could not load {gf}: {e}")
                continue
            
            q25, q50, q75 = df_gap['kl_gap'].quantile([0.25, 0.5, 0.75])
            print(f"\n{gf.stem}:")
            print(f"Quartiles: [{q25:.3f} | {q50:.3f} | {q75:.3f}]")
            print("Distribution:", end=" ")
            # Simple ASCII distribution
            hist, _ = np.histogram(df_gap['kl_gap'], bins=20)
            max_height = 10
            scaled = (hist * max_height / hist.max()).astype(int)
            print("\n" + "\n".join("".join("█" if i < h else " " for i in range(max_height)) for h in scaled))

def analyze_kl_divergence(prompt_df: pd.DataFrame, neuron_df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize KL divergence patterns"""
    if 'kl_patched' not in prompt_df.columns:
        return
    
    # Create regular and log-transformed KL plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regular KL distribution
    sns.histplot(data=prompt_df, x='kl_patched', bins=30, ax=ax1)
    ax1.set_title('Distribution of KL Divergence')
    ax1.set_xlabel('KL Divergence')
    ax1.set_ylabel('Count')
    
    # Log-transformed KL distribution
    log_kl = np.log1p(prompt_df['kl_patched'])  # log1p to handle zeros
    sns.histplot(data=pd.DataFrame({'log_kl': log_kl}), x='log_kl', bins=30, ax=ax2)
    ax2.set_title('Log-transformed KL Divergence')
    ax2.set_xlabel('log(1 + KL)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir/'kl_distributions.png')
    plt.close()
    
    # Analyze high-impact neurons
    if 'neuron_idx' in prompt_df.columns:
        high_kl = prompt_df.nlargest(10, 'kl_patched')
        impact_summary = pd.DataFrame({
            'layer': high_kl['layer'],
            'neuron': high_kl['neuron_idx'],
            'kl_impact': high_kl['kl_patched'],
            'activation_count': [
                neuron_df[(neuron_df['layer'] == l) & 
                         (neuron_df['neuron_idx'] == n)]['count'].iloc[0]
                for l, n in zip(high_kl['layer'], high_kl['neuron_idx'])
            ]
        })
        
        print("\n🔍 High-Impact Neurons Analysis:")
        print("Top 10 neurons by KL divergence impact:")
        print(impact_summary.to_string(index=False))
        print("\nInterpretation:")
        print("- Higher KL = stronger influence on model outputs")
        print("- Higher activation count = more frequently involved")
        
        impact_summary.to_csv(output_dir/'high_impact_neurons.csv', index=False)

def create_visualizations(patch_dir: Path, output_dir: Path):
    """Create and save visualizations from analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    prompt_df = pd.read_csv(patch_dir/"prompt_results.csv")
    layer_df = pd.read_csv(patch_dir/"layer_stats.csv")
    neuron_df = pd.read_csv(patch_dir/"neuron_stats.csv")
    
    # 1. Layer Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=layer_df, x='layer', y='count')
    plt.title('Distribution of Star Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Count')
    plt.savefig(output_dir/'layer_distribution.png')
    plt.close()
    
    # 2. Top Neurons Heatmap
    pivot_neurons = pd.pivot_table(
        neuron_df, 
        values='count',
        index='layer',
        columns='neuron_idx',
        fill_value=0
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_neurons, cmap='YlOrRd')
    plt.title('Neuron Activation Heatmap')
    plt.xlabel('Neuron Index')
    plt.ylabel('Layer')
    plt.savefig(output_dir/'neuron_heatmap.png')
    plt.close()
    
    # 3. KL Divergence Distribution
    if 'kl_patched' in prompt_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=prompt_df, x='kl_patched', bins=30)
        plt.title('Distribution of KL Divergence')
        plt.xlabel('KL Divergence')
        plt.ylabel('Count')
        plt.savefig(output_dir/'kl_distribution.png')
        plt.close()
    
    # 4. Pair Gap Analysis
    gap_files = list(patch_dir.glob("pairgap_*.csv"))
    if gap_files:
        for gf in gap_files:
            df_gap = pd.read_csv(gf)
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df_gap['kl_gap'])
            plt.title(f'KL Gap Distribution - {gf.stem}')
            plt.ylabel('KL Gap')
            plt.savefig(output_dir/f'{gf.stem}_boxplot.png')
            plt.close()

    # Add text visualization
    print("\n=== Text-based Visualization ===")
    text_visualize(patch_dir)

    # Add KL divergence analysis
    analyze_kl_divergence(prompt_df, neuron_df, output_dir)

def create_advanced_visualizations(patch_dir: Path, output_dir: Path):
    """Create comprehensive visualizations for bias neurons analysis"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    prompt_df = pd.read_csv(patch_dir/"prompt_results.csv")
    layer_df = pd.read_csv(patch_dir/"layer_stats.csv")
    neuron_df = pd.read_csv(patch_dir/"neuron_stats.csv")
    
    # 1. Type A: Bias Suppression Neurons
    plt.figure(figsize=(12, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Layer distribution
    sns.barplot(data=layer_df, x='layer', y='count', ax=ax1)
    ax1.set_title('Star Layer Distribution\n(Bias Suppression)')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Activation Count')
    
    # KL divergence effect
    if 'kl_patched' in prompt_df.columns and 'kl_ablated' in prompt_df.columns:
        data = pd.melt(prompt_df[['kl_patched', 'kl_ablated']], 
                      value_vars=['kl_patched', 'kl_ablated'],
                      var_name='State', value_name='KL Divergence')
        sns.boxplot(data=data, x='State', y='KL Divergence', ax=ax2)
        ax2.set_title('KL Divergence Before/After\nBias Neuron Patching')
    
    plt.tight_layout()
    plt.savefig(output_dir/'bias_suppression_analysis.png')
    plt.close()
    
    # 2. Type B: Safety-Aligned Neurons
    aln_path = patch_dir/"alignment_neurons.csv"
    if aln_path.exists():
        aln_df = pd.read_csv(aln_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=aln_df, x='layer', y='neuron_idx', s=100)
        plt.title('Safety-Aligned Neurons Distribution')
        plt.xlabel('Layer Index')
        plt.ylabel('Neuron Index')
        plt.savefig(output_dir/'safety_neurons.png')
        plt.close()
    
    # 3. Type C: Intersection Analysis
    inter_path = patch_dir/"intersection.csv"
    if inter_path.exists():
        inter_df = pd.read_csv(inter_path)
        
        plt.figure(figsize=(12, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Venn diagram-like visualization
        total_bias = len(neuron_df.groupby(['layer', 'neuron_idx']))
        total_safety = len(pd.read_csv(aln_path)) if aln_path.exists() else 0
        total_inter = len(inter_df)
        
        from matplotlib_venn import venn2
        venn2(subsets=(total_bias-total_inter, total_safety-total_inter, total_inter),
              set_labels=('Bias Suppression', 'Safety Aligned'),
              ax=ax1)
        ax1.set_title('Overlap between Bias and Safety Neurons')
        
        # Distribution of dual-purpose neurons
        sns.scatterplot(data=inter_df, x='layer', y='neuron_idx', 
                       s=100, color='purple', ax=ax2)
        ax2.set_title('Distribution of Dual-Purpose Neurons')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Neuron Index')
        
        plt.tight_layout()
        plt.savefig(output_dir/'dual_purpose_analysis.png')
        plt.close()
    
    # 4. Pair Gap Analysis
    gap_files = list(patch_dir.glob("pairgap_*.csv"))
    if gap_files:
        plt.figure(figsize=(12, 6))
        for gf in gap_files:
            df_gap = pd.read_csv(gf)
            sns.kdeplot(data=df_gap, x='kl_gap', label=gf.stem)
        plt.title('KL Gap Distribution\n(Positive = Better Bias Suppression)')
        plt.xlabel('KL Divergence Gap')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(output_dir/'kl_gap_distribution.png')
        plt.close()

def create_neuron_type_analysis(patch_dir: Path, output_dir: Path):
    """Create visualizations specifically for different neuron types"""
    
    # Load all relevant CSVs
    neuron_df = pd.read_csv(patch_dir/"neuron_stats.csv")  # Type A: Bias suppression
    aln_path = patch_dir/"alignment_neurons.csv"  # Type B: Safety-aligned
    inter_path = patch_dir/"intersection.csv"  # Type C: Dual-purpose
    
    # Create comprehensive neuron type visualization
    plt.figure(figsize=(15, 10))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Type A: Bias Suppression Neurons (from neuron_stats.csv)
    sns.scatterplot(
        data=neuron_df,
        x='layer',
        y='neuron_idx',
        size='count',
        sizes=(50, 400),
        alpha=0.6,
        color='blue',
        label='Bias Suppression',
        ax=ax1
    )
    ax1.set_title('Type A: Bias Suppression Neurons\nSize indicates activation frequency')
    
    # 2. Type B: Safety-Aligned Neurons
    if aln_path.exists():
        aln_df = pd.read_csv(aln_path)
        sns.scatterplot(
            data=aln_df,
            x='layer',
            y='neuron_idx',
            color='green',
            label='Safety Aligned',
            ax=ax2
        )
    ax2.set_title('Type B: Safety-Aligned Neurons')
    
    # 3. Type C: Dual-Purpose Neurons
    if inter_path.exists():
        inter_df = pd.read_csv(inter_path)
        sns.scatterplot(
            data=inter_df,
            x='layer',
            y='neuron_idx',
            color='purple',
            label='Dual Purpose',
            ax=ax3
        )
    ax3.set_title('Type C: Dual-Purpose Neurons\n(Intersection of A & B)')
    
    # 4. Combined visualization
    sns.scatterplot(data=neuron_df, x='layer', y='neuron_idx', 
                   alpha=0.3, color='blue', label='Bias Suppression', ax=ax4)
    if aln_path.exists():
        sns.scatterplot(data=aln_df, x='layer', y='neuron_idx',
                       alpha=0.3, color='green', label='Safety', ax=ax4)
    if inter_path.exists():
        sns.scatterplot(data=inter_df, x='layer', y='neuron_idx',
                       color='purple', label='Dual Purpose', ax=ax4)
    ax4.set_title('Combined Neuron Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir/'neuron_types_analysis.png')
    plt.close()

    # Print CSV contents explanation
    print("\n📁 CSV Contents Guide:")
    print("\nType A - Bias Suppression Neurons:")
    print("  • neuron_stats.csv: Contains neurons that reduce bias when patched")
    print("    - layer: Layer index")
    print("    - neuron_idx: Neuron index within layer")
    print("    - count: Number of times neuron was activated")
    
    print("\nType B - Safety-Aligned Neurons:")
    print("  • alignment_neurons.csv: Contains neurons specifically activated for safety")
    print("    - layer: Layer index")
    print("    - neuron_idx: Neuron index")
    
    print("\nType C - Dual-Purpose Neurons:")
    print("  • intersection.csv: Neurons that appear in both A and B")
    print("    - layer: Layer index")
    print("    - neuron_idx: Neuron index")
    
    print("\nDetailed Analysis:")
    print("  • prompt_results.csv: Per-prompt patching results")
    print("    - kl_patched: KL divergence after patching")
    print("    - kl_ablated: KL divergence after neuron ablation")
    print("  • layer_stats.csv: Distribution of star layers")
    
def load_and_report(patch_dir="patch_out"):
    # Capture console output
    old_stdout = sys.stdout
    output_buffer = StringIO()
    sys.stdout = output_buffer
    
    try:
        patch_dir = Path(patch_dir)
        # ── A: バイアス抑制（Bias-patch scan）結果 ──────────────
        print("── A: Bias-patch scan results ──────────────────")
        prompt_df = pd.read_csv(patch_dir/"prompt_results.csv")
        layer_df  = pd.read_csv(patch_dir/"layer_stats.csv")
        neuron_df = pd.read_csv(patch_dir/"neuron_stats.csv")

        print(f"• processed prompts: {len(prompt_df)}")
        print("★ Top layers (star counts):")
        print(layer_df.head(5).to_string(index=False))
        print("★ Top neurons (hit counts):")
        print(neuron_df.head(5).to_string(index=False))

        # ── B: 安全アライン専用ニューロン（Alignment-specific scan）結果 ────
        aln_path = patch_dir/"alignment_neurons.csv"
        if aln_path.exists():
            print("\n── B: Alignment-specific neurons ───────────────")
            aln_df = pd.read_csv(aln_path)
            print(f"total safe‐only neurons: {len(aln_df)}")
            print(aln_df.head(5).to_string(index=False))
        else:
            print("\n── B: Alignment scan was skipped or no results.")

        # ── C: 交差集合 & ペアギャップ解析 ─────────────────
        inter_path = patch_dir/"intersection.csv"
        if inter_path.exists():
            print("\n── C1: Intersection (bias ∩ alignment) ─────────")
            inter_df = pd.read_csv(inter_path)
            print(f"共通ニューロン数: {len(inter_df)}")
            print(inter_df.to_string(index=False))
        else:
            print("\n── C1: No intersection.csv (zero overlap)")

        # ペアギャップ
        gap_files = list(patch_dir.glob("pairgap_*.csv"))
        if gap_files:
            print("\n── C2: Pair‐gap analysis ───────────────────────")
            for gf in gap_files:
                df_gap = pd.read_csv(gf, index_col=0)
                print(f"\n● {gf.name}: mean ΔKL = {df_gap['kl_gap'].mean():+.4f}")
                print(df_gap["kl_gap"].sort_values(ascending=False).head().to_string())
        else:
            print("\n── C2: No pairgap_*.csv (no matched pairs)")

        # Add visualization
        vis_dir = Path(patch_dir)/"visualizations"
        create_visualizations(Path(patch_dir), vis_dir)
        print(f"\n📊 Visualizations saved to {vis_dir}")
        
        # Create detailed analysis text file
        print("\n=== Detailed Analysis Summary ===")
        print("\n1. Visualization Types:")
        print("  • Type A (Bias Suppression): Shows neurons that help reduce bias")
        print("  • Type B (Safety-Aligned): Shows neurons specific to safety/alignment")
        print("  • Type C (Dual-Purpose): Shows neurons that serve both functions")
        print("  • Combined View: Shows overlap and distribution of all types")
        
        print("\n2. CSV Files Contents:")
        print("  • neuron_stats.csv: Type A neurons - bias suppression data")
        print("  • alignment_neurons.csv: Type B neurons - safety alignment data")
        print("  • intersection.csv: Type C neurons - dual-purpose neurons")
        print("  • prompt_results.csv: Detailed results per prompt")
        print("  • layer_stats.csv: Statistics per layer")
        
        # Save console output to text file
        output_text = output_buffer.getvalue()
        with open(patch_dir/"analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        
    finally:
        # Restore stdout
        sys.stdout = old_stdout
        print(output_text)  # Show in console
        print(f"\n📝 Analysis report saved to {patch_dir/'analysis_report.txt'}")

def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize patch-out results")
    p.add_argument("--path", type=Path, default="patch_out",
                   help="Path to directory containing analysis results")
    return p.parse_args()

def main():
    args = get_cli()
    load_and_report(args.path)

if __name__ == "__main__":
    main()
