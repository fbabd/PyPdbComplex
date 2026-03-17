"""
Tabular Feature Extraction and Publication-Quality Visualizations

This module extracts per-residue features from analysis results into tabular format
and provides functions to generate publication-ready plots for comparing protein complexes.
"""
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union 
from pathlib import Path
import warnings


def _set_plot_style() -> None:
    """Apply publication-quality matplotlib/seaborn defaults for this module's plots."""
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    sns.set_style("ticks")




def extract_residue_features(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extract all per-residue features into a tabular format (DataFrame).
    
    This creates a comprehensive feature table with one row per residue per variant,
    suitable for statistical analysis, machine learning, or export to CSV/Excel.
    
    Args:
        results: Dictionary with variant names as keys, each containing:
            - 'residues': Dict of residue data with all properties
    
    Returns:
        pandas DataFrame with columns:
            - variant: Variant name (e.g., 'WT', 'Variant1')
            - residue_id: Full residue identifier (e.g., 'SER A:32')
            - residue_type: Three-letter amino acid code
            - chain: Chain identifier
            - position: Residue position number
            - sasa_unbound: Unbound SASA (Å²)
            - sasa_bound: Bound SASA (Å²)
            - sasa_delta: Change in SASA upon binding (Å²)
            - sasa_buried_fraction: Fraction buried (0-1)
            - is_interface: Boolean, is at interface
            - interface_partners: List of partner chains
            - interface_contact_count: Number of interface contacts
            - vdw_energy: Van der Waals energy (kcal/mol)
            - hydrogen_bonds: Number of hydrogen bonds
            - salt_bridges: Number of salt bridges
            - disulfide_bonds: Number of disulfide bonds
            - hydrophobic_contacts: Number of hydrophobic contacts
            - pi_stacking: Number of pi-stacking interactions
            - total_contacts: Total number of contacts
    
    Example:
        >>> results = analyzer.analyze()
        >>> df = extract_residue_features(results)
        >>> df.to_csv('residue_features.csv', index=False)
        >>> df.to_excel('residue_features.xlsx', index=False)
    """
    
    records = []
    
    for variant, data in results.items():
        for res_id, res_data in data['residues'].items():
            # Parse residue ID: "SER A:32"
            parts = res_id.split()
            if len(parts) != 2:
                continue
            
            res_type = parts[0]
            chain_pos = parts[1]
            chain, position = chain_pos.split(':')
            
            # Extract SASA data
            sasa = res_data.get('sasa', {})
            
            # Extract interface data
            interface = res_data.get('interface', {})
            if isinstance(interface, dict):
                is_interface = True
                interface_partners = interface.get('partners', [])
                interface_contact_count = interface.get('contact_count', 0)
            else:
                is_interface = False
                interface_partners = []
                interface_contact_count = 0
            
            # Extract VdW energy
            vdw_energy = res_data.get('vdw_energy', None)
            
            # Extract bond counts
            bonds = res_data.get('bonds', {})
            
            # Build record
            record = {
                'variant': variant,
                'residue_id': res_id,
                'residue_type': res_type,
                'chain': chain,
                'position': position,
                'position_numeric': int(''.join(filter(str.isdigit, position))),
                
                # SASA features
                'sasa_unbound': sasa.get('unbound', None),
                'sasa_bound': sasa.get('bound', None),
                'sasa_delta': sasa.get('delta', None),
                'sasa_buried_fraction': sasa.get('buried_fraction', None),
                
                # Interface features
                'is_interface': is_interface,
                'interface_partners': ','.join(interface_partners) if interface_partners else None,
                'interface_contact_count': interface_contact_count,
                
                # Energy features
                'vdw_energy': vdw_energy,
                
                # Bond/interaction features
                'hydrogen_bonds': bonds.get('hydrogen_bond', 0),
                'salt_bridges': bonds.get('salt_bridge', 0),
                'disulfide_bonds': bonds.get('disulfide', 0),
                'hydrophobic_contacts': bonds.get('hydrophobic', 0),
                'pi_stacking': bonds.get('pi_stacking', 0),
                'total_contacts': sum([
                    bonds.get('hydrogen_bond', 0),
                    bonds.get('salt_bridge', 0),
                    bonds.get('disulfide', 0),
                    bonds.get('hydrophobic', 0),
                    bonds.get('pi_stacking', 0)
                ])
            }
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by variant and position
    df = df.sort_values(['variant', 'chain', 'position_numeric'])
    df = df.reset_index(drop=True)
    
    return df


def compare_interface_residues(df: pd.DataFrame, 
                               wt_name: str = 'WT',
                               output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create a comparison table focusing on interface residues across variants.
    
    Args:
        df: DataFrame from extract_residue_features()
        wt_name: Name of the wild-type variant
        output_file: Optional path to save CSV
    
    Returns:
        DataFrame with interface residue comparison
    """
    
    # Filter to interface residues only
    interface_df = df[df['is_interface'] == True].copy()
    
    # Pivot to compare variants
    comparison = interface_df.pivot_table(
        index=['chain', 'position', 'residue_type'],
        columns='variant',
        values=['sasa_buried_fraction', 'vdw_energy', 'total_contacts'],
        aggfunc='first'
    )
    
    comparison = comparison.reset_index()
    
    if output_file:
        comparison.to_csv(output_file, index=False)
    
    return comparison

# ============================================================================
# STATISTICAL ANALYSIS HELPERS
# ============================================================================

def calculate_interface_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for interface residues by variant.
    
    Returns DataFrame with statistics like:
    - Number of interface residues
    - Average burial fraction
    - Average VdW energy
    - Total contacts
    """
    
    interface_df = df[df['is_interface'] == True]
    
    stats = interface_df.groupby('variant').agg({
        'residue_id': 'count',
        'sasa_buried_fraction': ['mean', 'std'],
        'vdw_energy': ['mean', 'std', 'sum'],
        'total_contacts': 'sum',
        'hydrogen_bonds': 'sum',
        'salt_bridges': 'sum',
        'hydrophobic_contacts': 'sum'
    })
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    
    stats.rename(columns={'residue_id_count': 'num_interface_residues'}, inplace=True)
    
    return stats


# ============================================================================
# PUBLICATION-QUALITY VISUALIZATION FUNCTIONS
# ============================================================================

def plot_interface_burial_comparison(
    df: pd.DataFrame, 
    output_file: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 6),
    split_by_chain: bool = False,           
    top_k: int = 20               
) -> plt.Figure:
    """
    Creates a grouped bar chart of interface burial.
    """
    _set_plot_style()

    # -----------------------
    # Filter to interface residues
    # -----------------------
    interface_df = df[df['is_interface'] == True].copy()
    
    interface_df['position_label'] = (
        interface_df['residue_type'] + ' ' + 
        interface_df['chain'] + ':' + 
        interface_df['position']
    )

    # ---------------------------------------------------------
    # CASE 1: No splitting (top-k in all chains)
    # ---------------------------------------------------------
    if not split_by_chain:
        avg_burial = interface_df.groupby('position_label')['sasa_buried_fraction'].mean()
        sorted_positions = avg_burial.sort_values(ascending=False).head(top_k).index
        plot_df = interface_df[interface_df['position_label'].isin(sorted_positions)]
        
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]  # treat as list for unified loop
        
        chain_groups = {"ALL": plot_df}

    # ---------------------------------------------------------
    # CASE 2: Split by chain + top-k per chain
    # ---------------------------------------------------------
    else:
        chains = sorted(interface_df['chain'].unique())
        n_chains = len(chains)

        # One subplot per chain, stacked vertically
        fig, axes = plt.subplots(
            nrows=n_chains,
            ncols=1,
            figsize=(figsize[0], figsize[1] * n_chains),
            squeeze=False
        )
        axes = axes.flatten()

        chain_groups = {}
        for chain in chains:
            sub = interface_df[interface_df['chain'] == chain].copy()

            avg_burial = sub.groupby('position_label')['sasa_buried_fraction'].mean()
            sorted_positions = avg_burial.sort_values(ascending=False).head(top_k).index

            chain_groups[chain] = sub[sub['position_label'].isin(sorted_positions)]

    # -----------------------
    # Plot each chain group
    # -----------------------
    scale = figsize[0] * 1.2    # or figsize[1], or sqrt(w*h)
    fs_small  = 6  * scale / 10
    fs_medium = 10 * scale / 10
    fs_large  = 14 * scale / 10
    global_variants = interface_df['variant'].unique()
    color_map = dict(zip(global_variants, sns.color_palette("Set2", len(global_variants))))
    bar_width = 0.8 / len(global_variants)

    for ax, (chain_name, plot_df) in zip(axes, chain_groups.items()):

        positions = plot_df['position_label'].unique()
        x = np.arange(len(positions))

        for j, variant in enumerate(global_variants):
            variant_data = plot_df[plot_df['variant'] == variant]

            values = [
                variant_data.loc[variant_data['position_label'] == pos,
                                'sasa_buried_fraction'].values[0]
                if pos in variant_data['position_label'].values else 0
                for pos in positions
            ]

            ax.bar(
                x + j * bar_width, values, bar_width,
                label=variant,
                color=color_map[variant],
                edgecolor="black",
                linewidth=0.5,
            )

        # Labels per subplot
        # ax.set_xlabel(f'{chain_name} Positions', fontsize=fs_large, fontweight='bold')
        ax.set_ylabel('Buried Fraction', fontsize=fs_medium, fontweight='bold')
        ax.set_title(f'Chain {chain_name}: Top {top_k}',
                    fontsize=fs_large, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(global_variants) - 1) / 2)
        ax.set_xticklabels(positions, fontsize=fs_medium, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs_medium) 
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)

    # -----------------------
    # Global horizontal legend at the bottom
    # -----------------------
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                    facecolor=color_map[v],
                    edgecolor='black',
                    linewidth=0.5)
        for v in global_variants
    ]
    fig.legend(
        handles,
        global_variants,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(global_variants),
        fontsize=fs_medium,
        frameon=True,
    )
    # Leave room at bottom for legend
    plt.tight_layout(rect=(0, 0.08, 1, 1))

    # -----------------------
    # Save
    # -----------------------
    if output_file:
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    return fig


def plot_vdw_energy_heatmap(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    split_by_chain: bool = False,
    top_k: int = 10
) -> plt.Figure:
    """
    Heatmap of VDW energies across variants, supporting:
      - split_by_chain (one heatmap per chain)
      - top_k_per_chain (selects strongest interacting residues)
      - relative font sizes (scaled to figsize)
    """
    _set_plot_style()

    # ------------------------------
    # Filter to valid interface residues
    # ------------------------------
    interface_df = df[
        (df["is_interface"] == True) &
        (df["vdw_energy"].notna())
    ].copy()

    # Create residue labels
    interface_df["position_label"] = (
        interface_df["residue_type"] + " " +
        interface_df["chain"] + ":" +
        interface_df["position"]
    )

    # ------------------------------
    # Compute font scaling
    # ------------------------------
    scale = figsize[0] * 2 
    fs_small  = scale * 0.55
    fs_medium = scale * 0.75
    fs_large  = scale * 1.0

    # ------------------------------
    # GLOBAL variant order
    # ------------------------------
    variants = sorted(interface_df["variant"].unique())

    # ------------------------------
    # CASE A: Single heatmap (all chains)
    # ------------------------------
    if not split_by_chain:
        pivot_df = interface_df.pivot_table(
            index="position_label",
            columns="variant",
            values="vdw_energy",
            aggfunc="mean"
        )

        pivot_df["mean"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("mean").drop("mean", axis=1)
        pivot_df = pivot_df.head(top_k)

        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        chain_groups = {"ALL": pivot_df}

    # ------------------------------
    # CASE B: Chain-by-chain heatmaps
    # ------------------------------
    else:
        chains = sorted(interface_df["chain"].unique())
        n_chains = len(chains)

        fig, axes = plt.subplots(
            nrows=n_chains,
            ncols=1,
            figsize=(figsize[0], figsize[1] * n_chains),
            squeeze=False
        )
        axes = axes.flatten()

        chain_groups = {}
        for chain in chains:
            sub = interface_df[interface_df["chain"] == chain]

            pivot_df = sub.pivot_table(
                index="position_label",
                columns="variant",
                values="vdw_energy",
                aggfunc="mean"
            )

            pivot_df["mean"] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values("mean").drop("mean", axis=1)
            pivot_df = pivot_df.head(top_k)

            chain_groups[chain] = pivot_df

    # ------------------------------
    # PLOTTING EACH HEATMAP
    # ------------------------------
    for ax, (chain_name, pivot_df) in zip(axes, chain_groups.items()):

        sns.heatmap(
            pivot_df,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "VdW Energy (kcal/mol)", "shrink": 0.9},
            annot=True,
            fmt=".1f",
            annot_kws={"size": fs_small}
        )

        # Titles + labels
        ax.set_title(
            f"Chain {chain_name}: Top {top_k}",
            fontsize=fs_large, fontweight="bold", pad=16
        )
        ax.set_xlabel("Variant", fontsize=fs_medium, fontweight="bold")
        ax.set_ylabel("Residue", fontsize=fs_medium, fontweight="bold")

        # Tick label sizes
        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=fs_small,
            rotation=45,
            ha="right"
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=fs_small
        )

    plt.tight_layout()

    # ------------------------------
    # Save
    # ------------------------------
    if output_file:
        plt.savefig(output_file, dpi=600, bbox_inches="tight")

    return fig


def plot_contact_network_comparison(df: pd.DataFrame,
                                    output_file: Optional[str] = None,
                                    figsize: Tuple[float, float] = (12, 7)) -> plt.Figure:
    """
    **Figure 3: Interaction Type Distribution**
    
    Stacked bar chart showing distribution of different interaction types
    (H-bonds, salt bridges, hydrophobic, etc.) for each variant.
    
    Recommended for: Understanding interaction profile changes
    
    Args:
        df: DataFrame from extract_residue_features()
        output_file: Path to save figure
        figsize: Figure size in inches
    
    Returns:
        matplotlib Figure object
    """
    _set_plot_style()

    # Sum interactions by variant
    interaction_cols = ['hydrogen_bonds', 'salt_bridges', 'disulfide_bonds',
                       'hydrophobic_contacts', 'pi_stacking']
    
    interaction_sums = df.groupby('variant')[interaction_cols].sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # ------------------------------
    # Compute font scaling
    # ------------------------------
    scale = figsize[0] * 2 
    fs_small  = scale * 0.55
    fs_medium = scale * 0.75
    fs_large  = scale * 1.0
    
    interaction_sums.plot(kind='bar', stacked=True, ax=ax,
                         color=sns.color_palette("Set3", len(interaction_cols)),
                         edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Variant', fontsize=fs_medium, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=fs_medium, fontweight='bold')
    ax.set_title('Interaction Type Distribution Across Variants',
                 fontsize=fs_large, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs_small, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs_small, ha='right')
    ax.legend_.remove() 
    for container in ax.containers:
        # container contains the rectangles for one interaction type
        for bar in container:
            height = bar.get_height()
            if height > 0:
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_y() + height / 2   # middle of each stacked block

                ax.text(
                    x, y,
                    f"{int(height)}",
                    ha='center', va='center',
                    fontsize=fs_small,
                    color='black'
                )
    sns.despine()
    
    # -----------------------
    # Global horizontal legend at the bottom
    # -----------------------
    colors = sns.color_palette("Set3", len(interaction_cols))

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                    facecolor=colors[i],
                    edgecolor='black',
                    linewidth=0.5)
        for i in range(len(interaction_cols))
    ]

    fig.legend(
        handles,
        interaction_cols,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(interaction_cols),
        fontsize=fs_small,
        frameon=True
    )

    plt.tight_layout(rect=(0, 0.10, 1, 1))  # leave room for legend
    
    if output_file:
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    return fig


def plot_mutation_impact_scatter(df: pd.DataFrame,
                                 wt_name: str = 'WT',
                                 output_file: Optional[str] = None,
                                 figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
    """
    **Figure 4: Mutation Impact Analysis (Scatter Plot)**
    
    Scatter plot showing Î”VdW energy vs Î”SASA burial for mutations.
    Helps identify mutations that alter both binding energy and surface burial.
    
    Recommended for: Analyzing how specific mutations affect binding
    
    Args:
        df: DataFrame from extract_residue_features()
        wt_name: Name of wild-type variant
        output_file: Path to save figure
        figsize: Figure size in inches
    
    Returns:
        matplotlib Figure object
    """
    _set_plot_style()

    # Get WT data
    wt_df = df[df['variant'] == wt_name].copy()
    wt_df = wt_df.set_index(['chain', 'position'])
    
    # Compare each variant to WT
    fig, axes = plt.subplots(1, len(df['variant'].unique()) - 1, 
                            figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    variant_idx = 0
    for variant in df['variant'].unique():
        if variant == wt_name:
            continue
        
        var_df = df[df['variant'] == variant].copy()
        var_df = var_df.set_index(['chain', 'position'])
        
        # Find common residues
        common_idx = wt_df.index.intersection(var_df.index)
        
        delta_burial = (var_df.loc[common_idx, 'sasa_buried_fraction'] - 
                       wt_df.loc[common_idx, 'sasa_buried_fraction'])
        delta_vdw = (var_df.loc[common_idx, 'vdw_energy'] - 
                    wt_df.loc[common_idx, 'vdw_energy'])
        
        # Filter out NaNs
        mask = delta_burial.notna() & delta_vdw.notna()
        delta_burial = delta_burial[mask]
        delta_vdw = delta_vdw[mask]
        
        ax = axes[variant_idx]
        
        # Color by mutation vs identical
        wt_types = wt_df.loc[common_idx[mask], 'residue_type']
        var_types = var_df.loc[common_idx[mask], 'residue_type']
        is_mutation = wt_types != var_types
        
        ax.scatter(delta_burial[~is_mutation], delta_vdw[~is_mutation],
                  alpha=0.5, s=50, c='gray', label='Identical')
        ax.scatter(delta_burial[is_mutation], delta_vdw[is_mutation],
                  alpha=0.7, s=80, c='red', edgecolor='black', 
                  linewidth=0.5, label='Mutated')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        
        ax.set_xlabel('Δ Burial Fraction', fontsize=10, fontweight='bold')
        ax.set_ylabel('Δ VdW Energy (kcal/mol)', fontsize=10, fontweight='bold')
        ax.set_title(f'{variant} vs {wt_name}', fontsize=11, fontweight='bold')
        ax.legend(frameon=True, fontsize=14)
        ax.grid(True, alpha=0.2)
        
        variant_idx += 1
    
    plt.suptitle('Mutation Impact on Binding Properties', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
    
    return fig


def plot_residue_property_profiles_combined(df: pd.DataFrame,
                                            chains: Optional[List[str]] = None,
                                            output_file: Optional[str] = None,
                                            figsize: Tuple[float, float] = (18, 12)) -> plt.Figure:
    """
    Combined Residue Property Profiles for All Chains
    
    Creates a single figure with all chains concatenated along the x-axis.
    Different variants are shown in different colors. Vertical lines mark chain boundaries.
    
    Args:
        df: DataFrame with residue features
        chains: List of chains to plot (None = all)
        output_file: Path to save figure
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    _set_plot_style()

    # Font and marker sizes scaled to figure
    scale = figsize[0] * 1.5
    fs_small = scale * 0.55
    fs_medium = scale * 0.75
    fs_large = scale * 1.00
    ms_small = scale * 0.1
    lw = scale * 0.05
    
    # Determine chains to plot
    if chains is None:
        chains = sorted(df['chain'].unique())
    elif isinstance(chains, str):
        chains = [chains]
    
    # Properties to plot
    properties = [
        ('sasa_buried_fraction', 'Buried Fraction\n', (0, 1)),
        ('vdw_energy', 'VdW Energy\n(kcal/mol)', None),
        ('total_contacts', '# Total Bonds\n', (0, None)),
        ('interface_contact_count', '# Interface\nContacts', (0, None))
    ]
    
    n_properties = len(properties)
    
    # Get variants and colors (one color per variant)
    variants = sorted(df['variant'].unique())
    variant_colors = sns.color_palette("Set2", len(variants))
    variant_color_map = dict(zip(variants, variant_colors))
    
    # Create figure
    fig, axes = plt.subplots(n_properties, 1, figsize=figsize, sharex=True)
    
    if n_properties == 1:
        axes = [axes]
    
    # Create concatenated x-axis positions for all chains
    # We'll offset each chain so they appear one after another
    chain_offsets = {}
    current_offset = 0
    chain_boundaries = []
    
    for chain in chains:
        chain_df = df[df['chain'] == chain]
        if not chain_df.empty:
            chain_offsets[chain] = current_offset
            max_pos = chain_df['position_numeric'].max()
            current_offset += max_pos + 10  # Add gap between chains
            chain_boundaries.append(current_offset - 5)  # Boundary in the middle of gap
    
    # Remove last boundary (no boundary after last chain)
    if chain_boundaries:
        chain_boundaries = chain_boundaries[:-1]
    
    # Plot each property
    for prop_idx, (prop, ylabel, ylim) in enumerate(properties):
        ax = axes[prop_idx]
        
        has_data = False
        
        # For each variant (each gets its own color)
        for variant in variants:
            variant_x = []
            variant_y = []
            
            # Concatenate data from all chains for this variant
            for chain in chains:
                chain_df = df[(df['chain'] == chain) & (df['variant'] == variant)].copy()
                
                if chain_df.empty:
                    continue
                
                chain_df = chain_df.sort_values('position_numeric')
                
                # Check if this property has data
                if chain_df[prop].isna().all():
                    continue
                
                has_data = True
                
                # Offset positions for this chain
                offset = chain_offsets[chain]
                x_positions = chain_df['position_numeric'].values + offset
                y_values = chain_df[prop].values
                
                variant_x.extend(x_positions)
                variant_y.extend(y_values)
            
            # Plot this variant's complete line (all chains concatenated)
            if variant_x:
                ax.plot(
                    variant_x, 
                    variant_y,
                    marker='o',
                    markersize=ms_small,
                    linewidth=lw,
                    label=variant,
                    color=variant_color_map[variant],
                    alpha=0.8
                )
        
        if not has_data:
            ax.text(0.5, 0.5, f'No data for {ylabel}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=fs_medium, color='gray')
        
        # Add vertical lines at chain boundaries
        for boundary in chain_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', 
                      alpha=0.5, linewidth=1.5)
        
        # Styling
        ax.set_ylabel(ylabel, fontsize=fs_medium, fontweight='bold')
        if ylim:
            if ylim[1] is None:
                ax.set_ylim(bottom=ylim[0])
            else:
                ax.set_ylim(ylim)
        
        ax.tick_params(axis='y', labelsize=fs_small)
        ax.tick_params(axis='x', labelsize=fs_small)
        ax.grid(True, alpha=0.2, linestyle='--')
        sns.despine(ax=ax)
    
    # X-axis label with chain annotations
    axes[-1].set_xlabel('Residues', 
                        fontsize=fs_large, fontweight='bold')
    axes[-1].set_xticklabels([])  # Hide tick labels but keep ticks

    
    # Add chain labels on x-axis
    chain_centers = []
    chain_labels = []
    for chain in chains:
        if chain in chain_offsets:
            chain_df = df[df['chain'] == chain]
            if not chain_df.empty:
                offset = chain_offsets[chain]
                max_pos = chain_df['position_numeric'].max()
                center = offset + max_pos / 2
                chain_centers.append(center)
                chain_labels.append(f'Chain {chain}')
    
    # Add secondary x-axis at TOP (on first subplot)
    if chain_centers:
        ax2 = axes[0].twiny()
        ax2.set_xlim(axes[0].get_xlim())
        ax2.set_xticks(chain_centers)
        ax2.set_xticklabels(chain_labels, fontsize=fs_medium, fontweight='bold')
        ax2.tick_params(axis='x', length=0)  # Hide tick mar
    
    # Title
    plt.suptitle('Residue Property Profiles - All Chains',
                 fontsize=fs_large * 1.1, fontweight='bold', y=0.995)
    
    # Legend at bottom
    handles = [
        plt.Line2D([0], [0],
                  color=variant_color_map[v],
                  marker='o',
                  markersize=ms_small,
                  linewidth=lw,
                  linestyle='-')
        for v in variants
    ]
    
    fig.legend(
        handles,
        variants,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(variants),
        fontsize=fs_medium,
        frameon=True
    )
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0.08, 1, 0.98))
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=600, bbox_inches='tight')
        print(f"Saved combined profile to {output_file}")
    
    return fig


def plot_residue_property_profiles(df: pd.DataFrame,
                                   chains: Optional[List[str]] = None,
                                   output_file: Optional[str] = None,
                                   figsize: Tuple[float, float] = (16, 10)) -> Dict[str, plt.Figure]:
    """
    Figure 5: Residue Property Profiles Along Chain(s)
    """
    _set_plot_style()

    # -----------------------------
    # Relative fontsize & markersize
    # -----------------------------
    scale = figsize[0]*1.5    # as requested

    fs_small  = scale * 0.55
    fs_medium = scale * 0.75
    fs_large  = scale * 1.00

    ms_small  = scale * 0.25   # good default for line markers
    lw        = scale * 0.10   # line width relative

    # -----------------------------
    # Determine chains to plot
    # -----------------------------
    if chains is None:
        chains = sorted(df['chain'].unique())
    elif isinstance(chains, str):
        chains = [chains]

    # Properties to plot
    properties = [
        ('sasa_buried_fraction', 'Buried Fraction\n', (0, 1)),
        ('vdw_energy', 'VdW Energy \n(kcal/mol)', None),
        ('total_contacts', '# Total Bonds\n ', (0, None)),
        ('interface_contact_count', '# Contacts with\n Epitope/Paratope ', (0, None))
    ]
    
    n_properties = len(properties)
    
    # Get color palette for variants
    variants = sorted(df['variant'].unique())
    colors = sns.color_palette("Set2", len(variants))
    variant_colors = dict(zip(variants, colors))
    
    figures = {}
    
    for chain in chains:
        chain_df = df[df['chain'] == chain].copy()
        
        if chain_df.empty:
            print(f"Warning: No data for Chain {chain}, skipping...")
            continue
        
        fig, axes = plt.subplots(n_properties, 1, figsize=figsize, sharex=True)
        
        if n_properties == 1:
            axes = [axes]
        
        for prop_idx, (prop, ylabel, ylim) in enumerate(properties):
            ax = axes[prop_idx]
            
            has_data = False
            for variant in variants:
                var_data = chain_df[chain_df['variant'] == variant].sort_values('position_numeric')
                
                if var_data.empty or var_data[prop].isna().all():
                    continue
                
                has_data = True
                
                ax.plot(var_data['position_numeric'], var_data[prop],
                        marker='o', 
                        markersize=ms_small,     
                        linewidth=lw,             
                        label=variant,
                        color=variant_colors[variant],
                        alpha=0.8)
            
            if not has_data:
                ax.text(0.5, 0.5, f'No data for {ylabel}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=fs_medium, color='gray')    
            
            ax.set_ylabel(ylabel, fontsize=fs_medium, fontweight='bold')     
            if ylim:
                if ylim[1] is None:
                    ax.set_ylim(bottom=ylim[0])
                else:
                    ax.set_ylim(ylim)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=fs_small) 
            
            ax.grid(True, alpha=0.2, linestyle='--')
            sns.despine(ax=ax)
        
        axes[-1].set_xlabel('Residue Position', 
                            fontsize=fs_large, fontweight='bold') 
        axes[-1].set_xticklabels(ax.get_xticklabels(), fontsize=fs_small)    
        
        plt.suptitle(f'Residue Property Profiles - Chain {chain}',
                     fontsize=fs_large, fontweight='bold', y=0.995)    
        
        # -----------------------
        # Global horizontal legend at the bottom
        # -----------------------
        handles = [
            plt.Line2D([0], [0], 
                      color=variant_colors[v],
                      marker='o',
                      markersize=ms_small,
                      linewidth=lw,
                      linestyle='-')
            for v in variants
        ]
        fig.legend(
            handles,
            variants,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(variants),
            fontsize=fs_medium,
            frameon=True,
        )
        
        # Leave room at bottom for legend
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        
        if output_file:
            from pathlib import Path
            p = Path(output_file)
            chain_output = p.parent / f"{p.stem}_chain_{chain}{p.suffix}"
            plt.savefig(chain_output, dpi=600, bbox_inches='tight')
            print(f"Saved Chain {chain} profile to {chain_output}")
        
        figures[chain] = fig
    
    return figures


def generate_publication_figures(results: Dict[str, Dict],
                                 output_dir: str = 'figures',
                                 wt_name: str = 'WT',
                                 chains: Optional[List[str]] = None,
                                 formats: List[str] = ['png', 'pdf'],
                                 figsize_burial: Tuple[float, float] = (12, 6),
                                 figsize_heatmap: Tuple[float, float] = (10, 8),
                                 figsize_contacts: Tuple[float, float] = (14, 6),
                                 figsize_scatter: Tuple[float, float] = (10, 8),
                                 figsize_profiles: Tuple[float, float] = (14, 10)) -> Dict[str, List[str]]:
    """
    Generate all publication-quality figures in one go.
    
    Args:
        results: Analysis results dictionary
        output_dir: Directory to save figures
        wt_name: Name of wild-type variant (used for mutation impact plot)
        chains: List of chain identifiers to plot. If None, plots all chains found in data
        formats: List of output formats ('png', 'pdf', 'svg')
        figsize_burial: Figure size for burial comparison
        figsize_heatmap: Figure size for VdW heatmap
        figsize_contacts: Figure size for contact distribution
        figsize_scatter: Figure size for mutation impact scatter
        figsize_profiles: Figure size for property profiles
    
    Returns:
        Dictionary mapping figure names to list of file paths
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract features
    print("Extracting residue features...")
    df = extract_residue_features(results)
    
    # Save feature table
    print("Saving feature tables...")
    df.to_csv(output_path / 'residue_features.csv', index=False)
    df.to_excel(output_path / 'residue_features.xlsx', index=False)
    
    figures = {}
    
    # Define each figure with its specific parameters
    figure_specs = [
        {
            'name': 'interface_burial',
            'function': plot_interface_burial_comparison,
            'params': {'figsize': figsize_burial}
        },
        {
            'name': 'vdw_heatmap',
            'function': plot_vdw_energy_heatmap,
            'params': {'figsize': figsize_heatmap}
        },
        {
            'name': 'contact_distribution',
            'function': plot_contact_network_comparison,
            'params': {'figsize': figsize_contacts}
        },
        {
            'name': 'mutation_impact',
            'function': plot_mutation_impact_scatter,
            'params': {'wt_name': wt_name, 'figsize': figsize_scatter}
        },
        {
            'name': 'property_profiles',
            'function': plot_residue_property_profiles,
            'params': {'chains': chains, 'figsize': figsize_profiles}
        },
    ]
    
    # Generate each figure in each format
    for spec in figure_specs:
        fig_name = spec['name']
        fig_func = spec['function']
        fig_params = spec['params']
        
        print(f"Generating {fig_name}...")
        fig_files: List[str] = []
        
        for fmt in formats:
            try:
                if fig_name == 'property_profiles':
                    # Special handling: this function produces multiple files (one per chain)
                    base_output = output_path / f'{fig_name}.{fmt}'
                    chain_figs = fig_func(df, output_file=str(base_output), **fig_params)
                    
                    # Use the keys from the returned dict to determine which chains were actually plotted
                    for chain_id in chain_figs.keys():
                        chain_file = output_path / f'{fig_name}_chain_{chain_id}.{fmt}'
                        fig_files.append(str(chain_file))
                    
                    plt.close('all')
                else:
                    output_file = output_path / f'{fig_name}.{fmt}'
                    fig_func(df, output_file=str(output_file), **fig_params)
                    fig_files.append(str(output_file))
                    plt.close('all')  # Close all figures to free memory

            except Exception as e:
                print(f"Warning: Could not generate {fig_name}.{fmt}: {e}")
        
        figures[fig_name] = fig_files
    
    print(f"\nâœ“ Generated publication figures in: {output_dir}/")
    print(f"âœ“ Feature tables saved: residue_features.csv, residue_features.xlsx")
    print(f"âœ“ Total figures: {len([f for files in figures.values() for f in files])}")
    
    return figures


def generate_publication_figures_for_web(results: Dict[str, Dict],
                                         wt_name: str = 'WT',
                                         chains: Optional[List[str]] = None,
                                         figsize_burial: Tuple[float, float] = (12, 6),
                                         figsize_heatmap: Tuple[float, float] = (10, 8),
                                         figsize_contacts: Tuple[float, float] = (14, 6),
                                         figsize_scatter: Tuple[float, float] = (10, 8),
                                         figsize_profiles: Tuple[float, float] = (14, 10),
                                         dpi: int = 600) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Generate figures and return them as base64-encoded strings for web display.
    
    Returns:
        Dictionary mapping figure names to base64-encoded PNG strings.
        For property_profiles, returns a dict of {chain_id: base64_string}
    """
    
    # Extract features
    df = extract_residue_features(results)
    
    figures_base64 = {}
    
    # Define each figure
    figure_specs = [
        {
            'name': 'interface_burial',
            'function': plot_interface_burial_comparison,
            'params': {'figsize': figsize_burial}
        },
        {
            'name': 'vdw_heatmap',
            'function': plot_vdw_energy_heatmap,
            'params': {'figsize': figsize_heatmap}
        },
        {
            'name': 'contact_distribution',
            'function': plot_contact_network_comparison,
            'params': {'figsize': figsize_contacts}
        },
        {
            'name': 'mutation_impact',
            'function': plot_mutation_impact_scatter,
            'params': {'wt_name': wt_name, 'figsize': figsize_scatter}
        },
        {
            'name': 'property_profiles_combined',
            'function': plot_residue_property_profiles_combined,
            'params': {'chains': chains, 'figsize': figsize_profiles}
        },
    ]
    
    # Generate each figure
    for spec in figure_specs:
        fig_name = spec['name']
        fig_func = spec['function']
        fig_params = spec['params']
        
        try:
            # Call the plotting function - all now return single Figure
            fig = fig_func(df, output_file=None, **fig_params)
            
            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            figures_base64[fig_name] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not generate {fig_name}: {e}")
            figures_base64[fig_name] = None
    
    return figures_base64
