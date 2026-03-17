"""
Multi-Complex Comparison Visualizations

This module provides interactive HTML visualizations for comparing multiple protein
complexes (WT and variants), showing mutations, property changes, and structural differences.

"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import json
from collections import defaultdict


def sort_key(position_key: str) -> Tuple:
    """
    Sort key for position strings like 'A:10', 'A:100', 'B:5'
    Returns tuple of (chain, numeric_position) for proper sorting
    """
    try:
        chain, pos = position_key.split(':')
        # Try to extract numeric part from position (handle insertion codes)
        pos_num = int(''.join(filter(str.isdigit, pos)))
        return (chain, pos_num, pos)
    except:
        return (position_key, 0, '')


def _parse_interface_data(interface_data):
    """Helper to parse interface data consistently"""
    if isinstance(interface_data, dict):
        return {
            'is_interface': True,
            'partners': interface_data.get('partners', []),
            'contact_count': interface_data.get('contact_count', 0)
        }
    else:
        return {
            'is_interface': False,
            'partners': [],
            'contact_count': 0
        }


def _parse_bonds(bonds_data):
    """Helper to parse bonds data, handling empty dicts"""
    if not bonds_data:  # Handle empty dict or None
        return {
            'counts': {
                'hydrogen_bond': 0,
                'salt_bridge': 0,
                'disulfide': 0,
                'hydrophobic': 0,
                'pi_stacking': 0,
                'total': 0
            },
            'details': {}
        }
    
    bond_summary = {}
    bond_types = ["hydrogen_bond",  "salt_bridge", "disulfide", "hydrophobic", "pi_stacking"]
    for bond_type, bond_count in bonds_data.items():
        if bond_type in bond_types:
            bond_summary[bond_type] = bond_count 
    
    bond_summary['total'] = sum(bond_summary.values())
    
    return {
        'counts': bond_summary,
        'details': bonds_data
    }
    


def compare_residues(results, wt_name: str = 'WT'):
    """
    Compare individual residue properties across variants.
    Handles mutations, insertions, and deletions.
    """
    variants = list(results.keys())
    
    # Get all unique residue positions (chain:position) across all variants
    all_positions = set()
    residue_identity_map = {}  # Maps position to residue identities in each variant
    
    for variant in variants:
        for res_id in results[variant]['residues'].keys():
            # Parse residue ID: e.g., "SER A:32" -> residue_type="SER", chain="A", position="32"
            parts = res_id.split()
            if len(parts) == 2:
                res_type = parts[0]
                chain_pos = parts[1]  # "A:32"
                chain, position = chain_pos.split(':')
                
                position_key = f"{chain}:{position}"
                all_positions.add(position_key)
                
                if position_key not in residue_identity_map:
                    residue_identity_map[position_key] = {}
                residue_identity_map[position_key][variant] = {
                    'full_id': res_id,
                    'residue_type': res_type,
                    'chain': chain,
                    'position': position
                }

    residue_comparisons = {}
    
    for position_key in sorted(all_positions, key=sort_key):
        residue_comparisons[position_key] = {
            'position': position_key,
            'variants': {},
            'mutation_status': {},
            'comparison_type': None  # Will be: 'identical', 'mutation', 'insertion', 'deletion'
        } 
        
        # Determine comparison type and mutation status
        present_variants = list(residue_identity_map[position_key].keys())
        residue_types = [residue_identity_map[position_key][v]['residue_type'] 
                        for v in present_variants]
        
        # Check if position exists in all variants
        if len(present_variants) == len(variants):
            # Position exists in all variants
            if len(set(residue_types)) == 1:
                residue_comparisons[position_key]['comparison_type'] = 'identical'
            else:
                residue_comparisons[position_key]['comparison_type'] = 'mutation'
                # Store what mutated to what
                if wt_name in present_variants:
                    wt_type = residue_identity_map[position_key][wt_name]['residue_type']
                    for variant in present_variants:
                        if variant != wt_name:
                            var_type = residue_identity_map[position_key][variant]['residue_type']
                            if wt_type != var_type:
                                residue_comparisons[position_key]['mutation_status'][variant] = {
                                    'from': wt_type,
                                    'to': var_type,
                                    'mutation': f"{wt_type}->{var_type}"
                                }
        else:
            # Position doesn't exist in all variants (insertion/deletion)
            if wt_name in present_variants:
                residue_comparisons[position_key]['comparison_type'] = 'deletion'
                wt_type = residue_identity_map[position_key][wt_name]['residue_type']
                for variant in variants:
                    if variant not in present_variants:
                        residue_comparisons[position_key]['mutation_status'][variant] = {
                            'status': 'deleted',
                            'wt_residue': wt_type
                        }
            else:
                residue_comparisons[position_key]['comparison_type'] = 'insertion'
                for variant in present_variants:
                    var_type = residue_identity_map[position_key][variant]['residue_type']
                    residue_comparisons[position_key]['mutation_status'][variant] = {
                        'status': 'inserted',
                        'residue_type': var_type
                    }
        
        # Store data for each variant
        for variant in variants:
            if variant in residue_identity_map[position_key]:
                res_id = residue_identity_map[position_key][variant]['full_id']
                res_type = residue_identity_map[position_key][variant]['residue_type']
                res_data = results[variant]['residues'][res_id]
    
                residue_comparisons[position_key]['variants'][variant] = {
                    'present': True,
                    'residue_id': res_id,
                    'residue_type': res_type,
                    'sasa': {
                        'buried_fraction': res_data['sasa']['buried_fraction'],
                        'delta': res_data['sasa']['delta'],
                        'bound': res_data['sasa']['bound'],
                        'unbound': res_data['sasa']['unbound']
                    },
                    'interface': _parse_interface_data(res_data['interface']),
                    'vdw_energy': res_data['vdw_energy'],
                    'bonds': _parse_bonds(res_data['bonds'])
                }
            else:
                # Position doesn't exist in this variant
                residue_comparisons[position_key]['variants'][variant] = {
                    'present': False,
                    'residue_id': None,
                    'residue_type': None,
                    'sasa': None,
                    'interface': None,
                    'vdw_energy': None,
                    'bonds': None
                }
    residue_comparisons = dict(sorted(residue_comparisons.items(), key=lambda x: sort_key(x[0]))) 
    return residue_comparisons


def generate_summary_comparison_html(results: Dict[str, Dict], title: str = "Summary Comparison", wt_name: str = 'WT') -> str:
    """
    Generate HTML visualization comparing summary statistics across variants.
    
    Args:
        results: Dictionary with variant names as keys, each containing 'summary' dict
        title: Title for the visualization
        
    Returns:
        HTML string with interactive summary comparison
    """
    variants = list(results.keys())
    
    # Extract summary data
    metrics = ['HYDROGEN_BOND', 'SALT_BRIDGE', 'HYDROPHOBIC', 'n_res_interface', 
               'buried_surface_area', 'total_contacts']
    
    summary_data = {}
    for metric in metrics:
        summary_data[metric] = {v: results[v]['summary'].get(metric, 0) for v in variants}
    
    # Calculate deltas from WT if available
    has_wt = wt_name in variants
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-card h3 {{
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 16px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-values {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .value-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }}
        .value-row .variant-name {{
            font-weight: 600;
            color: #34495e;
        }}
        .value-row .value {{
            font-size: 18px;
            font-weight: bold;
            color: #3498db;
        }}
        .value-row .delta {{
            font-size: 14px;
            margin-left: 10px;
            padding: 2px 8px;
            border-radius: 12px;
        }}
        .delta.positive {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .delta.negative {{
            background: #ffebee;
            color: #c62828;
        }}
        .delta.neutral {{
            background: #f5f5f5;
            color: #757575;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .comparison-table th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        .comparison-table td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .comparison-table tr:hover {{
            background: #f8f9fa;
        }}
        .variant-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            background: #3498db;
            color: white;
        }}
        .wt-badge {{
            background: #2ecc71;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <h2>Summary Metrics Comparison</h2>
        <div class="summary-grid">
"""
    
    # Generate metric cards
    for metric in metrics:
        values = summary_data[metric]
        
        html += f"""
            <div class="metric-card">
                <h3>{metric.replace('_', ' ').title()}</h3>
                <div class="metric-values">
"""
        
        wt_value = values.get(wt_name, 0) if has_wt else None

        for variant in variants:
            value = values[variant]

            # Calculate delta if WT exists
            delta_html = ""
            if has_wt and variant != wt_name and wt_value is not None:
                delta = value - wt_value
                pct_change = ((value - wt_value) / wt_value * 100) if wt_value != 0 else 0
                
                delta_class = "positive" if delta > 0 else ("negative" if delta < 0 else "neutral")
                delta_sign = "+" if delta > 0 else ""
                
                if isinstance(value, float):
                    delta_html = f'<span class="delta {delta_class}">{delta_sign}{delta:.2f} ({delta_sign}{pct_change:.1f}%)</span>'
                else:
                    delta_html = f'<span class="delta {delta_class}">{delta_sign}{delta} ({delta_sign}{pct_change:.1f}%)</span>'
            
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            badge_class = "wt-badge" if variant == wt_name else ""
            
            html += f"""
                    <div class="value-row">
                        <span class="variant-name">
                            <span class="variant-badge {badge_class}">{variant}</span>
                        </span>
                        <div>
                            <span class="value">{value_str}</span>
                            {delta_html}
                        </div>
                    </div>
"""
        
        html += """
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>Detailed Comparison Table</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
"""
    
    for variant in variants:
        badge_class = "wt-badge" if variant == wt_name else "" 
        html += f'<th><span class="variant-badge {badge_class}">{variant}</span></th>'

    if has_wt and len(variants) > 1:
        html += f'<th>Max Δ from {wt_name}</th>'
    
    html += """
                </tr>
            </thead>
            <tbody>
"""
    
    for metric in metrics:
        values = summary_data[metric]
        html += f"""
                <tr>
                    <td><strong>{metric.replace('_', ' ').title()}</strong></td>
"""
        
        for variant in variants:
            value = values[variant]
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            html += f'<td>{value_str}</td>'
        
        # Add max delta column if WT exists
        if has_wt and len(variants) > 1:
            wt_value = values.get(wt_name, 0)
            deltas = [abs(values[v] - wt_value) for v in variants if v != wt_name]
            max_delta = max(deltas) if deltas else 0
            max_delta_str = f"{max_delta:.2f}" if isinstance(max_delta, float) else str(max_delta)
            html += f'<td>{max_delta_str}</td>'
        
        html += """
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    return html


def generate_residue_comparison_table(
    results: Dict[str, Dict],
    title: str = "Residue-by-Residue Comparison",
    show_identical: bool = False,
    filter_interface_only: bool = False,
    wt_name: str = 'WT',
) -> str:
    """
    Generate interactive HTML table comparing all residues across variants.
    
    Shows overall differences in the table, with detailed per-variant information
    in interactive tooltips on hover.
    
    Args:
        results: Dictionary with variant data
        title: Title for the visualization
        show_identical: Show residues with no changes
        filter_interface_only: Only show interface residues
        
    Returns:
        HTML string with interactive comparison table
    """
    residue_comp = compare_residues(results, wt_name=wt_name)
    variants = list(results.keys())
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .controls label {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .controls input[type="checkbox"] {{
            width: 18px;
            height: 18px;
        }}
        .controls input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        .comparison-table th {{
            background: #34495e;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
            cursor: pointer;
            user-select: none;
        }}
        .comparison-table th:hover {{
            background: #2c3e50;
        }}
        .comparison-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .comparison-table tbody tr:hover {{
            background: #f8f9fa;
        }}
        .residue-cell {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .res-type {{
            font-weight: 600;
            font-family: monospace;
        }}
        .mutation-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            background: #f39c12;
            color: white;
        }}
        .deletion-badge {{
            background: #e74c3c;
        }}
        .insertion-badge {{
            background: #3498db;
        }}
        .interface-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            background: #2ecc71;
            color: white;
        }}
        .value-cell {{
            text-align: right;
            font-family: monospace;
        }}
        .value-positive {{
            color: #27ae60;
        }}
        .value-negative {{
            color: #c0392b;
        }}
        .row-mutation {{
            background: #fff9e6 !important;
        }}
        .row-deletion {{
            background: #ffe6e6 !important;
        }}
        .row-insertion {{
            background: #e6f3ff !important;
        }}
        .missing-cell {{
            color: #bdc3c7;
            font-style: italic;
        }}
        .sort-icon {{
            margin-left: 5px;
            font-size: 10px;
        }}
        .stats-bar {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 6px;
        }}
        .stat-item {{
            display: flex;
            flex-direction: column;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-value" id="total-residues">{len(residue_comp)}</span>
                <span class="stat-label">Total Positions</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="mutations-count">
                    {sum(1 for r in residue_comp.values() if r['comparison_type'] == 'mutation')}
                </span>
                <span class="stat-label">Mutations</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="insertions-count">
                    {sum(1 for r in residue_comp.values() if r['comparison_type'] == 'insertion')}
                </span>
                <span class="stat-label">Insertions</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="deletions-count">
                    {sum(1 for r in residue_comp.values() if r['comparison_type'] == 'deletion')}
                </span>
                <span class="stat-label">Deletions</span>
            </div>
        </div>
        
        <div class="controls">
            <label>
                <input type="checkbox" id="show-identical" {'checked' if show_identical else ''}>
                Show Identical Residues
            </label>
            <label>
                <input type="checkbox" id="interface-only" {'checked' if filter_interface_only else ''}>
                Interface Residues Only
            </label>
            <label>
                Search:
                <input type="text" id="search-box" placeholder="Position, residue...">
            </label>
        </div>
        
        <table class="comparison-table" id="comparison-table">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Position <span class="sort-icon"></span></th>
                    <th onclick="sortTable(1)">Type <span class="sort-icon"></span></th>
"""
    
    # Add columns for each variant
    for i, variant in enumerate(variants, start=2):
        html += f'<th onclick="sortTable({i})">{"🔵 " if variant == wt_name else ""}{variant} <span class="sort-icon"></span></th>'
    
    # Add property columns
    prop_col_start = len(variants) + 2
    html += f"""
                    <th onclick="sortTable({prop_col_start})">SASA Δ <span class="sort-icon"></span></th>
                    <th onclick="sortTable({prop_col_start + 1})">Interface (atom-atom contact) <span class="sort-icon"></span></th>
                    <th onclick="sortTable({prop_col_start + 2})">VDW Energy <span class="sort-icon"></span></th>
                    <th onclick="sortTable({prop_col_start + 3})">#Bonds <span class="sort-icon"></span></th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Generate table rows
    for position, data in residue_comp.items():
        comp_type = data['comparison_type']
        
        # Skip identical if not showing
        if not show_identical and comp_type == 'identical':
            # Check if any property changed
            has_property_change = False
            wt_data = data['variants'].get(wt_name)
            if wt_data and wt_data['present']:
                for variant in variants:
                    if variant == wt_name:
                        continue
                    var_data = data['variants'].get(variant)
                    if var_data and var_data['present']:
                        if abs(var_data['sasa']['buried_fraction'] - wt_data['sasa']['buried_fraction']) > 0.05:
                            has_property_change = True
                            break
                        if abs(var_data['vdw_energy'] - wt_data['vdw_energy']) > 1.0:
                            has_property_change = True
                            break
            
            if not has_property_change:
                continue
        
        # Filter interface if requested
        if filter_interface_only:
            is_interface = any(
                v['interface']['is_interface'] 
                for v in data['variants'].values() 
                if v['present'] and v['interface']
            )
            if not is_interface:
                continue
        
        # Row class based on type
        row_class = f"row-{comp_type}" if comp_type != 'identical' else ""
        
        # Type badge
        type_badge = ""
        if comp_type == 'mutation':
            type_badge = '<span class="mutation-badge">MUT</span>'
        elif comp_type == 'deletion':
            type_badge = '<span class="deletion-badge">DEL</span>'
        elif comp_type == 'insertion':
            type_badge = '<span class="insertion-badge">INS</span>'
        
        html += f"""
                <tr class="{row_class}">
                    <td><strong>{position}</strong></td>
                    <td>{type_badge}</td>
"""
        
        # Variant columns
        for variant in variants:
            var_data = data['variants'][variant]
            if var_data['present']:
                res_type = var_data['residue_type']
                interface_badge = '<span class="interface-badge">I</span>' if var_data['interface']['is_interface'] else ''
                
                # Check for mutation
                mutation_info = data.get('mutation_status', {}).get(variant, {})
                if 'mutation' in mutation_info:
                    mut_text = f" ({mutation_info['mutation']})"
                    html += f'<td><div class="residue-cell"><span class="res-type">{res_type}</span>{mut_text} {interface_badge}</div></td>'
                else:
                    html += f'<td><div class="residue-cell"><span class="res-type">{res_type}</span> {interface_badge}</div></td>'
            else:
                html += '<td class="missing-cell">&mdash;</td>'
        
        # Property columns - show range or specific values
        # SASA
        sasa_values = [v['sasa']['buried_fraction'] for v in data['variants'].values() if v['present']]
        if sasa_values:
            min_sasa = min(sasa_values)
            max_sasa = max(sasa_values)
            if abs(max_sasa - min_sasa) > 0.01:
                html += f'<td class="value-cell">{min_sasa:.2f} - {max_sasa:.2f}</td>'
            else:
                html += f'<td class="value-cell">{min_sasa:.2f}</td>'
        else:
            html += '<td class="missing-cell">&mdash;</td>'
        
        # Interface contacts
        contact_values = [v['interface']['contact_count'] for v in data['variants'].values() if v['present'] and v['interface']['is_interface']]
        if contact_values:
            min_contacts = min(contact_values)
            max_contacts = max(contact_values)
            if min_contacts != max_contacts:
                html += f'<td class="value-cell">{min_contacts} - {max_contacts}</td>'
            else:
                html += f'<td class="value-cell">{min_contacts}</td>'
        else:
            html += '<td class="missing-cell">&mdash;</td>'
        
        # VDW Energy
        energy_values = [v['vdw_energy'] for v in data['variants'].values() if v['present']]
        if energy_values:
            min_energy = min(energy_values)
            max_energy = max(energy_values)
            if abs(max_energy - min_energy) > 0.1:
                html += f'<td class="value-cell">{min_energy:.2f} - {max_energy:.2f}</td>'
            else:
                html += f'<td class="value-cell">{min_energy:.2f}</td>'
        else:
            html += '<td class="missing-cell">&mdash;</td>'
        
        # Bonds
        bond_values = [v['bonds']['counts']['total'] for v in data['variants'].values() if v['present']]
        if bond_values:
            min_bonds = min(bond_values)
            max_bonds = max(bond_values)
            if min_bonds != max_bonds:
                html += f'<td class="value-cell">{min_bonds} - {max_bonds}</td>'
            else:
                html += f'<td class="value-cell">{min_bonds}</td>'
        else:
            html += '<td class="missing-cell">&mdash;</td>'
        
        html += """
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <script>
        // Filter controls
        document.getElementById('show-identical').addEventListener('change', filterTable);
        document.getElementById('interface-only').addEventListener('change', filterTable);
        document.getElementById('search-box').addEventListener('input', filterTable);
        
        function filterTable() {
            const showIdentical = document.getElementById('show-identical').checked;
            const interfaceOnly = document.getElementById('interface-only').checked;
            const searchTerm = document.getElementById('search-box').value.toLowerCase();
            
            const rows = document.querySelectorAll('#comparison-table tbody tr');
            let visibleCount = 0;
            
            rows.forEach(row => {
                let show = true;
                
                // Check if row has mutation/insertion/deletion class
                const hasChange = row.classList.contains('row-mutation') || 
                                 row.classList.contains('row-deletion') || 
                                 row.classList.contains('row-insertion');
                
                if (!showIdentical && !hasChange) {
                    show = false;
                }
                
                // Check interface filter
                if (interfaceOnly) {
                    const hasInterface = row.textContent.includes('I');
                    if (!hasInterface) show = false;
                }
                
                // Check search
                if (searchTerm && !row.textContent.toLowerCase().includes(searchTerm)) {
                    show = false;
                }
                
                row.style.display = show ? '' : 'none';
                if (show) visibleCount++;
            });
        }
        
        // Table sorting
        let sortDirection = {};
        
        function sortTable(columnIndex) {
            const table = document.getElementById('comparison-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Toggle sort direction
            sortDirection[columnIndex] = !sortDirection[columnIndex];
            const ascending = sortDirection[columnIndex];
            
            // Update sort icons
            document.querySelectorAll('.sort-icon').forEach(icon => icon.textContent = '');
            const header = table.querySelectorAll('th')[columnIndex];
            header.querySelector('.sort-icon').textContent = ascending ? '▲' : '▼';
            
            rows.sort((a, b) => {
                const aCell = a.cells[columnIndex].textContent.trim();
                const bCell = b.cells[columnIndex].textContent.trim();
                
                // Try numeric comparison
                const aNum = parseFloat(aCell.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bCell.replace(/[^0-9.-]/g, ''));
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return ascending ? aNum - bNum : bNum - aNum;
                }
                
                // String comparison
                return ascending ? 
                    aCell.localeCompare(bCell) : 
                    bCell.localeCompare(aCell);
            });
            
            rows.forEach(row => tbody.appendChild(row));
        }
        
        // Initialize
        filterTable();
    </script>
</body>
</html>
"""
    
    return html


def generate_comparison_dashboard(
    results: Dict[str, Dict],
    title: str = "Protein Complex Comparison Dashboard",
    output_prefix: Optional[str] = None,
    include_3d: bool = True,
    wt_name: str = 'WT',
) -> Dict[str, str]:
    """
    Generate a comprehensive comparison dashboard with all visualizations.
    
    Args:
        results: Dictionary with variant data
        title: Dashboard title
        output_prefix: If provided, saves files with this prefix
        include_3d: Whether to include 3D structure visualization
        
    Returns:
        Dictionary with HTML strings for each visualization type
    """
    
    # Generate all visualizations
    visualizations = {
        'summary': generate_summary_comparison_html(results, title=f"{title} - Summary", wt_name=wt_name),
        'residues': generate_residue_comparison_table(results, title=f"{title} - Residues", wt_name=wt_name),
    }
    
    # Add 3D visualization if requested
    if include_3d:
        visualizations['structure_3d'] = generate_3d_structure_comparison(
            results, 
            title=f"{title} - 3D Structure"
        )
    
    # Save files if output_prefix provided
    if output_prefix:
        for viz_type, html in visualizations.items():
            filename = f"{output_prefix}_{viz_type}.html"
            with open(filename, 'w') as f:
                f.write(html)
            print(f"Saved {viz_type} visualization to {filename}")
    
    return visualizations 



def visualize_comparison(
    results: Dict[str, Dict],
    viz_type: str = 'all',
    output_file: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Main function to generate comparison visualizations.
    
    Args:
        results: Dictionary with variant comparison data
        viz_type: Type of visualization ('summary', 'residues', '3d', 'all')
        output_file: Output file path (without extension)
        **kwargs: Additional arguments passed to visualization functions
        
    Returns:
        Dictionary mapping visualization types to HTML strings or file paths
    """
    
    if viz_type == 'summary':
        html = generate_summary_comparison_html(results, **kwargs)
        if output_file:
            with open(f"{output_file}.html", 'w') as f:
                f.write(html)
            return {'summary': f"{output_file}.html"}
        return {'summary': html}
    
    elif viz_type == 'residues':
        html = generate_residue_comparison_table(results, **kwargs)
        if output_file:
            with open(f"{output_file}.html", 'w') as f:
                f.write(html)
            return {'residues': f"{output_file}.html"}
        return {'residues': html}
    
    elif viz_type == '3d' or viz_type == 'structure_3d':
        html = generate_3d_structure_comparison(results, output_file=output_file, **kwargs)
        if output_file:
            return {'structure_3d': output_file}
        return {'structure_3d': html}
    
    elif viz_type == 'all':
        return generate_comparison_dashboard(results, output_prefix=output_file, **kwargs)
    
    else:
        raise ValueError(f"Unknown viz_type: {viz_type}. Use 'summary', 'residues', '3d', or 'all'")



def generate_3d_structure_comparison(
    results: Dict[str, Dict],
    pdb_files: Tuple[str, str],
    output_file: Optional[str] = None
) -> str:
    """
    Generate interactive 3D visualization of multiple protein complexes using 3Dmol.js.
    
    This creates a real molecular viewer showing actual protein structures with toggleable
    property coloring. Each structure is loaded from PDB files and can be colored by
    different analytical properties.
    
    Args:
        results: Dictionary with variant names as keys, each containing:
            - 'residues': Dict of residue data with properties (SASA, interface, VdW, etc.)
            - 'pdb_path': Optional path to PDB file (can also use pdb_files parameter)
        pdb_files: Optional dict mapping variant names to PDB file paths
            If not provided, will try to use 'pdb_path' from results
        title: Title for the visualization
        output_file: Optional file path to save the HTML
        
    Returns:
        HTML string with interactive 3D molecular visualization
        
    Example:
        >>> results = comparator.analyze()
        >>> pdb_files = [('WT', 'wt.pdb'), ('Variant1', 'var1.pdb')] 
        >>> html = generate_3d_structure_comparison(
        ...     results, 
        ...     pdb_files=pdb_files,
        ...     output_file='comparison_3d.html'
        ... )
    """
    
    variants = list(results.keys())
    
    # Read PDB file contents
    pdb_contents = {}
    for variant, pdb_path in pdb_files:
        try:
            with open(pdb_path, 'r') as f:
                pdb_contents[variant] = f.read()
        except Exception as e:
            print(f"Warning: Could not read PDB file for {variant}: {e}")
            pdb_contents[variant] = ""
    
    # Prepare residue property data
    property_data = {}
    for variant in variants:
        residues = results[variant]['residues']
        variant_props = {}
        
        for res_id, res_data in residues.items():
            # Parse residue ID: "SER A:32" -> chain, resseq
            parts = res_id.split()
            if len(parts) != 2:
                continue
                
            chain_pos = parts[1]  # "A:32"
            chain, position = chain_pos.split(':')
            
            # Extract numeric part of position (handle insertion codes)
            resseq = int(''.join(filter(str.isdigit, position)))
            
            # Get properties
            sasa_buried = res_data['sasa']['buried_fraction'] if res_data['sasa'] else 0
            interface_data = _parse_interface_data(res_data['interface'])
            is_interface = 1.0 if interface_data['is_interface'] else 0.0
            vdw_energy = res_data['vdw_energy'] if res_data['vdw_energy'] is not None else 0
            bonds_data = _parse_bonds(res_data['bonds'])
            total_contacts = bonds_data['counts']['total']
            
            # Store by chain:resseq key for easy lookup
            key = f"{chain}:{resseq}"
            variant_props[key] = {
                'sasa': sasa_buried,
                'interface': is_interface,
                'vdw': vdw_energy,
                'contacts': total_contacts,
                'res_id': res_id
            }
        
        property_data[variant] = variant_props

    # Derive chain IDs that appear in the analysis results (the "selected" chains)
    analysis_chains = sorted({
        key.split(':')[0]
        for variant_props in property_data.values()
        for key in variant_props
    })

    # Generate HTML with 3Dmol.js
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">

    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        html, body {{
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #fcfdfe;
        }}

        .container {{
            height: 100%;
            background: white;
            overflow: hidden;
        }}

        .main-content {{
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }}

        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px 10px;
            padding: 7px 12px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            flex-shrink: 0;
        }}

        .toolbar-group {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .toolbar-sep {{
            width: 1px;
            height: 22px;
            background: #ddd;
            flex-shrink: 0;
        }}

        .toolbar-label {{
            font-size: 12px;
            font-weight: 600;
            color: #555;
            white-space: nowrap;
        }}

        .toolbar select {{
            width: auto;
            padding: 4px 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 13px;
            font-family: inherit;
        }}

        .toolbar button {{
            width: auto;
            padding: 4px 9px;
            border-radius: 5px;
            font-size: 12px;
            margin-top: 0;
            white-space: nowrap;
        }}

        .toolbar input[type="checkbox"] {{
            width: 14px;
            height: 14px;
            cursor: pointer;
        }}

        .viewer-container {{
            flex: 1;
            position: relative;
            background: #000;
            min-width: 0;
            min-height: 0;
        }}

        .legend-overlay {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.92);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.25);
            padding: 10px 14px;
            max-width: 220px;
            z-index: 500;
        }}

        #viewport {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        
        .control-section {{
            margin-bottom: 25px;
        }}
        
        .control-label {{
            font-size: 13px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        select, button {{
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
            font-family: inherit;
        }}
        
        select:hover, select:focus {{
            border-color: #667eea;
            outline: none;
        }}
        
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        .variant-checkboxes {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .checkbox-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .checkbox-item:hover {{
            border-color: #667eea;
            background: #f0f4ff;
        }}
        
        .checkbox-item input[type="checkbox"] {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            cursor: pointer;
        }}
        
        .checkbox-item label {{
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            flex: 1;
        }}
        
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            margin-top: 15px;
        }}
        
        .legend-title {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        
        .legend-gradient {{
            height: 20px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
        }}
        
        .info-badge {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            margin-top: 15px;
            line-height: 1.6;
        }}
        
        .residue-info-panel {{
            position: absolute;
            top: 8px;
            right: 8px;
            width: 300px;
            max-height: 400px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            padding: 20px;
            display: none;
            overflow-y: auto;
            z-index: 1000;
        }}
        
        .residue-info-panel.visible {{
            display: block;
        }}
        
        .residue-info-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .residue-info-title {{
            font-size: 16px;
            font-weight: 700;
            color: #333;
        }}
        
        .close-btn {{
            background: #ff6b6b;
            color: white;
            border: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
            padding: 0;
        }}
        
        .close-btn:hover {{
            background: #ff5252;
        }}
        
        .residue-info-content {{
            font-size: 13px;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .info-label {{
            font-weight: 600;
            color: #666;
        }}
        
        .info-value {{
            color: #333;
            text-align: right;
        }}
        
        .info-section {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 2px solid #f0f0f0;
        }}
        
        .info-section-title {{
            font-weight: 700;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            text-align: center;
        }}
        
        .loading-spinner {{
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .style-buttons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .style-button {{
            padding: 8px;
            font-size: 12px;
            margin: 0;
        }}
        
        .property-info {{
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            padding: 8px;
            background: #f0f4ff;
            border-radius: 6px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!--div class="header">
            <h1>Interactive Molecular Visualization with Property Coloring</h1>
        </div-->
        
        <div class="main-content">
            <div class="toolbar">
                <div class="toolbar-group">
                    <span class="toolbar-label">Color:</span>
                    <select id="property-select">
                        <option value="default">By Variant (distinct colors)</option>
                        <option value="sasa">SASA (Buried Fraction)</option>
                        <option value="interface">Interface Residues</option>
                        <option value="vdw">VdW Energy</option>
                        <option value="contacts">Total Contacts</option>
                        <option value="bfactor">B-factor</option>
                        <option value="chain">Chain ID</option>
                    </select>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <span class="toolbar-label">Style:</span>
                    <select id="style-select">
                        <option value="cartoon">Cartoon</option>
                        <option value="stick">Stick</option>
                        <option value="sphere">Sphere</option>
                        <option value="line">Line</option>
                        <option value="surface">Surface</option>
                    </select>
                    <button class="style-button" onclick="toggleSideChains()">Side Chains</button>
                    <button class="style-button" onclick="toggleChainOutlines()">Outlines</button>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <label style="display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer;">
                        <input type="checkbox" id="analysis-only" checked onchange="toggleAnalysisOnly()">
                        <span>Analysis chains only</span>
                    </label>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <span class="toolbar-label">Diff:</span>
                    <select id="difference-mode">
                        <option value="none">None</option>
                        <option value="mutations">Mutations Only</option>
                        <option value="changed">Changed Regions</option>
                    </select>
                    <label style="display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer;">
                        <input type="checkbox" id="fade-similar" onchange="updateVisualization()">
                        <span>Fade similar</span>
                    </label>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <span class="toolbar-label">Variants:</span>
"""
    
    # Add variant checkboxes (compact inline)
    for i, variant in enumerate(variants):
        checked = "checked" if i == 0 else ""
        html += f"""
                    <label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;padding:3px 7px;background:white;border:1px solid #ddd;border-radius:5px;">
                        <input type="checkbox" id="variant-{variant}" value="{variant}" {checked} onchange="updateVisualization()">
                        <span>{variant}</span>
                    </label>
"""

    html += """
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <button onclick="updateVisualization()">Update</button>
                    <button onclick="resetView()">Reset</button>
                    <button onclick="downloadImage()">Save</button>
                </div>
            </div>

            <div class="viewer-container">
                <div id="viewport"></div>
                <div class="loading" id="loading">
                    <div class="loading-spinner"></div>
                    Loading structures...
                </div>

                <!-- Legend overlay (bottom-left of viewer) -->
                <div class="legend-overlay" id="legend">
                    <div id="legend-content"></div>
                </div>

                <!-- Residue Information Panel (top-right of viewer) -->
                <div class="residue-info-panel" id="residue-info-panel">
                    <div class="residue-info-header">
                        <div class="residue-info-title" id="residue-title">Residue Info</div>
                        <button class="close-btn" onclick="closeResidueInfo()">×</button>
                    </div>
                    <div class="residue-info-content" id="residue-info-content">
                        Click on a residue to see details
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
"""
    
    # Embed PDB data and property data as JavaScript
    html += f"""
        // PDB data
        const pdbData = {json.dumps(pdb_contents, indent=2)};
        
        // Property data for each variant
        const propertyData = {json.dumps(property_data, indent=2)};
        
        // Variant list
        const variants = {json.dumps(variants)};

        // Chains included in the analysis (non-analysis chains can be hidden)
        const analysisChains = {json.dumps(analysis_chains)};

        // Initialize 3Dmol viewer
        let viewer = null;
        let showSideChains = false;
        let showChainOutlines = false;
        let showAnalysisOnly = true;
        
        // Cache for difference detection (avoid recalculating)
        let cachedMutations = null;
        let cachedChangedResidues = null;
        let lastDifferenceMode = 'none';
        
        // Define distinct colors for each variant
        const variantColors = [
            '#FF6B6B', // Red
            '#52B788',  // Green
            '#45B7D1', // Blue
            '#BB8FCE', // Purple
            '#F7DC6F', // Yellow
            '#4ECDC4', // Teal
            '#FFA07A', // Light Salmon
            '#98D8C8', // Mint
            '#85C1E2', // Sky Blue
            '#F8B739' // Orange
            
        ];
        
        // Map each variant to a color
        const variantColorMap = {{}};
        variants.forEach((variant, idx) => {{
            variantColorMap[variant] = variantColors[idx % variantColors.length];
        }});
        
        // Color scales configuration
        const colorScales = {{
            sasa: {{
                name: 'SASA Buried Fraction',
                min: 0,
                max: 1,
                colors: ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b'],
                info: 'Blue = buried (0), Red = exposed (1)',
                getLabel: (v) => v.toFixed(2)
            }},
            interface: {{
                name: 'Interface Residues',
                min: 0,
                max: 1,
                colors: ['#cccccc', '#ff6b6b'],
                info: 'Gray = non-interface, Red = interface',
                getLabel: (v) => v === 1 ? 'Interface' : 'Core'
            }},
            vdw: {{
                name: 'VdW Energy (kcal/mol)',
                min: -3,
                max: 3,
                colors: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
                info: 'Blue = favorable (negative), White = neutral (0), Red = unfavorable/clashing (positive)',
                getLabel: (v) => v.toFixed(1)
            }},
            contacts: {{
                name: 'Total Contacts',
                min: 0,
                max: 6,
                colors: ['#f0f0f0', '#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
                info: 'Light gray = no contacts, Orange/Red = many contacts',
                getLabel: (v) => Math.round(v)
            }},
            bfactor: {{
                name: 'B-factor',
                min: 5,
                max: 70,
                colors: ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020'],
                info: 'Blue = low B-factor (rigid), Red = high B-factor (flexible)',
                getLabel: (v) => Math.round(v)
            }},
            chain: {{
                name: 'Chain ID',
                categorical: true,
                colors: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#17becf', '#bcbd22'],
                info: 'Each chain colored with a distinct color.',
                getLabel: (v) => v
            }}
        }};
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', function() {{
            const element = document.getElementById('viewport');
            const config = {{ backgroundColor: 'white' }};
            viewer = $3Dmol.createViewer(element, config);

            // Delay first render so CSS flex layout computes element dimensions
            // before 3Dmol measures the canvas (avoids wrong initial zoom/size)
            requestAnimationFrame(() => requestAnimationFrame(() => updateVisualization()));
            
            // Event listeners
            document.getElementById('property-select').addEventListener('change', updateVisualization);
            document.getElementById('style-select').addEventListener('change', updateVisualization);
            document.getElementById('difference-mode').addEventListener('change', updateVisualization);
        }});
        
        function getSelectedVariants() {{
            const selected = [];
            variants.forEach(variant => {{
                const checkbox = document.getElementById(`variant-${{variant}}`);
                if (checkbox && checkbox.checked) {{
                    selected.push(variant);
                }}
            }});
            return selected;
        }}
        
        function identifyMutations(selectedVariants) {{
            // Identify positions where residue type differs between variants
            const mutations = new Map(); // key: chain:position, value: {{variants: {{variant: resType}}}}
            
            selectedVariants.forEach(variant => {{
                const props = propertyData[variant];
                if (!props) return;
                
                Object.keys(props).forEach(posKey => {{
                    if (!mutations.has(posKey)) {{
                        mutations.set(posKey, {{ variants: {{}} }});
                    }}
                    // We'd need residue type from property data - for now mark all as potential differences
                    mutations.get(posKey).variants[variant] = props[posKey].res_id;
                }});
            }});
            
            // Filter to only positions that differ
            const realMutations = new Map();
            mutations.forEach((value, key) => {{
                const resTypes = new Set(Object.values(value.variants));
                if (resTypes.size > 1) {{
                    realMutations.set(key, value);
                }}
            }});
            
            return realMutations;
        }}
        
        function identifyChangedResidues(selectedVariants) {{
            // Identify residues with significant property changes
            const changed = new Map();
            
            if (selectedVariants.length < 2) return changed;
            
            // Compare first variant to others
            const reference = selectedVariants[0];
            const refProps = propertyData[reference];
            if (!refProps) return changed;
            
            Object.keys(refProps).forEach(posKey => {{
                let maxDiff = 0;
                
                selectedVariants.slice(1).forEach(variant => {{
                    const varProps = propertyData[variant];
                    if (!varProps || !varProps[posKey]) return;
                    
                    // Calculate property differences
                    const sasaDiff = Math.abs(refProps[posKey].sasa - varProps[posKey].sasa);
                    const vdwDiff = Math.abs(refProps[posKey].vdw - varProps[posKey].vdw);
                    const contactDiff = Math.abs(refProps[posKey].contacts - varProps[posKey].contacts);
                    
                    // Weighted difference score
                    const diff = sasaDiff * 2 + vdwDiff * 0.5 + contactDiff * 0.3;
                    maxDiff = Math.max(maxDiff, diff);
                }});
                
                if (maxDiff > 0.3) {{ // Threshold for "significant" change
                    changed.set(posKey, maxDiff);
                }}
            }});
            
            return changed;
        }}
        
        function applyVisualizationStyle(variant, modelIdx, property, style, differenceMode, fadeSimilar, mutations, changedResidues) {{
            let styleSpec = {{}};
            const variantColor = variantColorMap[variant];
            const props = propertyData[variant];
            
            // Determine which residues to highlight based on difference mode
            let highlightPositions = new Set();
            
            if (differenceMode === 'mutations') {{
                mutations.forEach((value, key) => highlightPositions.add(key));
            }} else if (differenceMode === 'changed') {{
                changedResidues.forEach((value, key) => highlightPositions.add(key));
            }}
            
            // Base style
            if (property === 'default') {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ color: variantColor, clickable: true }};
                }} else if (style === 'stick') {{
                    styleSpec.stick = {{ radius: 0.15, color: variantColor, clickable: true }};
                }} else if (style === 'sphere') {{
                    styleSpec.sphere = {{ radius: 1.0, color: variantColor, clickable: true }};
                }} else if (style === 'line') {{
                    styleSpec.line = {{ color: variantColor, clickable: true }};
                }} else if (style === 'surface') {{
                    styleSpec.surface = {{ opacity: 0.8, color: variantColor, clickable: true }};
                }}
            }} else if (property === 'bfactor') {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ 
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }},
                        clickable: true
                    }};
                }} else {{
                    styleSpec[style] = {{ 
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }},
                        clickable: true
                    }};
                }}
            }} else {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ clickable: true }};
                }} else if (style === 'stick') {{
                    styleSpec.stick = {{ radius: 0.15, clickable: true }};
                }} else if (style === 'sphere') {{
                    styleSpec.sphere = {{ radius: 1.0, clickable: true }};
                }} else if (style === 'line') {{
                    styleSpec.line = {{ clickable: true }};
                }} else if (style === 'surface') {{
                    styleSpec.surface = {{ opacity: 0.8, clickable: true }};
                }}
            }}
            
            // Apply base style to all
            viewer.setStyle({{ model: modelIdx }}, styleSpec);
            
            // For surface representation, need to explicitly add surface
            if (style === 'surface') {{
                // Remove any existing surfaces first
                viewer.removeAllSurfaces();
                
                if (property === 'default') {{
                    // Add uniform colored surface
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        color: variantColor
                    }}, {{ model: modelIdx }});
                }} else if (property === 'bfactor') {{
                    // Add surface colored by B-factor
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }}
                    }}, {{ model: modelIdx }});
                }} else {{
                    // For other properties, we'll use B-factor coloring as base
                    // (per-residue property coloring on surfaces is complex in 3Dmol)
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        color: variantColor
                    }}, {{ model: modelIdx }});
                }}
            }}
            
            // Apply property coloring if not default (works for non-surface styles)
            // Chain coloring: color each chain with a distinct categorical color
            if (property === 'chain') {{
                const chainPalette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#17becf', '#bcbd22'];
                const modelData = viewer.getModel(modelIdx);
                if (modelData) {{
                    const atoms = modelData.selectedAtoms({{}});
                    const chains = [...new Set(atoms.map(a => a.chain))].sort();
                    chains.forEach((chainId, chainIdx) => {{
                        const chainColor = chainPalette[chainIdx % chainPalette.length];
                        const chainStyleSpec = {{}};
                        if (style === 'cartoon') chainStyleSpec.cartoon = {{ color: chainColor, clickable: true }};
                        else if (style === 'stick') chainStyleSpec.stick = {{ radius: 0.15, color: chainColor, clickable: true }};
                        else if (style === 'sphere') chainStyleSpec.sphere = {{ radius: 1.0, color: chainColor, clickable: true }};
                        else if (style === 'line') chainStyleSpec.line = {{ color: chainColor, clickable: true }};
                        else chainStyleSpec[style] = {{ color: chainColor, clickable: true }};
                        viewer.setStyle({{ chain: chainId, model: modelIdx }}, chainStyleSpec);
                    }});
                }}
            }}

            if (property !== 'default' && property !== 'bfactor' && property !== 'chain' && style !== 'surface' && props) {{
                Object.keys(props).forEach(key => {{
                    const [chain, resseq] = key.split(':');
                    const value = props[key][property];

                    if (value !== undefined) {{
                        const color = getColor(property, value, variant);
                        if (color) {{
                            const sel = {{ chain: chain, resi: parseInt(resseq), model: modelIdx }};
                            const coloredStyle = {{ ...styleSpec }};
                            coloredStyle[style] = {{ ...styleSpec[style], color: color, clickable: true }};
                            viewer.setStyle(sel, coloredStyle);
                        }}
                    }}
                }});
            }}

            // Apply difference highlighting - RESPECTING COLOR SCHEME
            if (differenceMode !== 'none' && props) {{
                if (differenceMode === 'changed') {{
                    // Fade or highlight based on whether residue changed
                    Object.keys(props).forEach(key => {{
                        const [chain, resseq] = key.split(':');
                        const isHighlighted = highlightPositions.has(key);
                        const sel = {{ chain: chain, resi: parseInt(resseq), model: modelIdx }};
                        
                        if (fadeSimilar && !isHighlighted) {{
                            // Fade non-changed residues
                            const fadedStyle = {{}};
                            fadedStyle[style] = {{ opacity: 0.15, clickable: true }};
                            viewer.setStyle(sel, fadedStyle);
                        }} else if (isHighlighted) {{
                            // Highlight changed residues - keep their property color
                            const highlightStyle = {{}};
                            
                            // Get the current color for this residue from property scheme
                            let currentColor = variantColor; // default fallback
                            if (property !== 'default' && property !== 'bfactor') {{
                                const value = props[key][property];
                                if (value !== undefined) {{
                                    const propColor = getColor(property, value, variant);
                                    if (propColor) currentColor = propColor;
                                }}
                            }}
                            
                            // Emphasize with thicker representation
                            if (style === 'cartoon') {{
                                highlightStyle.cartoon = {{ color: currentColor, thickness: 1.5, clickable: true }};
                            }} else if (style === 'sphere') {{
                                highlightStyle.sphere = {{ radius: 1.3, color: currentColor, clickable: true }};
                            }} else {{
                                highlightStyle[style] = {{ color: currentColor, clickable: true }};
                            }}
                            
                            // Add stick outline for extra emphasis
                            highlightStyle.stick = {{ radius: 0.3, color: currentColor, clickable: true }};
                            
                            viewer.setStyle(sel, highlightStyle);
                        }}
                    }});
                }} else if (differenceMode === 'mutations') {{
                    // Highlight mutation positions - keep property colors
                    highlightPositions.forEach(posKey => {{
                        const [chain, resseq] = posKey.split(':');
                        const sel = {{ chain: chain, resi: parseInt(resseq), model: modelIdx }};
                        
                        // Get the current color for this residue from property scheme
                        let currentColor = variantColor; // default fallback
                        if (property !== 'default' && property !== 'bfactor' && props[posKey]) {{
                            const value = props[posKey][property];
                            if (value !== undefined) {{
                                const propColor = getColor(property, value, variant);
                                if (propColor) currentColor = propColor;
                            }}
                        }}
                        
                        // Add emphasis while keeping the property color
                        const mutStyle = {{}};
                        mutStyle.stick = {{ radius: 0.4, color: currentColor, clickable: true }};
                        mutStyle.sphere = {{ radius: 1.8, color: currentColor, opacity: 0.8, clickable: true }};
                        viewer.setStyle(sel, mutStyle, {{ keepStyle: true }});
                    }});
                    
                    // Fade everything else if requested
                    if (fadeSimilar) {{
                        Object.keys(props).forEach(key => {{
                            if (!highlightPositions.has(key)) {{
                                const [chain, resseq] = key.split(':');
                                const sel = {{ chain: chain, resi: parseInt(resseq), model: modelIdx }};
                                const fadedStyle = {{}};
                                fadedStyle[style] = {{ opacity: 0.15, clickable: true }};
                                viewer.setStyle(sel, fadedStyle);
                            }}
                        }});
                    }}
                }}
            }}
        }}
        
        function getColor(property, value, variant) {{
            if (property === 'default' || !property) {{
                return null; // Use default coloring
            }}
            
            const scale = colorScales[property];
            if (!scale) return null;
            
            // Normalize value to [0, 1]
            const normalized = Math.max(0, Math.min(1, (value - scale.min) / (scale.max - scale.min)));
            
            // Get color from gradient
            const colors = scale.colors;
            const idx = normalized * (colors.length - 1);
            const lower = Math.floor(idx);
            const upper = Math.ceil(idx);
            const frac = idx - lower;
            
            if (lower === upper) {{
                return colors[lower];
            }}

            // Interpolate between adjacent colors
            const hex2rgb = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
            const rgb2hex = (r,g,b) => '#' + [r,g,b].map(v => Math.round(v).toString(16).padStart(2,'0')).join('');
            const c1 = hex2rgb(colors[lower]), c2 = hex2rgb(colors[upper]);
            return rgb2hex(c1[0]+(c2[0]-c1[0])*frac, c1[1]+(c2[1]-c1[1])*frac, c1[2]+(c2[2]-c1[2])*frac);
        }}
        
        function updateVisualization() {{
            if (!viewer) return;
            
            const property = document.getElementById('property-select').value;
            const style = document.getElementById('style-select').value;
            const displayMode = 'all';
            const differenceMode = document.getElementById('difference-mode').value;
            const fadeSimilar = document.getElementById('fade-similar').checked;
            const selectedVariants = getSelectedVariants();
            
            if (selectedVariants.length === 0) {{
                alert('Please select at least one variant to display');
                return;
            }}
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            
            // Clear viewer
            viewer.clear();
            
            // Only calculate differences if difference mode is active AND it changed
            let mutations = new Map();
            let changedResidues = new Map();
            
            if (differenceMode !== 'none') {{
                // Use cache if mode hasn't changed
                if (differenceMode === lastDifferenceMode && cachedMutations && cachedChangedResidues) {{
                    mutations = cachedMutations;
                    changedResidues = cachedChangedResidues;
                }} else {{
                    // Calculate fresh
                    if (differenceMode === 'mutations' || differenceMode === 'rmsd') {{
                        mutations = identifyMutations(selectedVariants);
                    }}
                    if (differenceMode === 'changed') {{
                        changedResidues = identifyChangedResidues(selectedVariants);
                    }}
                    // Cache results
                    cachedMutations = mutations;
                    cachedChangedResidues = changedResidues;
                    lastDifferenceMode = differenceMode;
                }}
            }} else {{
                // Clear cache when not using difference mode
                cachedMutations = null;
                cachedChangedResidues = null;
                lastDifferenceMode = 'none';
            }}
            
            // Calculate positions for grid layout
            const spacing = 60;
            const gridSize = Math.ceil(Math.sqrt(selectedVariants.length));
            
            selectedVariants.forEach((variant, idx) => {{
                if (!pdbData[variant]) {{
                    console.warn(`No PDB data for ${{variant}}`);
                    return;
                }}
                
                // Add model
                const model = viewer.addModel(pdbData[variant], 'pdb');
                
                // Position for grid layout
                if (displayMode === 'grid') {{
                    const row = Math.floor(idx / gridSize);
                    const col = idx % gridSize;
                    const offsetX = (col - gridSize / 2) * spacing;
                    const offsetY = (row - gridSize / 2) * spacing;
                    model.setCoordinates(model.getCoordinates().map(coord => ({{
                        x: coord.x + offsetX,
                        y: coord.y + offsetY,
                        z: coord.z
                    }})));
                }}
                
                // Apply styling - single call, no duplication
                applyVisualizationStyle(variant, idx, property, style, differenceMode, fadeSimilar, mutations, changedResidues);
            }});
            
            // Add side chains if requested
            if (showSideChains && property === 'default') {{
                viewer.setStyle({{ atom: ['CA', 'C', 'N', 'O'] }}, {{ stick: {{ radius: 0.2, clickable: true }} }}, {{ byres: false }});
            }}
            
            // Add chain outlines if requested (helps distinguish chains with property coloring)
            if (showChainOutlines) {{
                selectedVariants.forEach((variant, idx) => {{
                    if (!pdbData[variant]) return;
                    
                    // Get all unique chains in this variant
                    const modelData = viewer.getModel(idx);
                    if (!modelData) return;
                    
                    const atoms = modelData.selectedAtoms({{}});
                    const chains = [...new Set(atoms.map(a => a.chain))];
                    
                    // Define colors for chain outlines (darker, more saturated colors)
                    const chainOutlineColors = [
                        '#8B0000', // Dark Red
                        '#006064', // Dark Cyan  
                        '#1A237E', // Dark Blue
                        '#E65100', // Dark Orange
                        '#004D40', // Dark Teal
                        '#F57F17', // Dark Yellow
                        '#4A148C', // Dark Purple
                        '#01579B', // Dark Sky Blue
                        '#BF360C', // Dark Deep Orange
                        '#1B5E20'  // Dark Green
                    ];
                    
                    chains.forEach((chain, chainIdx) => {{
                        const outlineColor = chainOutlineColors[chainIdx % chainOutlineColors.length];
                        
                        // Add thin cartoon trace for chain outline
                        viewer.setStyle(
                            {{ chain: chain, model: idx }},
                            {{ cartoon: {{ color: outlineColor, thickness: 0.4, opacity: 0.7 }} }},
                            {{ keepStyle: true }}
                        );
                    }});
                }});
            }}

            // Hide non-analysis chains if requested
            if (showAnalysisOnly && analysisChains.length > 0) {{
                selectedVariants.forEach((variant, idx) => {{
                    if (!pdbData[variant]) return;
                    const modelData = viewer.getModel(idx);
                    if (!modelData) return;
                    const atoms = modelData.selectedAtoms({{}});
                    const allChains = [...new Set(atoms.map(a => a.chain))];
                    allChains.forEach(chain => {{
                        if (!analysisChains.includes(chain)) {{
                            viewer.setStyle({{ chain: chain, model: idx }}, {{}});
                        }}
                    }});
                }});
            }}

            // Set up click handler after styles are applied
            viewer.setClickable({{}}, true, function(atom, viewer, event, container) {{
                if (atom) {{
                    showResidueInfo(atom);
                    event.stopPropagation();
                }}
            }});
            
            viewer.resize();
            const zoomSel = (showAnalysisOnly && analysisChains.length > 0)
                ? {{ chain: analysisChains }} : {{}};
            viewer.zoomTo(zoomSel);
            viewer.render();

            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            // Update legend
            updateLegend(property, selectedVariants);
            
            // Update property info
            updatePropertyInfo(property);
        }}
        
        function updateLegend(property, selectedVariants) {{
            const legendContent = document.getElementById('legend-content');
            
            if (property === 'default') {{
                // Show variant color assignments
                let html = '<div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">Variant Colors</div>';
                selectedVariants.forEach(variant => {{
                    const color = variantColorMap[variant];
                    html += `
                        <div style="display: flex; align-items: center; margin: 6px 0;">
                            <div style="width: 24px; height: 16px; background: ${{color}}; border-radius: 3px; margin-right: 8px; border: 1px solid #ccc;"></div>
                            <span style="font-size: 12px;">${{variant}}</span>
                        </div>
                    `;
                }});
                legendContent.innerHTML = html;
                return;
            }}
            
            const scale = colorScales[property];
            if (!scale) return;

            if (scale.categorical) {{
                // Chain: show dynamic per-chain swatches from propertyData keys
                const chainPalette = scale.colors;
                const firstVariant = variants[0];
                const props = propertyData[firstVariant];
                const allChains = props ? [...new Set(Object.keys(props).map(k => k.split(':')[0]))].sort() : [];
                let html = `<div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">${{scale.name}}</div>`;
                allChains.forEach((chainId, idx) => {{
                    const color = chainPalette[idx % chainPalette.length];
                    html += `<div style="display:flex;align-items:center;margin:4px 0;"><div style="width:20px;height:14px;background:${{color}};border-radius:3px;margin-right:8px;border:1px solid #ccc;flex-shrink:0;"></div><span style="font-size:12px;">Chain ${{chainId}}</span></div>`;
                }});
                html += `<div style="font-size:11px;color:#666;margin-top:8px;">${{scale.info}}</div>`;
                legendContent.innerHTML = html;
                return;
            }}

            const gradient = scale.colors.join(', ');

            legendContent.innerHTML = `
                <div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">${{scale.name}}</div>
                <div class="legend-gradient" style="background: linear-gradient(to right, ${{gradient}});"></div>
                <div class="legend-labels">
                    <span>${{scale.getLabel(scale.min)}}</span>
                    <span>${{scale.getLabel(scale.max)}}</span>
                </div>
                <div style="font-size: 11px; color: #666; margin-top: 8px;">${{scale.info}}</div>
            `;
        }}

        function updatePropertyInfo(property) {{
            const infoDiv = document.getElementById('property-info');
            if (!infoDiv) return;

            const descriptions = {{
                'default': 'Each variant/complex is colored with a distinct color for easy identification.',
                'sasa': 'Solvent Accessible Surface Area - measures how buried each residue is in the structure.',
                'interface': 'Highlights residues at protein-protein interfaces (contacts with other chains).',
                'vdw': 'Van der Waals energy - negative values indicate favorable interactions.',
                'contacts': 'Number of atomic contacts each residue makes with neighboring residues.',
                'bfactor': 'Temperature/B-factor from PDB - indicates structural flexibility or disorder.',
                'chain': 'Colors each chain with a distinct color for easy identification of chain boundaries.'
            }};

            infoDiv.textContent = descriptions[property] || '';
        }}
        
        function toggleAnalysisOnly() {{
            showAnalysisOnly = document.getElementById('analysis-only').checked;
            updateVisualization();
        }}

        function toggleSideChains() {{
            showSideChains = !showSideChains;
            updateVisualization();
        }}

        function toggleChainOutlines() {{
            showChainOutlines = !showChainOutlines;
            updateVisualization();
        }}

        function resetView() {{
            if (viewer) {{
                viewer.resize();
                const zoomSel = (showAnalysisOnly && analysisChains.length > 0)
                    ? {{ chain: analysisChains }} : {{}};
                viewer.zoomTo(zoomSel);
                viewer.render();
            }}
        }}
        
        function downloadImage() {{
            if (viewer) {{
                viewer.render();
                const imgData = viewer.pngURI();
                const link = document.createElement('a');
                link.href = imgData;
                link.download = '3d_structure_comparison.png';
                link.click();
            }}
        }}
        
        function showResidueInfo(atom) {{
            const panel = document.getElementById('residue-info-panel');
            const title = document.getElementById('residue-title');
            const content = document.getElementById('residue-info-content');
            
            // Get atom information
            const resname = atom.resn || 'Unknown';
            const chain = atom.chain || '?';
            const resi = atom.resi || '?';
            const atomName = atom.atom || '?';
            const elem = atom.elem || '?';
            const bfactor = atom.b !== undefined ? atom.b.toFixed(2) : 'N/A';
            
            // Get model index to identify variant
            const modelIndex = atom.model || 0;
            const selectedVariants = getSelectedVariants();
            const variant = selectedVariants[modelIndex] || 'Unknown';
            
            // Find residue properties from our data
            const resKey = `${{chain}}:${{resi}}`;
            const variantProps = propertyData[variant];
            const resProps = variantProps ? variantProps[resKey] : null;
            
            // Build title
            title.textContent = `${{resname}} ${{chain}}:${{resi}}`;
            
            // Build content
            let html = `
                <div class="info-row">
                    <span class="info-label">Variant:</span>
                    <span class="info-value" style="color: ${{variantColorMap[variant]}}; font-weight: 700;">${{variant}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Residue Type:</span>
                    <span class="info-value">${{resname}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Chain:</span>
                    <span class="info-value">${{chain}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Position:</span>
                    <span class="info-value">${{resi}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Atom Clicked:</span>
                    <span class="info-value">${{atomName}} (${{elem}})</span>
                </div>
                <div class="info-row">
                    <span class="info-label">B-factor:</span>
                    <span class="info-value">${{bfactor}}</span>
                </div>
            `;
            
            // Add analysis properties if available
            if (resProps) {{
                html += `
                    <div class="info-section">
                        <div class="info-section-title">Analysis Properties</div>
                        
                        <div class="info-row">
                            <span class="info-label">SASA (Buried):</span>
                            <span class="info-value">${{resProps.sasa.toFixed(2)}}</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">Interface:</span>
                            <span class="info-value">${{resProps.interface === 1 ? 'Yes' : 'No'}}</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">VdW Energy:</span>
                            <span class="info-value">${{resProps.vdw.toFixed(2)}} kcal/mol</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">Total Contacts:</span>
                            <span class="info-value">${{resProps.contacts}}</span>
                        </div>
                    </div>
                `;
            }} else {{
                html += `
                    <div class="info-section">
                        <div class="info-section-title">Analysis Properties</div>
                        <div style="color: #999; font-style: italic; font-size: 12px;">
                            No analysis data available for this residue
                        </div>
                    </div>
                `;
            }}
            
            content.innerHTML = html;
            panel.classList.add('visible');
        }}
        
        function closeResidueInfo() {{
            const panel = document.getElementById('residue-info-panel');
            panel.classList.remove('visible');
        }}
    </script>
</body>
</html>
"""
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Saved 3D visualization to {output_file}")
    
    return html




def generate_3d_sidebyside_comparison(
    results: Dict[str, Dict],
    pdb_files: Tuple[str, str],
    output_file: Optional[str] = None
) -> str:
    """
    Generate side-by-side 3D visualization of multiple protein complexes with synchronized controls.
    
    Each structure is displayed in its own viewer panel, but all viewers share the same
    styling, coloring, and viewing options for easy comparison.
    
    Args:
        results: Dictionary with variant names as keys, each containing:
            - 'residues': Dict of residue data with properties (SASA, interface, VdW, etc.)
        pdb_files: List of tuples mapping variant names to PDB file paths
        title: Title for the visualization
        output_file: Optional file path to save the HTML
        
    Returns:
        HTML string with side-by-side 3D molecular visualization
    """
    
    variants = list(results.keys())
    
    # Read PDB file contents
    pdb_contents = {}
    for variant, pdb_path in pdb_files:
        try:
            with open(pdb_path, 'r') as f:
                pdb_contents[variant] = f.read()
        except Exception as e:
            print(f"Warning: Could not read PDB file for {variant}: {e}")
            pdb_contents[variant] = ""
    
    # Prepare residue property data
    property_data = {}
    for variant in variants:
        residues = results[variant]['residues']
        variant_props = {}
        
        for res_id, res_data in residues.items():
            # Parse residue ID: "SER A:32" -> chain, resseq
            parts = res_id.split()
            if len(parts) != 2:
                continue
                
            chain_pos = parts[1]  # "A:32"
            chain, position = chain_pos.split(':')
            
            # Extract numeric part of position (handle insertion codes)
            resseq = int(''.join(filter(str.isdigit, position)))
            
            # Get properties
            sasa_buried = res_data['sasa']['buried_fraction'] if res_data['sasa'] else 0
            interface_data = _parse_interface_data(res_data['interface'])
            is_interface = 1.0 if interface_data['is_interface'] else 0.0
            vdw_energy = res_data['vdw_energy'] if res_data['vdw_energy'] is not None else 0
            bonds_data = _parse_bonds(res_data['bonds'])
            total_contacts = bonds_data['counts']['total']
            
            # Store by chain:resseq key for easy lookup
            key = f"{chain}:{resseq}"
            variant_props[key] = {
                'sasa': sasa_buried,
                'interface': is_interface,
                'vdw': vdw_energy,
                'contacts': total_contacts,
                'res_id': res_id
            }
        
        property_data[variant] = variant_props

    # Derive chain IDs that appear in the analysis results (the "selected" chains)
    analysis_chains_sbs = sorted({
        key.split(':')[0]
        for variant_props in property_data.values()
        for key in variant_props
    })

    # Generate HTML with 3Dmol.js
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">

    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        html, body {{
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #fcfdfe;
        }}

        .container {{
            height: 100%;
            background: white;
            overflow: hidden;
        }}

        .main-content {{
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
        }}

        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px 10px;
            padding: 7px 12px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            flex-shrink: 0;
        }}

        .toolbar-group {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .toolbar-sep {{
            width: 1px;
            height: 22px;
            background: #ddd;
            flex-shrink: 0;
        }}

        .toolbar-label {{
            font-size: 12px;
            font-weight: 600;
            color: #555;
            white-space: nowrap;
        }}

        .toolbar select {{
            width: auto;
            padding: 4px 8px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 13px;
            font-family: inherit;
        }}

        .toolbar button {{
            width: auto;
            padding: 4px 9px;
            border-radius: 5px;
            font-size: 12px;
            margin-top: 0;
            white-space: nowrap;
        }}

        .toolbar input[type="checkbox"] {{
            width: 14px;
            height: 14px;
            cursor: pointer;
        }}

        .viewers-container {{
            flex: 1;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            grid-auto-rows: 440px;
            gap: 4px;
            background: #e0e0e0;
            padding: 4px;
            align-content: start;
            overflow-y: auto;
            min-height: 0;
        }}

        .viewer-panel {{
            background: #000;
            position: relative;
            display: flex;
            flex-direction: column;
        }}

        .viewer-header {{
            background: #f0f0f0;
            color: #222;
            padding: 6px 12px;
            font-weight: 600;
            font-size: 13px;
            text-align: center;
            letter-spacing: 0.5px;
            flex-shrink: 0;
            border-bottom: 1px solid #ddd;
        }}

        .viewer-viewport {{
            flex: 1;
            position: relative;
            min-height: 0;
        }}

        .control-section {{
            margin-bottom: 14px;
        }}
        
        .control-label {{
            font-size: 13px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        select, button {{
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
            font-family: inherit;
        }}
        
        select:hover, select:focus {{
            border-color: #667eea;
            outline: none;
        }}
        
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            margin-top: 15px;
        }}
        
        .legend-title {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        
        .legend-gradient {{
            height: 20px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
        }}
        
        .info-badge {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            margin-top: 15px;
            line-height: 1.6;
        }}
        
        .residue-info-panel {{
            position: fixed;
            top: 8px;
            right: 8px;
            width: 300px;
            max-height: 500px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            padding: 20px;
            display: none;
            overflow-y: auto;
            z-index: 2000;
        }}
        
        .residue-info-panel.visible {{
            display: block;
        }}
        
        .residue-info-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .residue-info-title {{
            font-size: 16px;
            font-weight: 700;
            color: #333;
        }}
        
        .close-btn {{
            background: #ff6b6b;
            color: white;
            border: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
            padding: 0;
        }}
        
        .close-btn:hover {{
            background: #ff5252;
        }}
        
        .residue-info-content {{
            font-size: 13px;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .info-label {{
            font-weight: 600;
            color: #666;
        }}
        
        .info-value {{
            color: #333;
            text-align: right;
        }}
        
        .info-section {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 2px solid #f0f0f0;
        }}
        
        .info-section-title {{
            font-weight: 700;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 14px;
            text-align: center;
            z-index: 10;
        }}
        
        .loading-spinner {{
            border: 3px solid rgba(255,255,255,0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .style-buttons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .style-button {{
            padding: 8px;
            font-size: 12px;
            margin: 0;
        }}
        
        .property-info {{
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            padding: 8px;
            background: #f0f4ff;
            border-radius: 6px;
            line-height: 1.5;
        }}
        
        .sync-checkbox {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px;
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .sync-checkbox input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        
        .sync-checkbox label {{
            font-size: 13px;
            font-weight: 600;
            color: #856404;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!--div class="header">
            <h1>Side-by-Side Molecular Visualization with Synchronized Controls</h1>
        </div-->
        
        <div class="main-content">
            <div class="toolbar">
                <div class="toolbar-group">
                    <label style="display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer; padding: 3px 7px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
                        <input type="checkbox" id="sync-rotation" onchange="toggleSync()">
                        <span>Sync Rotation</span>
                    </label>
                    <label style="display: flex; align-items: center; gap: 5px; font-size: 12px; cursor: pointer;">
                        <input type="checkbox" id="analysis-only" checked onchange="toggleAnalysisOnly()">
                        <span>Analysis only</span>
                    </label>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <span class="toolbar-label">Color:</span>
                    <select id="property-select" onchange="updateAllViewers()">
                        <option value="default">By Variant (distinct colors)</option>
                        <option value="sasa">SASA (Buried Fraction)</option>
                        <option value="interface">Interface Residues</option>
                        <option value="vdw">VdW Energy</option>
                        <option value="contacts">Total Contacts</option>
                        <option value="bfactor">B-factor</option>
                        <option value="chain">Chain ID</option>
                    </select>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <span class="toolbar-label">Style:</span>
                    <select id="style-select" onchange="updateAllViewers()">
                        <option value="cartoon">Cartoon</option>
                        <option value="stick">Stick</option>
                        <option value="sphere">Sphere</option>
                        <option value="line">Line</option>
                        <option value="surface">Surface</option>
                    </select>
                    <button class="style-button" onclick="toggleSideChains()">Side Chains</button>
                    <button class="style-button" onclick="toggleChainOutlines()">Outlines</button>
                </div>
                <div class="toolbar-sep"></div>
                <div class="toolbar-group">
                    <button onclick="updateAllViewers()">Update</button>
                    <button onclick="resetAllViews()">Reset</button>
                    <button onclick="downloadAllImages()">Save</button>
                </div>
            </div>

            <div class="viewers-container" id="viewers-container">
"""
    
    # Create viewer panels for each variant
    for variant in variants:
        html += f"""
                <div class="viewer-panel">
                    <div class="viewer-header">{variant}</div>
                    <div class="viewer-viewport" id="viewport-{variant}">
                        <div class="loading" id="loading-{variant}">
                            <div class="loading-spinner"></div>
                            Loading...
                        </div>
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <!-- Legend overlay (bottom-left, fixed) -->
        <div id="legend" style="position:fixed;bottom:10px;left:10px;background:rgba(255,255,255,0.92);border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.25);padding:10px 14px;max-width:220px;z-index:500;display:none;">
            <div id="legend-content"></div>
        </div>

        <!-- Residue Information Panel (top-right, fixed) -->
        <div class="residue-info-panel" id="residue-info-panel">
            <div class="residue-info-header">
                <div class="residue-info-title" id="residue-title">Residue Info</div>
                <button class="close-btn" onclick="closeResidueInfo()">×</button>
            </div>
            <div class="residue-info-content" id="residue-info-content">
                Click on a residue to see details
            </div>
        </div>
    </div>

    <script>
"""
    
    # Embed PDB data and property data as JavaScript
    html += f"""
        // PDB data
        const pdbData = {json.dumps(pdb_contents, indent=2)};
        
        // Property data for each variant
        const propertyData = {json.dumps(property_data, indent=2)};
        
        // Variant list
        const variants = {json.dumps(variants)};

        // Chains included in the analysis (non-analysis chains can be hidden)
        const analysisChains = {json.dumps(analysis_chains_sbs)};

        // Viewer instances
        const viewers = {{}};
        let showSideChains = false;
        let showChainOutlines = false;
        let syncRotation = false;
        let showAnalysisOnly = true;
        
        // Define distinct colors for each variant
        const variantColors = [
            '#FF6B6B', // Red
            '#52B788',  // Green
            '#45B7D1', // Blue
            '#BB8FCE', // Purple
            '#F7DC6F', // Yellow
            '#4ECDC4', // Teal
            '#FFA07A', // Light Salmon
            '#98D8C8', // Mint
            '#85C1E2', // Sky Blue
            '#F8B739' // Orange
        ];
        
        // Map each variant to a color
        const variantColorMap = {{}};
        variants.forEach((variant, idx) => {{
            variantColorMap[variant] = variantColors[idx % variantColors.length];
        }});
        
        // Color scales configuration
        const colorScales = {{
            sasa: {{
                name: 'SASA Buried Fraction',
                min: 0,
                max: 1,
                colors: ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b'],
                info: 'Blue = buried (0), Red = exposed (1)',
                getLabel: (v) => v.toFixed(2)
            }},
            interface: {{
                name: 'Interface Residues',
                min: 0,
                max: 1,
                colors: ['#cccccc', '#ff6b6b'],
                info: 'Gray = non-interface, Red = interface',
                getLabel: (v) => v === 1 ? 'Interface' : 'Core'
            }},
            vdw: {{
                name: 'VdW Energy (kcal/mol)',
                min: -3,
                max: 3,
                colors: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
                info: 'Blue = favorable (negative), White = neutral (0), Red = unfavorable/clashing (positive)',
                getLabel: (v) => v.toFixed(1)
            }},
            contacts: {{
                name: 'Total Contacts',
                min: 0,
                max: 6,
                colors: ['#f0f0f0', '#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
                info: 'Light gray = no contacts, Orange/Red = many contacts',
                getLabel: (v) => Math.round(v)
            }},
            bfactor: {{
                name: 'B-factor',
                min: 5,
                max: 70,
                colors: ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020'],
                info: 'Blue = low B-factor (rigid), Red = high B-factor (flexible)',
                getLabel: (v) => Math.round(v)
            }},
            chain: {{
                name: 'Chain ID',
                categorical: true,
                colors: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#17becf', '#bcbd22'],
                info: 'Each chain colored with a distinct color.',
                getLabel: (v) => v
            }}
        }};
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', function() {{
            // Create viewer for each variant
            variants.forEach(variant => {{
                const element = document.getElementById(`viewport-${{variant}}`);
                const config = {{ backgroundColor: 'white' }};
                viewers[variant] = $3Dmol.createViewer(element, config);

                // Set up synchronized rotation if enabled
                if (syncRotation) {{
                    setupSyncedRotation(variant);
                }}
            }});

            // Delay first render so CSS grid layout computes element dimensions
            // before 3Dmol measures the canvas (avoids wrong initial zoom/size)
            requestAnimationFrame(() => requestAnimationFrame(() => updateAllViewers()));
        }});
        
        function setupSyncedRotation(sourceVariant) {{
            const sourceViewer = viewers[sourceVariant];
            
            // Listen for rotation changes
            sourceViewer.spin(false); // Ensure spin is off
            
            // Note: 3Dmol.js doesn't have built-in rotation sync events
            // We'll implement a simple sync on mouse up
            const viewportElement = document.getElementById(`viewport-${{sourceVariant}}`);
            
            viewportElement.addEventListener('mouseup', function() {{
                if (!syncRotation) return;
                
                // Get current view matrix from source
                const view = sourceViewer.getView();
                
                // Apply to all other viewers
                variants.forEach(variant => {{
                    if (variant !== sourceVariant) {{
                        viewers[variant].setView(view);
                        viewers[variant].render();
                    }}
                }});
            }});
            
            // Also sync on zoom (wheel event)
            viewportElement.addEventListener('wheel', function() {{
                if (!syncRotation) return;
                
                setTimeout(() => {{
                    const view = sourceViewer.getView();
                    variants.forEach(variant => {{
                        if (variant !== sourceVariant) {{
                            viewers[variant].setView(view);
                            viewers[variant].render();
                        }}
                    }});
                }}, 50);
            }});
        }}
        
        function toggleSync() {{
            syncRotation = document.getElementById('sync-rotation').checked;

            if (syncRotation) {{
                // Re-setup sync for all viewers
                variants.forEach(variant => setupSyncedRotation(variant));
            }}
        }}

        function toggleAnalysisOnly() {{
            showAnalysisOnly = document.getElementById('analysis-only').checked;
            updateAllViewers();
        }}

        function getColor(property, value, variant) {{
            if (property === 'default' || !property) {{
                return null;
            }}
            
            const scale = colorScales[property];
            if (!scale) return null;
            
            const normalized = Math.max(0, Math.min(1, (value - scale.min) / (scale.max - scale.min)));
            
            const colors = scale.colors;
            const idx = normalized * (colors.length - 1);
            const lower = Math.floor(idx);
            const upper = Math.ceil(idx);
            const frac = idx - lower;

            if (lower === upper) {{
                return colors[lower];
            }}

            // Interpolate between adjacent colors
            const hex2rgb = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
            const rgb2hex = (r,g,b) => '#' + [r,g,b].map(v => Math.round(v).toString(16).padStart(2,'0')).join('');
            const c1 = hex2rgb(colors[lower]), c2 = hex2rgb(colors[upper]);
            return rgb2hex(c1[0]+(c2[0]-c1[0])*frac, c1[1]+(c2[1]-c1[1])*frac, c1[2]+(c2[2]-c1[2])*frac);
        }}
        
        function applyVisualizationStyle(variant, property, style) {{
            const viewer = viewers[variant];
            if (!viewer) return;
            
            let styleSpec = {{}};
            const variantColor = variantColorMap[variant];
            const props = propertyData[variant];
            
            // Base style
            if (property === 'default') {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ color: variantColor, clickable: true }};
                }} else if (style === 'stick') {{
                    styleSpec.stick = {{ radius: 0.15, color: variantColor, clickable: true }};
                }} else if (style === 'sphere') {{
                    styleSpec.sphere = {{ radius: 1.0, color: variantColor, clickable: true }};
                }} else if (style === 'line') {{
                    styleSpec.line = {{ color: variantColor, clickable: true }};
                }} else if (style === 'surface') {{
                    styleSpec.surface = {{ opacity: 0.8, color: variantColor, clickable: true }};
                }}
            }} else if (property === 'bfactor') {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ 
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }},
                        clickable: true
                    }};
                }} else {{
                    styleSpec[style] = {{ 
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }},
                        clickable: true
                    }};
                }}
            }} else {{
                if (style === 'cartoon') {{
                    styleSpec.cartoon = {{ clickable: true }};
                }} else if (style === 'stick') {{
                    styleSpec.stick = {{ radius: 0.15, clickable: true }};
                }} else if (style === 'sphere') {{
                    styleSpec.sphere = {{ radius: 1.0, clickable: true }};
                }} else if (style === 'line') {{
                    styleSpec.line = {{ clickable: true }};
                }} else if (style === 'surface') {{
                    styleSpec.surface = {{ opacity: 0.8, clickable: true }};
                }}
            }}
            
            // Apply base style to all
            viewer.setStyle({{}}, styleSpec);
            
            // For surface representation, need to explicitly add surface
            if (style === 'surface') {{
                // Remove any existing surfaces first
                viewer.removeAllSurfaces();
                
                if (property === 'default') {{
                    // Add uniform colored surface
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        color: variantColor
                    }});
                }} else if (property === 'bfactor') {{
                    // Add surface colored by B-factor
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        colorscheme: {{ prop: 'b', gradient: 'roygb', min: 0, max: 100 }}
                    }});
                }} else {{
                    // For other properties, use variant color
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: 0.8,
                        color: variantColor
                    }});
                }}
            }}
            
            // Apply property coloring if not default (works for non-surface styles)
            // Chain coloring: color each chain with a distinct categorical color
            if (property === 'chain') {{
                const chainPalette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#17becf', '#bcbd22'];
                const modelData = viewer.getModel(0);
                if (modelData) {{
                    const atoms = modelData.selectedAtoms({{}});
                    const chains = [...new Set(atoms.map(a => a.chain))].sort();
                    chains.forEach((chainId, chainIdx) => {{
                        const chainColor = chainPalette[chainIdx % chainPalette.length];
                        const chainStyleSpec = {{}};
                        if (style === 'cartoon') chainStyleSpec.cartoon = {{ color: chainColor, clickable: true }};
                        else if (style === 'stick') chainStyleSpec.stick = {{ radius: 0.15, color: chainColor, clickable: true }};
                        else if (style === 'sphere') chainStyleSpec.sphere = {{ radius: 1.0, color: chainColor, clickable: true }};
                        else if (style === 'line') chainStyleSpec.line = {{ color: chainColor, clickable: true }};
                        else chainStyleSpec[style] = {{ color: chainColor, clickable: true }};
                        viewer.setStyle({{ chain: chainId }}, chainStyleSpec);
                    }});
                }}
            }}

            if (property !== 'default' && property !== 'bfactor' && property !== 'chain' && style !== 'surface' && props) {{
                Object.keys(props).forEach(key => {{
                    const [chain, resseq] = key.split(':');
                    const value = props[key][property];

                    if (value !== undefined) {{
                        const color = getColor(property, value, variant);
                        if (color) {{
                            const sel = {{ chain: chain, resi: parseInt(resseq) }};
                            const coloredStyle = {{ ...styleSpec }};
                            coloredStyle[style] = {{ ...styleSpec[style], color: color, clickable: true }};
                            viewer.setStyle(sel, coloredStyle);
                        }}
                    }}
                }});
            }}

            // Add side chains if requested
            if (showSideChains && property === 'default') {{
                viewer.setStyle({{ atom: ['CA', 'C', 'N', 'O'] }}, {{ stick: {{ radius: 0.2, clickable: true }} }}, {{ byres: false }});
            }}
            
            // Add chain outlines if requested
            if (showChainOutlines) {{
                const modelData = viewer.getModel(0);
                if (modelData) {{
                    const atoms = modelData.selectedAtoms({{}});
                    const chains = [...new Set(atoms.map(a => a.chain))];
                    
                    const chainOutlineColors = [
                        '#8B0000', '#006064', '#1A237E', '#E65100', '#004D40',
                        '#F57F17', '#4A148C', '#01579B', '#BF360C', '#1B5E20'
                    ];
                    
                    chains.forEach((chain, chainIdx) => {{
                        const outlineColor = chainOutlineColors[chainIdx % chainOutlineColors.length];
                        viewer.setStyle(
                            {{ chain: chain }},
                            {{ cartoon: {{ color: outlineColor, thickness: 0.4, opacity: 0.7 }} }},
                            {{ keepStyle: true }}
                        );
                    }});
                }}
            }}

            // Hide non-analysis chains if requested (must run last to override all other styles)
            if (showAnalysisOnly && analysisChains.length > 0) {{
                const modelData = viewer.getModel(0);
                if (modelData) {{
                    const atoms = modelData.selectedAtoms({{}});
                    const allChains = [...new Set(atoms.map(a => a.chain))];
                    allChains.forEach(chain => {{
                        if (!analysisChains.includes(chain)) {{
                            viewer.setStyle({{ chain: chain }}, {{}});
                        }}
                    }});
                }}
            }}
        }}

        function updateAllViewers() {{
            const property = document.getElementById('property-select').value;
            const style = document.getElementById('style-select').value;
            
            variants.forEach(variant => {{
                const viewer = viewers[variant];
                if (!viewer) return;
                
                // Show loading
                const loading = document.getElementById(`loading-${{variant}}`);
                if (loading) loading.style.display = 'block';
                
                // Clear and reload
                viewer.clear();
                
                if (!pdbData[variant]) {{
                    console.warn(`No PDB data for ${{variant}}`);
                    if (loading) loading.style.display = 'none';
                    return;
                }}
                
                // Add model
                viewer.addModel(pdbData[variant], 'pdb');
                
                // Apply styling
                applyVisualizationStyle(variant, property, style);
                
                // Set up click handler
                viewer.setClickable({{}}, true, function(atom, viewer, event, container) {{
                    if (atom) {{
                        showResidueInfo(atom, variant);
                        event.stopPropagation();
                    }}
                }});
                
                viewer.resize();
                const zoomSel = (showAnalysisOnly && analysisChains.length > 0)
                    ? {{ chain: analysisChains }} : {{}};
                viewer.zoomTo(zoomSel);
                viewer.render();

                // Hide loading
                if (loading) loading.style.display = 'none';
            }});
            
            // Update legend and property info
            updateLegend(property);
            updatePropertyInfo(property);
        }}
        
        function updateLegend(property) {{
            const legendContent = document.getElementById('legend-content');
            const legendEl = document.getElementById('legend');
            if (legendEl) legendEl.style.display = 'block';

            if (property === 'default') {{
                let html = '<div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">Variant Colors</div>';
                variants.forEach(variant => {{
                    const color = variantColorMap[variant];
                    html += `
                        <div style="display: flex; align-items: center; margin: 6px 0;">
                            <div style="width: 24px; height: 16px; background: ${{color}}; border-radius: 3px; margin-right: 8px; border: 1px solid #ccc;"></div>
                            <span style="font-size: 12px;">${{variant}}</span>
                        </div>
                    `;
                }});
                legendContent.innerHTML = html;
                return;
            }}
            
            const scale = colorScales[property];
            if (!scale) return;

            if (scale.categorical) {{
                // Chain: show dynamic per-chain swatches from propertyData keys
                const chainPalette = scale.colors;
                const firstVariant = variants[0];
                const props = propertyData[firstVariant];
                const allChains = props ? [...new Set(Object.keys(props).map(k => k.split(':')[0]))].sort() : [];
                let html = `<div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">${{scale.name}}</div>`;
                allChains.forEach((chainId, idx) => {{
                    const color = chainPalette[idx % chainPalette.length];
                    html += `<div style="display:flex;align-items:center;margin:4px 0;"><div style="width:20px;height:14px;background:${{color}};border-radius:3px;margin-right:8px;border:1px solid #ccc;flex-shrink:0;"></div><span style="font-size:12px;">Chain ${{chainId}}</span></div>`;
                }});
                html += `<div style="font-size:11px;color:#666;margin-top:8px;">${{scale.info}}</div>`;
                legendContent.innerHTML = html;
                return;
            }}

            const gradient = scale.colors.join(', ');

            legendContent.innerHTML = `
                <div style="font-size: 11px; font-weight: 600; margin-bottom: 8px;">${{scale.name}}</div>
                <div class="legend-gradient" style="background: linear-gradient(to right, ${{gradient}});"></div>
                <div class="legend-labels">
                    <span>${{scale.getLabel(scale.min)}}</span>
                    <span>${{scale.getLabel(scale.max)}}</span>
                </div>
                <div style="font-size: 11px; color: #666; margin-top: 8px;">${{scale.info}}</div>
            `;
        }}

        function updatePropertyInfo(property) {{
            const infoDiv = document.getElementById('property-info');
            if (!infoDiv) return;

            const descriptions = {{
                'default': 'Each variant is colored with a distinct color for easy identification.',
                'sasa': 'Solvent Accessible Surface Area - measures how buried each residue is.',
                'interface': 'Highlights residues at protein-protein interfaces.',
                'vdw': 'Van der Waals energy - negative values indicate favorable interactions.',
                'contacts': 'Number of atomic contacts each residue makes.',
                'bfactor': 'Temperature/B-factor - indicates structural flexibility or disorder.',
                'chain': 'Colors each chain with a distinct color for easy identification of chain boundaries.'
            }};

            infoDiv.textContent = descriptions[property] || '';
        }}
        
        function toggleSideChains() {{
            showSideChains = !showSideChains;
            updateAllViewers();
        }}
        
        function toggleChainOutlines() {{
            showChainOutlines = !showChainOutlines;
            updateAllViewers();
        }}
        
        function resetAllViews() {{
            const zoomSel = (showAnalysisOnly && analysisChains.length > 0)
                ? {{ chain: analysisChains }} : {{}};
            variants.forEach(variant => {{
                const viewer = viewers[variant];
                if (viewer) {{
                    viewer.resize();
                    viewer.zoomTo(zoomSel);
                    viewer.render();
                }}
            }});
        }}
        
        function downloadAllImages() {{
            variants.forEach(variant => {{
                const viewer = viewers[variant];
                if (viewer) {{
                    viewer.render();
                    const imgData = viewer.pngURI();
                    const link = document.createElement('a');
                    link.href = imgData;
                    link.download = `${{variant}}_structure.png`;
                    link.click();
                }}
            }});
        }}
        
        function showResidueInfo(atom, variant) {{
            const panel = document.getElementById('residue-info-panel');
            const title = document.getElementById('residue-title');
            const content = document.getElementById('residue-info-content');
            
            const resname = atom.resn || 'Unknown';
            const chain = atom.chain || '?';
            const resi = atom.resi || '?';
            const atomName = atom.atom || '?';
            const elem = atom.elem || '?';
            const bfactor = atom.b !== undefined ? atom.b.toFixed(2) : 'N/A';
            
            const resKey = `${{chain}}:${{resi}}`;
            const variantProps = propertyData[variant];
            const resProps = variantProps ? variantProps[resKey] : null;
            
            title.textContent = `${{resname}} ${{chain}}:${{resi}} (${{variant}})`;
            
            let html = `
                <div class="info-row">
                    <span class="info-label">Variant:</span>
                    <span class="info-value" style="color: ${{variantColorMap[variant]}}; font-weight: 700;">${{variant}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Residue Type:</span>
                    <span class="info-value">${{resname}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Chain:</span>
                    <span class="info-value">${{chain}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Position:</span>
                    <span class="info-value">${{resi}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Atom Clicked:</span>
                    <span class="info-value">${{atomName}} (${{elem}})</span>
                </div>
                <div class="info-row">
                    <span class="info-label">B-factor:</span>
                    <span class="info-value">${{bfactor}}</span>
                </div>
            `;
            
            if (resProps) {{
                html += `
                    <div class="info-section">
                        <div class="info-section-title">Analysis Properties</div>
                        
                        <div class="info-row">
                            <span class="info-label">SASA (Buried):</span>
                            <span class="info-value">${{resProps.sasa.toFixed(2)}}</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">Interface:</span>
                            <span class="info-value">${{resProps.interface === 1 ? 'Yes' : 'No'}}</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">VdW Energy:</span>
                            <span class="info-value">${{resProps.vdw.toFixed(2)}} kcal/mol</span>
                        </div>
                        
                        <div class="info-row">
                            <span class="info-label">Total Contacts:</span>
                            <span class="info-value">${{resProps.contacts}}</span>
                        </div>
                    </div>
                `;
            }} else {{
                html += `
                    <div class="info-section">
                        <div class="info-section-title">Analysis Properties</div>
                        <div style="color: #999; font-style: italic; font-size: 12px;">
                            No analysis data available
                        </div>
                    </div>
                `;
            }}
            
            content.innerHTML = html;
            panel.classList.add('visible');
        }}
        
        function closeResidueInfo() {{
            const panel = document.getElementById('residue-info-panel');
            panel.classList.remove('visible');
        }}
    </script>
</body>
</html>
"""
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Saved side-by-side 3D visualization to {output_file}")
    
    return html