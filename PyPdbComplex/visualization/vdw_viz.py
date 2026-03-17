from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import os

from ..models import Complex, Residue
from ..vdw import (
    per_residue_LJ_decomposition,
    per_residue_pair_LJ,
    identify_energetic_hotspots,
    compare_energetic_contributions,
)


# ---------------------------
# Color Mapping for Energy Visualization
# ---------------------------
def energy_to_rgb(energy: float, vmin: float = -5.0, vmax: float = 1.0) -> Tuple[int, int, int]:
    """
    Map energy value to RGB tuple.
    
    Blue (favorable) -> White (neutral) -> Red (unfavorable)
    
    Args:
        energy: VDW energy value in kcal/mol
        vmin: Minimum energy for color scale
        vmax: Maximum energy for color scale
    
    Returns:
        RGB tuple (r, g, b) with values 0-255
    """
    # Normalize energy to [0, 1]
    if energy <= vmin:
        norm = 0.0
    elif energy >= vmax:
        norm = 1.0
    else:
        norm = (energy - vmin) / (vmax - vmin)
    
    # Blue -> White -> Red gradient
    if norm < 0.5:
        # Blue to White
        t = norm * 2.0
        r = int(0 + t * 255)
        g = int(0 + t * 255)
        b = 255
    else:
        # White to Red
        t = (norm - 0.5) * 2.0
        r = 255
        g = int(255 - t * 255)
        b = int(255 - t * 255)
    
    return (r, g, b)


def energy_to_hex(energy: float, vmin: float = -5.0, vmax: float = 1.0) -> str:
    """Convert energy to hex color string."""
    r, g, b = energy_to_rgb(energy, vmin, vmax)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------
# 3D Molecular Visualization with 3Dmol.js
# ---------------------------
def generate_3dmol_viewer(
    pdb_content: str,
    per_res_energy_left: Dict[str, float],
    per_res_energy_right: Dict[str, float],
    viewer_id: str = "vdw_viewer",
    width: int = 800,
    height: int = 600,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    energy_threshold: float = -2.0,
) -> str:
    """
    Generate HTML div with 3Dmol.js viewer showing VDW energies for both groups.

    Args:
        pdb_content: PDB file content as string
        per_res_energy_left:  residue_id -> energy for Group A (left selection)
        per_res_energy_right: residue_id -> energy for Group B (right selection)
        viewer_id: HTML element ID for the viewer
        width: Viewer width in pixels
        height: Viewer height in pixels
        vmin: Minimum energy for colour scale (auto-detected if None)
        vmax: Maximum energy for colour scale (auto-detected if None)
        energy_threshold: Hotspot cut-off (kcal/mol); used in info panel badge

    Returns:
        HTML string with embedded 3Dmol.js viewer and side info panel
    """
    all_energies_combined = list(per_res_energy_left.values()) + list(per_res_energy_right.values())
    if not all_energies_combined:
        return "<p>No energy data to visualize</p>"

    if vmin is None:
        vmin = min(all_energies_combined)
    if vmax is None:
        vmax = max(all_energies_combined)

    def _parse_res(res_id: str, group: str, energy: float) -> Optional[dict]:
        parts = res_id.split()
        if len(parts) < 2:
            return None
        resname = parts[0]
        chain_res = parts[1]
        if '_' in chain_res:
            chain, resnum = chain_res.split('_', 1)
        elif ':' in chain_res:
            chain, resnum = chain_res.split(':', 1)
        else:
            return None
        return {
            'color':   energy_to_hex(energy, vmin, vmax),
            'energy':  round(energy, 3),
            'resname': resname,
            'chain':   chain,
            'resnum':  resnum,
            'group':   group,
            'hotspot': energy <= energy_threshold,
        }

    # Build per-key residue data; right side first so left overwrites on collision
    residue_data: dict = {}
    for res_id, energy in per_res_energy_right.items():
        d = _parse_res(res_id, 'B', energy)
        if d:
            residue_data[f"{d['chain']}:{d['resnum']}"] = d
    for res_id, energy in per_res_energy_left.items():
        d = _parse_res(res_id, 'A', energy)
        if d:
            residue_data[f"{d['chain']}:{d['resnum']}"] = d

    pdb_escaped   = json.dumps(pdb_content)
    res_data_json = json.dumps(residue_data)

    html = f"""
<div style="display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap;">
  <!-- 3D Viewer column -->
  <div style="flex:0 0 auto;">
    <!-- Outer border wrapper — keep overflow visible so 3Dmol canvas is never clipped -->
    <div style="border:1px solid #e0e0e0; border-radius:8px; display:inline-block;">
      <div id="{viewer_id}" style="width:{width}px; height:{height}px; position:relative;"></div>
    </div>
    <!-- Colour-scale legend -->
    <div style="width:{width}px; margin:8px 0 0 0; font-size:12px; color:#555;">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="white-space:nowrap;">Favorable<br/>{vmin:.2f} kcal/mol</span>
        <div style="flex:1; height:14px; background:linear-gradient(to right,#0000ff,#ffffff,#ff0000); border:1px solid #ccc; border-radius:3px;"></div>
        <span style="white-space:nowrap;">Unfavorable<br/>{vmax:.2f} kcal/mol</span>
      </div>
      <div style="margin-top:6px; color:#888; font-size:11px; text-align:center;">
        Click a coloured residue for details &nbsp;|&nbsp; Click empty space to clear
      </div>
    </div>
  </div>

  <!-- Info panel column -->
  <div id="{viewer_id}_panel"
       style="flex:0 0 260px; min-height:180px; background:#f8f9fa; border:1px solid #e0e0e0;
              border-radius:8px; padding:16px; font-size:13px; color:#444;">
    <div style="font-weight:600; margin-bottom:8px; color:#667eea;">Residue Details</div>
    <div style="color:#aaa; font-style:italic;">Click a coloured residue in the 3D viewer to see its VDW energy details.</div>
  </div>
</div>

<script>
(function initVdwViewer() {{
    if (typeof $3Dmol === 'undefined' || typeof $ === 'undefined') {{
        setTimeout(initVdwViewer, 100);
        return;
    }}
    $(document).ready(function() {{
        var viewer    = $3Dmol.createViewer("{viewer_id}", {{backgroundColor: 'white'}});
        var pdbData   = {pdb_escaped};
        var resData   = {res_data_json};
        var threshold = {energy_threshold};

        viewer.addModel(pdbData, "pdb");

        // All atoms: gray cartoon backbone
        viewer.setStyle({{}}, {{cartoon: {{color: '#cccccc'}}}});

        // Interface residues: colored cartoon only
        for (var resKey in resData) {{
            var d     = resData[resKey];
            var resiN = parseInt(d.resnum, 10) || d.resnum;
            viewer.setStyle({{chain: d.chain, resi: resiN}}, {{cartoon: {{color: d.color}}}});
        }}

        // Lookup by both string and integer resi (3Dmol returns atom.resi as integer)
        var byKey = {{}};
        for (var k in resData) {{
            var d = resData[k];
            byKey[d.chain + ':' + d.resnum]               = d;
            byKey[d.chain + ':' + parseInt(d.resnum, 10)] = d;
        }}

        var resetHtml = '<div style="font-weight:600;margin-bottom:8px;color:#667eea;">Residue Details</div>'
            + '<div style="color:#aaa;font-style:italic;">Click a coloured residue to see details.</div>';

        // setClickable BEFORE render — 3Dmol builds the pick buffer during render()
        viewer.setClickable({{}}, true, function(atom, viewer) {{
            var panel = document.getElementById('{viewer_id}_panel');
            if (!atom || !atom.chain) {{ panel.innerHTML = resetHtml; return; }}
            var d = byKey[atom.chain + ':' + String(atom.resi)];
            if (!d) {{ panel.innerHTML = resetHtml; return; }}

            var hotBadge   = d.hotspot ? '<span style="background:#e74c3c;color:white;padding:2px 7px;border-radius:10px;font-size:11px;margin-left:6px;">Hotspot</span>' : '';
            var groupBadge = d.group === 'A'
                ? '<span style="background:#3498db;color:white;padding:2px 7px;border-radius:10px;font-size:11px;">Group A</span>'
                : '<span style="background:#e67e22;color:white;padding:2px 7px;border-radius:10px;font-size:11px;">Group B</span>';
            var barPct   = Math.max(0, Math.min(100, ((d.energy - threshold) / (0 - threshold)) * 100));
            var barColor = d.energy <= threshold ? '#3498db' : (d.energy < 0 ? '#f39c12' : '#e74c3c');

            panel.innerHTML =
                '<div style="font-weight:600;margin-bottom:10px;color:#667eea;">Residue Details</div>'
                + '<div style="margin-bottom:8px;">' + groupBadge + hotBadge + '</div>'
                + '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                + '<tr><td style="padding:4px 0;color:#888;">Residue</td><td style="padding:4px 0;font-weight:600;">' + d.resname + '</td></tr>'
                + '<tr><td style="padding:4px 0;color:#888;">Chain</td><td style="padding:4px 0;">' + d.chain + '</td></tr>'
                + '<tr><td style="padding:4px 0;color:#888;">Residue #</td><td style="padding:4px 0;">' + d.resnum + '</td></tr>'
                + '<tr><td style="padding:4px 0;color:#888;">VDW Energy</td><td style="padding:4px 0;font-weight:600;color:' + barColor + ';">' + d.energy.toFixed(3) + ' kcal/mol</td></tr>'
                + '</table>';
        }});

        viewer.zoomTo();
        viewer.render();
    }});
}})();
</script>
"""
    return html


# ---------------------------
# Interactive Plotly Charts
# ---------------------------
def create_plotly_distribution(
    per_res_energy: Dict[str, float],
    title: str = "VDW Energy Distribution",
) -> str:
    """
    Create interactive Plotly histogram of energy distribution.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        title: Chart title
    
    Returns:
        HTML string with embedded Plotly chart
    """
    energies = list(per_res_energy.values())
    
    data = [{
        'x': energies,
        'type': 'histogram',
        'nbinsx': 50,
        'marker': {
            'color': energies,
            'colorscale': [[0, 'blue'], [0.5, 'white'], [1, 'red']],
            'colorbar': {
                'title': 'Energy<br>(kcal/mol)',
                'thickness': 15,
                'len': 0.7
            },
            'line': {'color': 'black', 'width': 1}
        },
        'name': 'Residues'
    }]
    
    layout = {
        'title': {
            'text': title,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        'xaxis': {
            'title': 'VDW Energy (kcal/mol)',
            'gridcolor': '#e0e0e0'
        },
        'yaxis': {
            'title': 'Number of Residues',
            'gridcolor': '#e0e0e0'
        },
        'plot_bgcolor': 'white',
        'hovermode': 'closest',
        'showlegend': False
    }
    
    # Add statistics annotation
    import statistics
    mean_e = statistics.mean(energies)
    median_e = statistics.median(energies)
    std_e = statistics.stdev(energies) if len(energies) > 1 else 0
    
    annotation = {
        'text': f'Total: {len(energies)} residues<br>' +
                f'Mean: {mean_e:.2f} kcal/mol<br>' +
                f'Median: {median_e:.2f} kcal/mol<br>' +
                f'Std: {std_e:.2f} kcal/mol',
        'xref': 'paper',
        'yref': 'paper',
        'x': 0.02,
        'y': 0.98,
        'xanchor': 'left',
        'yanchor': 'top',
        'showarrow': False,
        'bgcolor': 'wheat',
        'bordercolor': 'black',
        'borderwidth': 1,
        'font': {'size': 11}
    }
    layout['annotations'] = [annotation]
    
    config = {
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False
    }
    
    plot_json = {
        'data': data,
        'layout': layout,
        'config': config
    }
    
    div_id = "plotly_distribution_" + str(hash(title))[:8]
    
    html = f"""
<div id="{div_id}" style="width: 100%; height: 500px;"></div>
<script>
    Plotly.newPlot('{div_id}', {json.dumps(plot_json['data'])}, 
                   {json.dumps(plot_json['layout'])}, 
                   {json.dumps(plot_json['config'])});
</script>
"""
    
    return html


def create_plotly_chain_comparison(
    per_res_energy: Dict[str, float],
    title: str = "VDW Energy by Chain",
) -> str:
    """
    Create interactive Plotly bar chart comparing chains.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        title: Chart title
    
    Returns:
        HTML string with embedded Plotly chart
    """
    summary = compare_energetic_contributions(per_res_energy)
    
    if not summary:
        return "<p>No chain data available</p>"
    
    chains = sorted(summary.keys())
    
    # Prepare data for grouped bar chart
    data = [
        {
            'x': chains,
            'y': [summary[c]['total_energy'] for c in chains],
            'name': 'Total Energy',
            'type': 'bar',
            'marker': {'color': 'steelblue'}
        },
        {
            'x': chains,
            'y': [summary[c]['favorable_energy'] for c in chains],
            'name': 'Favorable (< 0)',
            'type': 'bar',
            'marker': {'color': 'green'}
        },
        {
            'x': chains,
            'y': [summary[c]['unfavorable_energy'] for c in chains],
            'name': 'Unfavorable (≥ 0)',
            'type': 'bar',
            'marker': {'color': 'red'}
        }
    ]
    
    layout = {
        'title': {
            'text': title,
            'font': {'size': 18}
        },
        'xaxis': {'title': 'Chain'},
        'yaxis': {'title': 'VDW Energy (kcal/mol)'},
        'barmode': 'group',
        'plot_bgcolor': 'white',
        'hovermode': 'closest',
        'showlegend': True,
        'legend': {'x': 1.02, 'y': 1}
    }
    
    config = {'responsive': True, 'displaylogo': False}
    
    div_id = "plotly_chains_" + str(hash(title))[:8]
    
    html = f"""
<div id="{div_id}" style="width: 100%; height: 500px;"></div>
<script>
    Plotly.newPlot('{div_id}', {json.dumps(data)}, 
                   {json.dumps(layout)}, 
                   {json.dumps(config)});
</script>
"""
    
    return html


def create_plotly_hotspot_ranking(
    per_res_energy: Dict[str, float],
    energy_threshold: float = -2.0,
    top_n: int = 20,
    title: str = "Top Energetic Hotspots",
) -> str:
    """
    Create interactive Plotly horizontal bar chart of hotspots.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        top_n: Number of top hotspots to display
        title: Chart title
    
    Returns:
        HTML string with embedded Plotly chart
    """
    hotspots = identify_energetic_hotspots(per_res_energy, energy_threshold, top_n=top_n)
    
    if not hotspots:
        return "<p>No hotspots found</p>"
    
    res_ids, energies = zip(*hotspots)
    
    # Shorten residue IDs for display
    display_ids = [rid.split()[1] if ' ' in rid else rid for rid in res_ids]
    
    data = [{
        'y': display_ids[::-1],  # Reverse to show strongest at top
        'x': list(energies)[::-1],
        'type': 'bar',
        'orientation': 'h',
        'marker': {
            'color': list(energies)[::-1],
            'colorscale': [[0, 'darkblue'], [0.5, 'blue'], [1, 'lightblue']],
            'colorbar': {
                'title': 'Energy<br>(kcal/mol)',
                'thickness': 15,
                'len': 0.7
            },
            'line': {'color': 'black', 'width': 1}
        },
        'text': [f'{e:.2f}' for e in energies][::-1],
        'textposition': 'inside',
        'textfont': {'color': 'white', 'size': 10},
        'hovertemplate': '<b>%{y}</b><br>Energy: %{x:.2f} kcal/mol<extra></extra>'
    }]
    
    layout = {
        'title': {
            'text': title,
            'font': {'size': 18}
        },
        'xaxis': {
            'title': 'VDW Energy (kcal/mol)',
            'gridcolor': '#e0e0e0'
        },
        'yaxis': {
            'title': '',
            'tickfont': {'size': 10}
        },
        'plot_bgcolor': 'white',
        'height': max(400, len(hotspots) * 50),
        'margin': {'l': 100, 'r': 100, 't': 80, 'b': 60},
        'showlegend': False
    }
    
    config = {'responsive': True, 'displaylogo': False}
    
    div_id = "plotly_hotspots_" + str(hash(title))[:8]
    
    html = f"""
<div id="{div_id}" style="width: 100%; height: {layout['height']}px;"></div>
<script>
    Plotly.newPlot('{div_id}', {json.dumps(data)}, 
                   {json.dumps(layout)}, 
                   {json.dumps(config)});
</script>
"""
    
    return html


def create_plotly_residue_profile(
    per_res_energy: Dict[str, float],
    chain_id: str,
    title: Optional[str] = None,
    highlight_threshold: float = -2.0,
) -> str:
    """
    Create interactive Plotly line chart of energy along residue sequence.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        chain_id: Chain ID to plot
        title: Chart title
        highlight_threshold: Threshold for highlighting hotspots
    
    Returns:
        HTML string with embedded Plotly chart
    """
    # Filter residues for this chain
    chain_data = []
    for res_id, energy in per_res_energy.items():
        parts = res_id.split()
        if len(parts) >= 2:
            chain_res = parts[1]
            if '_' in chain_res:
                chain, resnum = chain_res.split('_')
            elif ':' in chain_res:
                chain, resnum = chain_res.split(':')
            else:
                continue
            
            if chain == chain_id:
                try:
                    resnum_int = int(resnum)
                    chain_data.append((resnum_int, energy, res_id))
                except ValueError:
                    continue
    
    if not chain_data:
        return f"<p>No data found for chain {chain_id}</p>"
    
    # Sort by residue number
    chain_data.sort(key=lambda x: x[0])
    resnums, energies, res_ids = zip(*chain_data)
    
    # Main energy trace
    data = [
        {
            'x': list(resnums),
            'y': list(energies),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'VDW Energy',
            'line': {'color': 'steelblue', 'width': 2},
            'marker': {'size': 6, 'color': 'steelblue'},
            'hovertemplate': '<b>Residue %{x}</b><br>Energy: %{y:.2f} kcal/mol<extra></extra>'
        }
    ]
    
    # Add hotspot markers
    hotspot_x = [r for r, e in zip(resnums, energies) if e <= highlight_threshold]
    hotspot_y = [e for e in energies if e <= highlight_threshold]
    
    if hotspot_x:
        data.append({
            'x': hotspot_x,
            'y': hotspot_y,
            'type': 'scatter',
            'mode': 'markers',
            'name': f'Hotspots (E ≤ {highlight_threshold})',
            'marker': {
                'size': 12,
                'color': 'red',
                'symbol': 'star',
                'line': {'color': 'darkred', 'width': 1}
            },
            'hovertemplate': '<b>Hotspot: Res %{x}</b><br>Energy: %{y:.2f} kcal/mol<extra></extra>'
        })
    
    if title is None:
        title = f"VDW Energy Profile - Chain {chain_id}"
    
    layout = {
        'title': {
            'text': title,
            'font': {'size': 18}
        },
        'xaxis': {
            'title': 'Residue Number',
            'gridcolor': '#e0e0e0'
        },
        'yaxis': {
            'title': 'VDW Energy (kcal/mol)',
            'gridcolor': '#e0e0e0',
            'zeroline': True,
            'zerolinecolor': 'black',
            'zerolinewidth': 2
        },
        'plot_bgcolor': 'white',
        'hovermode': 'closest',
        'showlegend': True,
        'shapes': [
            {
                'type': 'line',
                'x0': min(resnums),
                'x1': max(resnums),
                'y0': highlight_threshold,
                'y1': highlight_threshold,
                'line': {
                    'color': 'red',
                    'dash': 'dash',
                    'width': 2
                }
            }
        ]
    }
    
    # Add statistics annotation
    import statistics
    mean_e = statistics.mean(energies)
    
    annotation = {
        'text': f'Chain {chain_id}<br>' +
                f'Residues: {len(energies)}<br>' +
                f'Mean: {mean_e:.2f}<br>' +
                f'Hotspots: {len(hotspot_x)}',
        'xref': 'paper',
        'yref': 'paper',
        'x': 0.02,
        'y': 0.98,
        'xanchor': 'left',
        'yanchor': 'top',
        'showarrow': False,
        'bgcolor': 'wheat',
        'bordercolor': 'black',
        'borderwidth': 1
    }
    layout['annotations'] = [annotation]
    
    config = {'responsive': True, 'displaylogo': False}
    
    div_id = f"plotly_profile_{chain_id}_" + str(hash(title or ""))[:8]
    
    html = f"""
<div id="{div_id}" style="width: 100%; height: 500px;"></div>
<script>
    Plotly.newPlot('{div_id}', {json.dumps(data)}, 
                   {json.dumps(layout)}, 
                   {json.dumps(config)});
</script>
"""
    
    return html


def create_plotly_contact_matrix(
    cx: Complex,
    chain_exprs: List[str],
    cutoff: float = 5.0,
    title: str = "VDW Contact Matrix",
) -> str:
    """
    Create interactive Plotly heatmap of chain-chain VDW energies.
    
    Args:
        cx: Complex object
        chain_exprs: List of chain selection expressions
        cutoff: Distance cutoff for contacts
        title: Chart title
    
    Returns:
        HTML string with embedded Plotly heatmap
    """
    # Calculate pairwise energies
    pair_energies = per_residue_pair_LJ(cx, chain_exprs, cutoff=cutoff)
    
    # Build matrix
    n = len(chain_exprs)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i, chain1 in enumerate(chain_exprs):
        for j, chain2 in enumerate(chain_exprs):
            if i == j:
                continue
            key = (chain1, chain2)
            if key in pair_energies:
                total_energy = sum(pair_energies[key].values())
                matrix[i][j] = total_energy
    
    # Create annotations for cell values
    annotations = []
    for i in range(n):
        for j in range(n):
            if i != j:
                annotations.append({
                    'x': j,
                    'y': i,
                    'text': f'{matrix[i][j]:.1f}',
                    'showarrow': False,
                    'font': {'size': 12, 'color': 'black'}
                })
    
    data = [{
        'z': matrix,
        'x': chain_exprs,
        'y': chain_exprs,
        'type': 'heatmap',
        'colorscale': [[0, 'blue'], [0.5, 'white'], [1, 'red']],
        'colorbar': {
            'title': 'Total VDW<br>Energy<br>(kcal/mol)',
            'thickness': 20,
            'len': 0.7
        },
        'hovertemplate': '<b>%{y} → %{x}</b><br>Energy: %{z:.2f} kcal/mol<extra></extra>'
    }]
    
    layout = {
        'title': {
            'text': title,
            'font': {'size': 18}
        },
        'xaxis': {
            'title': 'Chain',
            'side': 'bottom'
        },
        'yaxis': {
            'title': 'Chain',
            'autorange': 'reversed'
        },
        'annotations': annotations,
        'width': 600,
        'height': 600
    }
    
    config = {'responsive': True, 'displaylogo': False}
    
    div_id = "plotly_matrix_" + str(hash(title))[:8]
    
    html = f"""
<div id="{div_id}" style="width: 600px; height: 600px; margin: 20px auto;"></div>
<script>
    Plotly.newPlot('{div_id}', {json.dumps(data)}, 
                   {json.dumps(layout)}, 
                   {json.dumps(config)});
</script>
"""
    
    return html


# ---------------------------
# Complete HTML Report Generator
# ---------------------------
def generate_html_report(
    cx: Complex,
    left_exprs: List[str],
    right_exprs: List[str],
    pdb_content: str,
    output_file: str = "vdw_report.html",
    cutoff: float = 5.0,
    energy_threshold: float = -2.0,
    top_n: int = 20,
    include_3d: bool = True,
) -> str:
    """
    Generate complete HTML report with interactive visualizations.
    
    Args:
        cx: Complex object
        left_exprs: Selection expression for left side
        right_exprs: Selection expression for right side
        pdb_content: PDB file content as string
        output_file: Output HTML filename
        cutoff: Distance cutoff (Angstroms)
        energy_threshold: Hotspot threshold (kcal/mol)
        top_n: Number of top hotspots
        include_3d: Include 3D molecular viewer
    
    Returns:
        Path to generated HTML file
    """
    # Calculate energies for BOTH groups
    per_res_energy_left = per_residue_LJ_decomposition(
        cx, left_exprs, right_exprs, cutoff=cutoff
    )
    per_res_energy_right = per_residue_LJ_decomposition(
        cx, right_exprs, left_exprs, cutoff=cutoff
    )

    # Combined dict for shared charts (left overwrites right if key collision is impossible
    # since they are different chain groups)
    per_res_energy = {**per_res_energy_left, **per_res_energy_right}

    if not per_res_energy:
        print("Error: No energies calculated")
        return ""

    # -------------------------------------------------------
    # Build data-table rows (one per residue, both groups)
    # -------------------------------------------------------
    def _parse_res_id(res_id: str):
        parts = res_id.split()
        resname = parts[0] if parts else '?'
        chain_res = parts[1] if len(parts) > 1 else ''
        if '_' in chain_res:
            chain, resnum = chain_res.split('_', 1)
        elif ':' in chain_res:
            chain, resnum = chain_res.split(':', 1)
        else:
            chain, resnum = '?', chain_res
        return resname, chain, resnum

    _table_rows: list = []
    for res_id, energy in per_res_energy_left.items():
        resname, chain, resnum = _parse_res_id(res_id)
        _table_rows.append({
            'residue': res_id, 'resname': resname, 'chain': chain, 'resnum': resnum,
            'group': 'A', 'energy': round(energy, 4),
            'is_hotspot': energy <= energy_threshold,
        })
    for res_id, energy in per_res_energy_right.items():
        resname, chain, resnum = _parse_res_id(res_id)
        _table_rows.append({
            'residue': res_id, 'resname': resname, 'chain': chain, 'resnum': resnum,
            'group': 'B', 'energy': round(energy, 4),
            'is_hotspot': energy <= energy_threshold,
        })
    _table_rows.sort(key=lambda r: r['energy'])
    vdw_table_json = json.dumps(_table_rows)

    # -------------------------------------------------------
    # Generate visualizations
    # -------------------------------------------------------
    viewer_html = ""
    if include_3d:
        viewer_html = generate_3dmol_viewer(
            pdb_content, per_res_energy_left, per_res_energy_right,
            energy_threshold=energy_threshold,
        )

    distribution_html = create_plotly_distribution(per_res_energy)
    hotspot_ranking_html = create_plotly_hotspot_ranking(per_res_energy, energy_threshold, top_n=top_n)

    # Generate per-chain profiles
    summary = compare_energetic_contributions(per_res_energy)
    chain_profiles_html = ""
    for chain_id in sorted(summary.keys()):
        profile = create_plotly_residue_profile(
            per_res_energy,
            chain_id,
            highlight_threshold=energy_threshold
        )
        chain_profiles_html += f"""
        <div class="chart-container">
            <h3>Chain {chain_id} Energy Profile</h3>
            {profile}
        </div>
        """

    # Calculate summary statistics
    all_energies = list(per_res_energy.values())
    total_energy = sum(all_energies)
    favorable = sum(e for e in all_energies if e < 0)
    unfavorable = sum(e for e in all_energies if e >= 0)
    hotspots = identify_energetic_hotspots(per_res_energy, energy_threshold=energy_threshold)

    import statistics
    mean_energy = statistics.mean(all_energies)
    median_energy = statistics.median(all_energies)


    # Build HTML document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VDW Energy Analysis Report</title>
    
    <!-- External Libraries -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: white;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #fffffff 0%, #ffffff 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            color:black;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .stat-card .unit {{
            font-size: 0.8em;
            color: #666;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .chart-container {{
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .chart-container h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .viewer-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .info-box strong {{
            color: #1976d2;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}

        /* ---- Data Table ---- */
        .section-toggle {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-toggle::after {{
            content: '▼';
            font-size: 0.7em;
            transition: transform 0.2s;
        }}
        .section-toggle.collapsed::after {{
            transform: rotate(-90deg);
        }}
        .section-content {{
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        .section-content.collapsed {{
            display: none;
        }}
        .dt-toolbar {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin: 14px 0;
        }}
        .dt-toolbar input {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            width: 220px;
        }}
        .dt-toolbar select {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
        }}
        .dt-dl-btn {{
            padding: 7px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: transform 0.15s;
        }}
        .dt-dl-btn:hover {{ transform: translateY(-1px); }}
        .dt-scroll {{
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            max-height: 520px;
            overflow-y: auto;
        }}
        .dt-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .dt-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            position: sticky;
            top: 0;
            white-space: nowrap;
        }}
        .dt-table th:hover {{ background: linear-gradient(135deg, #5568d3 0%, #65408d 100%); }}
        .dt-table th::after {{ content: ' ⇅'; opacity: 0.5; }}
        .dt-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #f0f0f0;
            white-space: nowrap;
        }}
        .dt-table tr:hover td {{ background: #f8f9fa; }}
        .dt-badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            color: white;
        }}
        .dt-badge-A  {{ background: #3498db; }}
        .dt-badge-B  {{ background: #e67e22; }}
        .dt-badge-hot {{ background: #e74c3c; }}
        .dt-energy-fav   {{ color: #2980b9; font-weight: 600; }}
        .dt-energy-neut  {{ color: #f39c12; font-weight: 600; }}
        .dt-energy-unfav {{ color: #e74c3c; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Comprehensive van der Waals interaction analysis</h1>
        </header>
        
        <div class="summary">
            <div class="stat-card">
                <h3>Group A Residues</h3>
                <div class="value">{len(per_res_energy_left)}</div>
                <div class="unit">{", ".join(left_exprs)}</div>
            </div>
            <div class="stat-card">
                <h3>Group B Residues</h3>
                <div class="value">{len(per_res_energy_right)}</div>
                <div class="unit">{", ".join(right_exprs)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Energy</h3>
                <div class="value">{total_energy:.2f}</div>
                <div class="unit">kcal/mol</div>
            </div>
            <div class="stat-card">
                <h3>Mean Energy</h3>
                <div class="value">{mean_energy:.2f}</div>
                <div class="unit">kcal/mol (per residue)</div>
            </div>
            <div class="stat-card">
                <h3>Hotspots</h3>
                <div class="value">{len(hotspots)}</div>
                <div class="unit">E &lt; {energy_threshold:.1f} kcal/mol</div>
            </div>
        </div>
        
        <div class="content">
            {"" if not include_3d else f'''
            <div class="section">
                <h2>3D Molecular Viewer</h2>
                <div class="info-box">
                    <strong>Colour coding:</strong> Blue → Favorable &nbsp;|&nbsp; Red → Unfavorable &nbsp;|&nbsp;
                    <span style="display:inline-block;background:#3498db;color:white;padding:1px 8px;border-radius:10px;font-size:12px;">A</span> Group A (left) &nbsp;
                    <span style="display:inline-block;background:#e67e22;color:white;padding:1px 8px;border-radius:10px;font-size:12px;">B</span> Group B (right)<br/>
                    Rotate: drag &nbsp;|&nbsp; Zoom: scroll &nbsp;|&nbsp; <strong>Click residue</strong>: show VDW details
                </div>
                <div class="viewer-container">
                    {viewer_html}
                </div>
            </div>
            '''}
            
            <div class="section">
                <h2>Energy Distribution</h2>
                <div class="chart-container">
                    {distribution_html}
                </div>
            </div>
            
            <div class="section">
                <h2>Top Energetic Hotspots</h2>
                <div class="info-box">
                    Residues with strongest favorable VDW interactions (most negative energies)
                </div>
                <div class="chart-container">
                    {hotspot_ranking_html}
                </div>
            </div>
            
            <div class="section">
                <h2>Per-Chain Energy Profiles</h2>
                {chain_profiles_html}
            </div>
            
            <!-- Data Table Section -->
            <div class="section">
                <h2 class="section-toggle" onclick="toggleVdwSection(this)">📋 Data Table</h2>
                <div class="section-content">
                    <div class="dt-toolbar">
                        <input type="text" id="vdwDtSearch" placeholder="Filter residue, chain, group…" oninput="vdwFilterDt()">
                        <select id="vdwDtGroup" onchange="vdwFilterDt()">
                            <option value="">All groups</option>
                            <option value="A">Group A</option>
                            <option value="B">Group B</option>
                        </select>
                        <button class="dt-dl-btn" onclick="vdwDownloadDt()">💾 Download CSV</button>
                    </div>
                    <div class="dt-scroll">
                        <table class="dt-table" id="vdwDtTable">
                            <thead>
                                <tr>
                                    <th onclick="vdwSortDt(0)">Residue</th>
                                    <th onclick="vdwSortDt(1)">Chain</th>
                                    <th onclick="vdwSortDt(2)">Res #</th>
                                    <th onclick="vdwSortDt(3)">Group</th>
                                    <th onclick="vdwSortDt(4)">VDW Energy (kcal/mol)</th>
                                    <th onclick="vdwSortDt(5)">Hotspot</th>
                                </tr>
                            </thead>
                            <tbody id="vdwDtBody"></tbody>
                        </table>
                    </div>
                    <div id="vdwDtCount" style="margin-top:6px; font-size:12px; color:#888;"></div>
                </div>
            </div>

        </div>

        <footer>
            <p>Generated VDW Energy Analysis Report | Cutoff: {cutoff:.1f} Å | Threshold: {energy_threshold:.1f} kcal/mol</p>
            <p>Blue = Favorable interactions | Red = Unfavorable interactions | Groups: A={left_exprs} · B={right_exprs}</p>
        </footer>
    </div>

<script>
// ---- Data Table Logic ----
var _vdwAllRows = {vdw_table_json};
var _vdwFiltered = _vdwAllRows.slice();
var _vdwSortCol = 4;
var _vdwSortAsc = true;

function _vdwEnergyClass(e) {{
    if (e <= {energy_threshold}) return 'dt-energy-fav';
    if (e < 0) return 'dt-energy-neut';
    return 'dt-energy-unfav';
}}

function _vdwRenderDt() {{
    var tbody = document.getElementById('vdwDtBody');
    tbody.innerHTML = '';
    _vdwFiltered.forEach(function(r) {{
        var tr = document.createElement('tr');
        var hotBadge = r.is_hotspot ? '<span class="dt-badge dt-badge-hot">Hotspot</span>' : '';
        tr.innerHTML =
            '<td>' + r.resname + '</td>' +
            '<td>' + r.chain + '</td>' +
            '<td>' + r.resnum + '</td>' +
            '<td><span class="dt-badge dt-badge-' + r.group + '">' + r.group + '</span></td>' +
            '<td class="' + _vdwEnergyClass(r.energy) + '">' + r.energy.toFixed(4) + '</td>' +
            '<td>' + hotBadge + '</td>';
        tbody.appendChild(tr);
    }});
    var countEl = document.getElementById('vdwDtCount');
    if (countEl) countEl.textContent = 'Showing ' + _vdwFiltered.length + ' of ' + _vdwAllRows.length + ' residues';
}}

function vdwFilterDt() {{
    var q = document.getElementById('vdwDtSearch').value.toLowerCase();
    var grp = document.getElementById('vdwDtGroup').value;
    _vdwFiltered = _vdwAllRows.filter(function(r) {{
        var matchQ = !q || r.resname.toLowerCase().includes(q) || r.chain.toLowerCase().includes(q) ||
                     r.resnum.toLowerCase().includes(q) || r.group.toLowerCase().includes(q);
        var matchG = !grp || r.group === grp;
        return matchQ && matchG;
    }});
    _vdwApplySortDt();
    _vdwRenderDt();
}}

function _vdwApplySortDt() {{
    _vdwFiltered.sort(function(a, b) {{
        var va, vb;
        switch (_vdwSortCol) {{
            case 0: va = a.resname; vb = b.resname; break;
            case 1: va = a.chain;   vb = b.chain;   break;
            case 2: va = parseInt(a.resnum, 10) || 0; vb = parseInt(b.resnum, 10) || 0; break;
            case 3: va = a.group;   vb = b.group;   break;
            case 4: va = a.energy;  vb = b.energy;  break;
            default: va = a.is_hotspot ? 1 : 0; vb = b.is_hotspot ? 1 : 0;
        }}
        if (typeof va === 'number') return _vdwSortAsc ? va - vb : vb - va;
        return _vdwSortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    }});
}}

function vdwSortDt(col) {{
    if (_vdwSortCol === col) {{ _vdwSortAsc = !_vdwSortAsc; }}
    else {{ _vdwSortCol = col; _vdwSortAsc = col === 4 ? true : true; }}
    _vdwApplySortDt();
    _vdwRenderDt();
}}

function vdwDownloadDt() {{
    var header = 'residue,resname,chain,resnum,group,vdw_energy_kcal_mol,is_hotspot';
    var rows = _vdwAllRows.map(function(r) {{
        return r.residue + ',' + r.resname + ',' + r.chain + ',' + r.resnum + ',' +
               r.group + ',' + r.energy + ',' + (r.is_hotspot ? 'true' : 'false');
    }});
    var blob = new Blob([header + '\\n' + rows.join('\\n')], {{type: 'text/csv'}});
    var a = document.createElement('a');
    a.download = 'vdw_data_table.csv';
    a.href = URL.createObjectURL(blob);
    a.click();
}}

function toggleVdwSection(toggle) {{
    toggle.classList.toggle('collapsed');
    var content = toggle.nextElementSibling;
    content.classList.toggle('collapsed');
}}

// Initial render
_vdwApplySortDt();
_vdwRenderDt();
</script>
</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n3. HTML report generated: {output_file}")
    print("=" * 80)
    
    return os.path.abspath(output_file)


# ---------------------------
# Utility function to generate standalone HTML components
# ---------------------------
def generate_html_component(
    per_res_energy: Dict[str, float],
    component_type: str = "distribution",
    **kwargs
) -> str:
    """
    Generate a standalone HTML component for embedding.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        component_type: Type of component ("distribution", "chains", 
                       "hotspots", "profile", "matrix")
        **kwargs: Additional arguments passed to the component function
    
    Returns:
        HTML string with the component (includes necessary scripts)
    
    Example:
        >>> energies = per_residue_LJ_decomposition(cx, ["A"], ["B"])
        >>> html = generate_html_component(energies, "distribution")
        >>> # Embed in your web page
    """
    # Common header with required libraries
    header = """
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
"""
    
    if component_type == "distribution":
        content = create_plotly_distribution(per_res_energy, **kwargs)
    elif component_type == "chains":
        content = create_plotly_chain_comparison(per_res_energy, **kwargs)
    elif component_type == "hotspots":
        content = create_plotly_hotspot_ranking(per_res_energy, **kwargs)
    elif component_type == "profile":
        if 'chain_id' not in kwargs:
            return "<p>Error: chain_id required for profile component</p>"
        content = create_plotly_residue_profile(per_res_energy, **kwargs)
    elif component_type == "matrix":
        if 'cx' not in kwargs or 'chain_exprs' not in kwargs:
            return "<p>Error: cx and chain_exprs required for matrix component</p>"
        content = create_plotly_contact_matrix(**kwargs)
    else:
        return f"<p>Error: Unknown component type '{component_type}'</p>"
    
    return header + content