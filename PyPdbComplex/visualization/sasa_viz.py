"""
SASA Analysis Interactive HTML Visualizations

This module provides interactive HTML visualizations for SASA analysis results,
designed for web application integration. Follows the same pattern as geometry_viz.py.

Functions:
    - generate_burial_plot: Interactive burial fraction scatter/histogram
    - generate_sasa_comparison: Bound vs unbound SASA comparison
    - generate_chain_heatmap: Chain-by-chain burial heatmap
    - generate_interface_table: Interactive sortable interface residue table
    - generate_sasa_dashboard: Comprehensive dashboard with all plots
    - visualize_sasa: Main function for generating visualizations (like visualize_geometry)

Usage:
    >>> from sasa_analysis import compare_bound_unbound_sasa
    >>> from sasa_viz import visualize_sasa
    >>> 
    >>> # Run analysis
    >>> sasa_results = compare_bound_unbound_sasa(complex, ["chain A", "chain B"])
    >>> 
    >>> # Generate all visualizations
    >>> files = visualize_sasa(sasa_results, viz_type='all', 
    ...                        output_file='results/sasa_analysis')
    >>> 
    >>> # Returns: {'dashboard': 'results/sasa_analysis.html', ...}
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json
from collections import defaultdict

from ._base import html_page


# Helper function for parsing residue IDs
def _parse_residue_id(res_id: str) -> tuple:
    """
    Parse residue ID into components.
    
    Args:
        res_id: Residue ID string (e.g., "ASP A:123")
    
    Returns:
        Tuple of (resname, chain, resnum)
    """
    try:
        if '_' in res_id:
            parts = res_id.split('_')
            resname_chain = parts[0].strip().split()
            resname = resname_chain[0] if resname_chain else "UNK"
            chain = resname_chain[1] if len(resname_chain) > 1 else "?"
            resnum = parts[1] if len(parts) > 1 else "?"
        elif ':' in res_id:
            parts = res_id.split(':')
            resname_chain = parts[0].strip().split()
            resname = resname_chain[0] if resname_chain else "UNK"
            chain = resname_chain[1] if len(resname_chain) > 1 else "?"
            resnum = parts[1] if len(parts) > 1 else "?"
        else:
            resname = res_id[:3] if len(res_id) >= 3 else res_id
            chain = "?"
            resnum = "?"
    except:
        resname = "UNK"
        chain = "?"
        resnum = "?"
    
    return resname, chain, resnum

def generate_burial_plot(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "Residue Burial Distribution"
) -> str:
    """
    Generate interactive burial fraction distribution plot.
    
    Creates a toggleable scatter/histogram plot showing burial fractions.
    Color-coded by burial level with interactive controls.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Plot title
    
    Returns:
        HTML string with interactive plot
    
    Example:
        >>> html = generate_burial_plot(sasa_results)
        >>> with open('burial.html', 'w') as f:
        ...     f.write(html)
    """
    
    # Prepare data
    data_points = []
    for res_id, values in sasa_results.items():
        resname, chain, resnum = _parse_residue_id(res_id)
        
        data_points.append({
            "residue": res_id,
            "resname": resname,
            "chain": chain,
            "resnum": resnum,
            "burial": values['buried_fraction'],
            "delta": values['delta'],
            "unbound": values['unbound'],
            "bound": values['bound']
        })
    
    # Count by category
    highly_buried = sum(1 for d in data_points if d['burial'] >= 0.75)
    moderately_buried = sum(1 for d in data_points if 0.5 <= d['burial'] < 0.75)
    slightly_buried = sum(1 for d in data_points if 0.25 <= d['burial'] < 0.5)
    exposed = sum(1 for d in data_points if d['burial'] < 0.25)
    
    extra_css = """
        .stat-card.red { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .stat-card.orange { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
        .stat-card.yellow { background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); }
        .stat-card.cyan { background: linear-gradient(135deg, #74ebd5 0%, #9face6 100%); }

        .controls {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-size: 14px;
            font-weight: 500;
        }
        .control-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        .control-group select {
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 14px;
        }
        .size-controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .size-controls input[type="range"] {
            width: 200px;
        }
        .btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .chart-container {
            position: relative;
            margin: 20px 0;
        }
        #burialChart {
            max-width: 100%;
        }
        .legend {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .legend-items {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Residues</div>
                <div class="stat-value">{len(data_points)}</div>
            </div>
            <div class="stat-card red">
                <div class="stat-label">Highly Buried (≥75%)</div>
                <div class="stat-value">{highly_buried}</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-label">Moderately (50-75%)</div>
                <div class="stat-value">{moderately_buried}</div>
            </div>
            <div class="stat-card yellow">
                <div class="stat-label">Slightly (25-50%)</div>
                <div class="stat-value">{slightly_buried}</div>
            </div>
            <div class="stat-card cyan">
                <div class="stat-label">Exposed (<25%)</div>
                <div class="stat-value">{exposed}</div>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="plotType">Plot Type:</label>
                <select id="plotType">
                    <option value="scatter">Scatter Plot</option>
                    <option value="histogram">Histogram</option>
                </select>
            </div>
            <div class="control-group">
                <input type="checkbox" id="showThreshold" checked>
                <label for="showThreshold">Show interface threshold (50%)</label>
            </div>
            <div class="control-group">
                <input type="checkbox" id="colorByChain">
                <label for="colorByChain">Color by chain</label>
            </div>

            <div class="size-controls">
                <label for="sizeSlider">Chart Height:</label>
                <input type="range" id="sizeSlider" min="400" max="800" value="600" step="50">
                <span id="sizeValue">600px</span>
            </div>

            <button class="btn" onclick="downloadSnapshot()">📸 Save</button>
            <button class="btn" onclick="downloadData()">💾 Export CSV</button>
        </div>

        <div class="legend">
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-box" style="background: #e74c3c;"></div>
                    <span>Highly buried (≥75%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #e67e22;"></div>
                    <span>Moderately buried (50-75%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #f39c12;"></div>
                    <span>Slightly buried (25-50%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #3498db;"></div>
                    <span>Exposed (<25%)</span>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="burialChart"></canvas>
        </div>

        <div class="info-box">
            <p><strong>Burial Fraction:</strong> (Unbound SASA - Bound SASA) / Unbound SASA</p>
            <p><strong>Interface Residues:</strong> Typically defined as burial fraction ≥ 0.5 (50%)</p>
            <p><strong>ΔSASA:</strong> Change in solvent accessible surface area upon binding (Å²)</p>
        </div>
    </div>

    <script>
        const dataPoints = {json.dumps(data_points)};
        let currentChart = null;
        let chartHeight = 600;

        const chainColors = {{
            'H': 'rgba(231, 76, 60, 0.6)',
            'L': 'rgba(52, 152, 219, 0.6)',
            'A': 'rgba(46, 204, 113, 0.6)',
            'B': 'rgba(241, 196, 15, 0.6)',
            'C': 'rgba(155, 89, 182, 0.6)',
        }};

        function getColorByBurial(burial) {{
            if (burial >= 0.75) return 'rgba(231, 76, 60, 0.6)';
            if (burial >= 0.50) return 'rgba(230, 126, 34, 0.6)';
            if (burial >= 0.25) return 'rgba(243, 156, 18, 0.6)';
            return 'rgba(52, 152, 219, 0.6)';
        }}

        function createChart() {{
            const ctx = document.getElementById('burialChart').getContext('2d');
            const plotType = document.getElementById('plotType').value;
            const showThreshold = document.getElementById('showThreshold').checked;
            const colorByChain = document.getElementById('colorByChain').checked;

            if (currentChart) {{
                currentChart.destroy();
            }}

            const canvas = document.getElementById('burialChart');
            canvas.height = chartHeight;

            if (plotType === 'scatter') {{
                // Scatter plot
                const scatterData = dataPoints.map((p, idx) => ({{
                    x: idx,
                    y: p.burial * 100,
                    residue: p.residue,
                    chain: p.chain,
                    delta: p.delta
                }}));

                const pointColors = colorByChain
                    ? scatterData.map(p => chainColors[p.chain] || 'rgba(100, 100, 100, 0.6)')
                    : scatterData.map(p => getColorByBurial(p.y / 100));

                const datasets = [{{
                    label: 'Residues',
                    data: scatterData,
                    backgroundColor: pointColors,
                    borderColor: pointColors.map(c => c.replace('0.6', '1')),
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }}];

                if (showThreshold) {{
                    datasets.push({{
                        label: 'Interface threshold (50%)',
                        data: [{{x: 0, y: 50}}, {{x: dataPoints.length, y: 50}}],
                        borderColor: 'rgba(231, 76, 60, 0.8)',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        pointRadius: 0,
                        fill: false,
                        type: 'line'
                    }});
                }}

                currentChart = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{ datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: true, position: 'top' }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const point = context.raw;
                                        if (point.residue) {{
                                            return [
                                                point.residue,
                                                `Burial: ${{point.y.toFixed(1)}}%`,
                                                `ΔSASA: ${{point.delta.toFixed(1)}} Å²`
                                            ];
                                        }}
                                        return context.dataset.label;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Residue Index', font: {{ size: 14, weight: 'bold' }} }},
                                grid: {{ color: 'rgba(0,0,0,0.05)' }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Burial Fraction (%)', font: {{ size: 14, weight: 'bold' }} }},
                                min: 0,
                                max: 100,
                                ticks: {{ stepSize: 20 }},
                                grid: {{ color: 'rgba(0,0,0,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }} else {{
                // Histogram
                const bins = new Array(20).fill(0);
                dataPoints.forEach(p => {{
                    const binIndex = Math.min(19, Math.floor(p.burial * 20));
                    bins[binIndex]++;
                }});

                const labels = [];
                const colors = [];
                for (let i = 0; i < 20; i++) {{
                    const start = i * 5;
                    const end = (i + 1) * 5;
                    labels.push(`${{start}}-${{end}}%`);
                    colors.push(getColorByBurial(start / 100));
                }}

                const datasets = [{{
                    label: 'Residue Count',
                    data: bins,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.6', '1')),
                    borderWidth: 1
                }}];

                currentChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{ labels, datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return `Count: ${{context.parsed.y}}`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Burial Fraction (%)', font: {{ size: 14, weight: 'bold' }} }},
                                ticks: {{ maxRotation: 45, minRotation: 45 }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Number of Residues', font: {{ size: 14, weight: 'bold' }} }},
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            }}
        }}

        function downloadSnapshot() {{
            const canvas = document.getElementById('burialChart');
            const link = document.createElement('a');
            link.download = 'burial_plot.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function downloadData() {{
            const csv = 'residue,chain,resnum,burial_fraction,delta_sasa,unbound_sasa,bound_sasa\\n' +
                dataPoints.map(p =>
                    `${{p.residue}},${{p.chain}},${{p.resnum}},${{p.burial}},${{p.delta}},${{p.unbound}},${{p.bound}}`
                ).join('\\n');

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const link = document.createElement('a');
            link.download = 'burial_data.csv';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        // Event listeners
        document.getElementById('plotType').addEventListener('change', createChart);
        document.getElementById('showThreshold').addEventListener('change', createChart);
        document.getElementById('colorByChain').addEventListener('change', createChart);
        document.getElementById('sizeSlider').addEventListener('input', (e) => {{
            chartHeight = parseInt(e.target.value);
            document.getElementById('sizeValue').textContent = chartHeight + 'px';
            createChart();
        }});

        // Initial render
        createChart();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"])


def generate_sasa_comparison(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "Bound vs Unbound SASA Comparison",
    top_n: int = 30
) -> str:
    """
    Generate bound vs unbound SASA comparison for top interface residues.
    
    Creates a horizontal bar chart showing unbound and bound SASA side-by-side
    for the top N residues ranked by burial (ΔSASA).
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Plot title
        top_n: Number of top residues to show (default: 30)
    
    Returns:
        HTML string with interactive comparison plot
    
    Example:
        >>> html = generate_sasa_comparison(sasa_results, top_n=20)
        >>> with open('comparison.html', 'w') as f:
        ...     f.write(html)
    """
    
    # Get top interface residues by ΔSASA (burial amount)
    sorted_residues = sorted(
        sasa_results.items(),
        key=lambda x: x[1]['delta'],
        reverse=True
    )[:min(top_n, len(sasa_results))]
    
    data_points = []
    for res_id, values in sorted_residues:
        resname, chain, resnum = _parse_residue_id(res_id)
        
        # Create short label for chart
        short_label = f"{chain}:{resnum}"
        
        data_points.append({
            "residue": res_id,
            "label": short_label,
            "resname": resname,
            "unbound": values['unbound'],
            "bound": values['bound'],
            "delta": values['delta'],
            "burial": values['buried_fraction']
        })
    
    extra_css = """
        .controls {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-size: 14px;
            font-weight: 500;
        }
        .control-group input[type="range"] {
            width: 150px;
        }
        .btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .chart-container {
            position: relative;
            margin: 20px 0;
        }
        #comparisonChart {
            max-width: 100%;
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <div class="control-group">
                <label for="topN">Show top:</label>
                <input type="range" id="topN" min="10" max="50" value="{min(top_n, 50)}" step="5">
                <span id="topNValue">{min(top_n, 50)}</span> residues
            </div>
            <button class="btn" onclick="downloadSnapshot()">📸 Save</button>
            <button class="btn" onclick="downloadData()">💾 Export CSV</button>
        </div>

        <div class="chart-container">
            <canvas id="comparisonChart"></canvas>
        </div>

        <div class="info-box">
            <p><strong>Top interface residues</strong> ranked by ΔSASA (change in solvent accessible surface area)</p>
            <p>Blue bars show unbound (isolated) SASA, orange bars show bound (complex) SASA</p>
            <p>Larger difference = more buried upon binding = more important for interface</p>
        </div>
    </div>

    <script>
        const allDataPoints = {json.dumps(data_points)};
        let currentChart = null;
        let numResidues = {min(top_n, len(data_points))};

        function createChart() {{
            const ctx = document.getElementById('comparisonChart').getContext('2d');

            if (currentChart) {{
                currentChart.destroy();
            }}

            const dataSlice = allDataPoints.slice(0, numResidues);
            const labels = dataSlice.map(d => d.label);
            const unboundData = dataSlice.map(d => d.unbound);
            const boundData = dataSlice.map(d => d.bound);

            const canvas = document.getElementById('comparisonChart');
            canvas.height = Math.max(500, numResidues * 15);

            currentChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Unbound SASA',
                            data: unboundData,
                            backgroundColor: 'rgba(52, 152, 219, 0.8)',
                            borderColor: 'rgba(52, 152, 219, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Bound SASA',
                            data: boundData,
                            backgroundColor: 'rgba(230, 126, 34, 0.8)',
                            borderColor: 'rgba(230, 126, 34, 1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top',
                            labels: {{
                                font: {{ size: 12, weight: 'bold' }},
                                padding: 15
                            }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    const idx = context[0].dataIndex;
                                    return dataSlice[idx].residue;
                                }},
                                afterLabel: function(context) {{
                                    const idx = context.dataIndex;
                                    const data = dataSlice[idx];
                                    return [
                                        `ΔSASA: ${{data.delta.toFixed(1)}} Å²`,
                                        `Burial: ${{(data.burial * 100).toFixed(1)}}%`
                                    ];
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'SASA (Å²)', font: {{ size: 14, weight: 'bold' }} }},
                            beginAtZero: true,
                            grid: {{ color: 'rgba(0,0,0,0.1)' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Residue', font: {{ size: 14, weight: 'bold' }} }},
                            ticks: {{ font: {{ size: 11 }} }}
                        }}
                    }}
                }}
            }});
        }}

        function downloadSnapshot() {{
            const canvas = document.getElementById('comparisonChart');
            const link = document.createElement('a');
            link.download = 'sasa_comparison.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function downloadData() {{
            const csv = 'residue,label,resname,unbound_sasa,bound_sasa,delta_sasa,burial_fraction\\n' +
                allDataPoints.slice(0, numResidues).map(p =>
                    `${{p.residue}},${{p.label}},${{p.resname}},${{p.unbound}},${{p.bound}},${{p.delta}},${{p.burial}}`
                ).join('\\n');

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const link = document.createElement('a');
            link.download = 'sasa_comparison.csv';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        // Event listener for slider
        document.getElementById('topN').addEventListener('input', (e) => {{
            numResidues = parseInt(e.target.value);
            document.getElementById('topNValue').textContent = numResidues;
            createChart();
        }});

        // Initial render
        createChart();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"],
                     max_width=1600)

def generate_chain_heatmap(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "Residue Burial Heatmap by Chain"
) -> str:
    """
    Generate chain-by-chain burial heatmap.
    
    Creates one bar chart per chain showing burial fraction across the sequence.
    Color-coded from blue (exposed) to red (buried).
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Plot title
    
    Returns:
        HTML string with interactive heatmap
    
    Example:
        >>> html = generate_chain_heatmap(sasa_results)
        >>> with open('heatmap.html', 'w') as f:
        ...     f.write(html)
    """
    
    # Organize data by chain and residue number
    chain_residues = defaultdict(dict)
    
    for res_id, values in sasa_results.items():
        resname, chain, resnum = _parse_residue_id(res_id)
        
        try:
            resnum_int = int(resnum)
        except:
            continue
        
        chain_residues[chain][resnum_int] = {
            'burial': values['buried_fraction'],
            'delta': values['delta'],
            'residue': res_id
        }
    
    # Prepare data for JavaScript
    chain_data_js = {}
    for chain, residues in sorted(chain_residues.items()):
        resnums = sorted(residues.keys())
        chain_data_js[chain] = {
            'resnums': resnums,
            'burial': [residues[rn]['burial'] for rn in resnums],
            'delta': [residues[rn]['delta'] for rn in resnums],
            'residues': [residues[rn]['residue'] for rn in resnums]
        }
    
    n_chains = len(chain_data_js)
    
    extra_css = """
        .controls {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
            margin: 20px 0;
        }
        .chart-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }
        .chart-card h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
        }
        .chart-container {
            position: relative;
            height: 150px;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            align-items: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .legend-gradient {
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #3498db, #f39c12, #e74c3c);
            border-radius: 4px;
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <button class="btn" onclick="downloadAllSnapshots()">📸 Save All</button>
        </div>

        <div class="legend">
            <div class="legend-item">
                <span>Exposed</span>
                <div class="legend-gradient"></div>
                <span>Buried</span>
            </div>
        </div>

        <div class="charts-grid" id="chartsContainer">
        </div>

        <div class="info-box">
            <p><strong>Color scale:</strong> Blue (exposed/low burial) → Yellow (moderate) → Red (buried/high burial)</p>
            <p>Each row represents one chain, showing burial fraction across the sequence</p>
        </div>
    </div>

    <script>
        const chainData = {json.dumps(chain_data_js)};
        const chains = Object.keys(chainData).sort();
        const charts = {{}};

        function getColorByBurial(burial) {{
            if (burial < 0.25) return 'rgba(52, 152, 219, 0.8)';  // Blue
            if (burial < 0.50) return 'rgba(243, 156, 18, 0.8)';  // Yellow
            if (burial < 0.75) return 'rgba(230, 126, 34, 0.8)';  // Orange
            return 'rgba(231, 76, 60, 0.8)';  // Red
        }}

        function createChainChart(chain) {{
            const data = chainData[chain];
            const colors = data.burial.map(b => getColorByBurial(b));

            const canvas = document.getElementById(`chart_${{chain}}`);
            const ctx = canvas.getContext('2d');

            charts[chain] = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.resnums,
                    datasets: [{{
                        label: `Chain ${{chain}}`,
                        data: data.burial.map(b => b * 100),
                        backgroundColor: colors,
                        borderColor: colors.map(c => c.replace('0.8', '1')),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    indexAxis: 'x',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    const idx = context[0].dataIndex;
                                    return data.residues[idx];
                                }},
                                label: function(context) {{
                                    const idx = context.dataIndex;
                                    return [
                                        `Burial: ${{context.parsed.y.toFixed(1)}}%`,
                                        `ΔSASA: ${{data.delta[idx].toFixed(1)}} Å²`
                                    ];
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Residue Number', font: {{ size: 12, weight: 'bold' }} }},
                            ticks: {{
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 20
                            }},
                            grid: {{ color: 'rgba(0,0,0,0.05)' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Burial %', font: {{ size: 12, weight: 'bold' }} }},
                            min: 0,
                            max: 100,
                            ticks: {{ stepSize: 25 }},
                            grid: {{ color: 'rgba(0,0,0,0.1)' }}
                        }}
                    }}
                }}
            }});
        }}

        function renderAllCharts() {{
            const container = document.getElementById('chartsContainer');

            chains.forEach(chain => {{
                const card = document.createElement('div');
                card.className = 'chart-card';
                card.innerHTML = `
                    <h3>Chain ${{chain}} (${{chainData[chain].resnums.length}} residues)</h3>
                    <div class="chart-container">
                        <canvas id="chart_${{chain}}"></canvas>
                    </div>
                `;
                container.appendChild(card);

                createChainChart(chain);
            }});
        }}

        function downloadAllSnapshots() {{
            chains.forEach(chain => {{
                const canvas = document.getElementById(`chart_${{chain}}`);
                const link = document.createElement('a');
                link.download = `burial_heatmap_chain_${{chain}}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            }});
        }}

        // Initial render
        renderAllCharts();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"],
                     max_width=1800)

def generate_interface_table(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "SASA Data Table",
    burial_threshold: float = 0.5
) -> str:
    """
    Generate interactive sortable table of interface residues.
    
    Creates a filterable, sortable HTML table showing interface residues
    with burial progress bars and statistics.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Table title
        burial_threshold: Minimum burial fraction for interface (default: 0.5)
    
    Returns:
        HTML string with interactive table
    
    Example:
        >>> html = generate_interface_table(sasa_results, burial_threshold=0.5)
        >>> with open('table.html', 'w') as f:
        ...     f.write(html)
    """
    
    # Filter and prepare interface residues
    interface_residues = []
    for res_id, values in sasa_results.items():
        if values['buried_fraction'] >= burial_threshold:
            resname, chain, resnum = _parse_residue_id(res_id)
            
            interface_residues.append({
                "residue": res_id,
                "resname": resname,
                "chain": chain,
                "resnum": resnum,
                "unbound": values['unbound'],
                "bound": values['bound'],
                "delta": values['delta'],
                "burial": values['buried_fraction']
            })
    
    # Sort by burial fraction (descending)
    interface_residues.sort(key=lambda x: x['burial'], reverse=True)
    
    # Calculate statistics
    avg_burial = sum(r['burial'] for r in interface_residues) / len(interface_residues) * 100 if interface_residues else 0
    total_delta = sum(r['delta'] for r in interface_residues)
    num_chains = len(set(r['chain'] for r in interface_residues))
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-align: center;
        }}
        .stat-card.red {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-card.green {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
        .stat-label {{
            font-size: 11px;
            opacity: 0.9;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .control-group input[type="text"] {{
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-size: 14px;
            width: 200px;
        }}
        .btn {{
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
        }}
        .btn:hover {{ transform: translateY(-2px); }}
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            background: linear-gradient(135deg, #5568d3 0%, #65408d 100%);
        }}
        th::after {{
            content: ' ⇅';
            opacity: 0.5;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e1e4e8;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .burial-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .bar-container {{
            flex: 1;
            height: 20px;
            background: #e1e4e8;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .burial-high {{ background: #e74c3c; }}
        .burial-med {{ background: #e67e22; }}
        .burial-low {{ background: #f39c12; }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #666;
        }}
        .info-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196f3;
        }}
        .info-box p {{
            margin: 5px 0;
            font-size: 13px;
            color: #1565c0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Interface Residues</div>
                <div class="stat-value">{len(interface_residues)}</div>
            </div>
            <div class="stat-card red">
                <div class="stat-label">Avg Burial</div>
                <div class="stat-value">{avg_burial:.1f}%</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">Total ΔSASA</div>
                <div class="stat-value">{total_delta:.0f} Å²</div>
            </div>
            <div class="stat-card green">
                <div class="stat-label">Chains</div>
                <div class="stat-value">{num_chains}</div>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>Search:</label>
                <input type="text" id="searchInput" placeholder="Filter by residue, chain...">
            </div>
            <button class="btn" onclick="downloadData()">💾 Download CSV</button>
        </div>
        
        <div class="table-container">
            <table id="interfaceTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Residue</th>
                        <th onclick="sortTable(1)">Chain</th>
                        <th onclick="sortTable(2)">Resnum</th>
                        <th onclick="sortTable(3)">Unbound (Å²)</th>
                        <th onclick="sortTable(4)">Bound (Å²)</th>
                        <th onclick="sortTable(5)">ΔSASA (Å²)</th>
                        <th onclick="sortTable(6)">Burial %</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                </tbody>
            </table>
        </div>
        
        <div class="info-box">
            <p><strong>Interface residues:</strong> Defined as burial fraction ≥ {burial_threshold * 100:.0f}%</p>
            <p>Click column headers to sort. Use search box to filter results.</p>
        </div>
    </div>
    
    <script>
        const allResidues = {json.dumps(interface_residues)};
        let filteredResidues = [...allResidues];
        let sortColumn = 6;  // Start sorted by burial
        let sortAscending = false;
        
        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            if (filteredResidues.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="7" class="no-data">No residues match your filter</td></tr>';
                return;
            }}
            
            filteredResidues.forEach(res => {{
                const row = tbody.insertRow();
                
                // Residue
                row.insertCell().textContent = res.resname;
                
                // Chain
                row.insertCell().textContent = res.chain;
                
                // Resnum
                row.insertCell().textContent = res.resnum;
                
                // Unbound
                row.insertCell().textContent = res.unbound.toFixed(1);
                
                // Bound
                row.insertCell().textContent = res.bound.toFixed(1);
                
                // Delta
                row.insertCell().textContent = res.delta.toFixed(1);
                
                // Burial with bar
                const burialCell = row.insertCell();
                const burialPercent = (res.burial * 100).toFixed(1);
                const barClass = res.burial >= 0.75 ? 'burial-high' : 
                                res.burial >= 0.50 ? 'burial-med' : 'burial-low';
                
                burialCell.innerHTML = `
                    <div class="burial-bar">
                        <div class="bar-container">
                            <div class="bar-fill ${{barClass}}" style="width: ${{burialPercent}}%"></div>
                        </div>
                        <span>${{burialPercent}}%</span>
                    </div>
                `;
            }});
        }}
        
        function sortTable(column) {{
            if (sortColumn === column) {{
                sortAscending = !sortAscending;
            }} else {{
                sortColumn = column;
                sortAscending = column <= 2;  // Ascending for text columns
            }}
            
            filteredResidues.sort((a, b) => {{
                let valA, valB;
                
                switch(column) {{
                    case 0: valA = a.resname; valB = b.resname; break;
                    case 1: valA = a.chain; valB = b.chain; break;
                    case 2: valA = parseInt(a.resnum) || 0; valB = parseInt(b.resnum) || 0; break;
                    case 3: valA = a.unbound; valB = b.unbound; break;
                    case 4: valA = a.bound; valB = b.bound; break;
                    case 5: valA = a.delta; valB = b.delta; break;
                    case 6: valA = a.burial; valB = b.burial; break;
                }}
                
                if (valA < valB) return sortAscending ? -1 : 1;
                if (valA > valB) return sortAscending ? 1 : -1;
                return 0;
            }});
            
            renderTable();
        }}
        
        function filterTable() {{
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            
            filteredResidues = allResidues.filter(res => 
                res.residue.toLowerCase().includes(searchTerm) ||
                res.resname.toLowerCase().includes(searchTerm) ||
                res.chain.toLowerCase().includes(searchTerm) ||
                res.resnum.toString().includes(searchTerm)
            );
            
            renderTable();
        }}
        
        function downloadData() {{
            const csv = 'residue,resname,chain,resnum,unbound_sasa,bound_sasa,delta_sasa,burial_fraction\\n' +
                allResidues.map(r =>
                    `${{r.residue}},${{r.resname}},${{r.chain}},${{r.resnum}},${{r.unbound.toFixed(2)}},${{r.bound.toFixed(2)}},${{r.delta.toFixed(2)}},${{r.burial.toFixed(4)}}`
                ).join('\\n');
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const link = document.createElement('a');
            link.download = 'sasa_data_table.csv';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        // Event listeners
        document.getElementById('searchInput').addEventListener('input', filterTable);

        // Initial render
        renderTable();
    </script>
</body>
</html>
"""
    
    return html



def generate_sasa_dashboard(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "SASA Analysis Dashboard"
) -> str:
    """
    Generate comprehensive SASA analysis dashboard with all visualizations.
    
    Creates a complete dashboard combining multiple plots:
    - Summary statistics cards
    - Burial distribution (toggleable scatter/histogram)
    - Top interface residues comparison
    - Chain-by-chain analysis
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Dashboard title
    
    Returns:
        HTML string with complete dashboard
    
    Example:
        >>> html = generate_sasa_dashboard(sasa_results)
        >>> with open('dashboard.html', 'w') as f:
        ...     f.write(html)
    """
    
    # Miller et al. (1987) maximum ASA values (Å²) used for RSA computation
    # Source: Miller et al. 1987 / Wikipedia "Relative accessible surface area"
    _MILLER_MAX_ASA = {
        'ALA': 113, 'ARG': 241, 'ASN': 158, 'ASP': 151, 'CYS': 140,
        'GLN': 189, 'GLU': 183, 'GLY': 85,  'HIS': 194, 'ILE': 182,
        'LEU': 180, 'LYS': 211, 'MET': 204, 'PHE': 218, 'PRO': 143,
        'SER': 122, 'THR': 146, 'TRP': 259, 'TYR': 229, 'VAL': 160,
    }

    # Prepare data for various plots
    data_points = []
    for res_id, values in sasa_results.items():
        resname, chain, resnum = _parse_residue_id(res_id)

        try:
            resnum_int = int(resnum)
        except:
            resnum_int = 0

        # Compute RSA using Miller et al. max ASA values
        _max_asa = _MILLER_MAX_ASA.get(resname.upper())
        if _max_asa and _max_asa > 0:
            _rsa_u = values['unbound'] / _max_asa
            _rsa_b = values['bound'] / _max_asa
            # Levy (2010) classification
            if _rsa_u >= 0.25 and _rsa_b < 0.25:
                _levy = 'core'
            elif _rsa_u < 0.25 and _rsa_b < 0.25:
                _levy = 'support'
            elif _rsa_u >= 0.25 and _rsa_b >= 0.25:
                _levy = 'rim'
            else:
                _levy = 'other'
        else:
            _rsa_u = None
            _rsa_b = None
            _levy = 'unknown'

        data_points.append({
            "residue": res_id,
            "resname": resname,
            "chain": chain,
            "resnum": resnum_int,
            "burial": values['buried_fraction'],
            "delta": values['delta'],
            "unbound": values['unbound'],
            "bound": values['bound'],
            "rsa_unbound": round(_rsa_u, 4) if _rsa_u is not None else None,
            "rsa_bound":   round(_rsa_b, 4) if _rsa_b is not None else None,
            "levy_class":  _levy,
        })
    
    # Calculate statistics
    total_residues = len(data_points)
    interface_residues = sum(1 for d in data_points if d['burial'] >= 0.5)
    highly_buried = sum(1 for d in data_points if d['burial'] >= 0.75)
    total_buried_sasa = sum(d['delta'] for d in data_points)
    avg_burial = sum(d['burial'] for d in data_points) / len(data_points) * 100 if data_points else 0
    num_chains = len(set(d['chain'] for d in data_points))
    
    # Get top interface residues
    top_interface = sorted(data_points, key=lambda x: x['delta'], reverse=True)[:20]
    
    # Organize by chain for comparison
    chain_data = defaultdict(list)
    for dp in sorted(data_points, key=lambda x: (x['chain'], x['resnum'])):
        chain_data[dp['chain']].append(dp)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #ffffff 0%, #ffffff 100%);
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 36px;
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .header p {{
            font-size: 16px;
            color: #555;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .panel {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .panel-full {{
            grid-column: 1 / -1;
        }}
        .panel h2 {{
            color: #2c3e50;
            margin: 0 0 20px 0;
            font-size: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .panel h2::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
            text-align: center;
        }}
        .stat-card.red {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-card.green {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
        .stat-card.orange {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
        .stat-label {{
            font-size: 11px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 15px;
        }}
        .chart-container.tall {{
            height: 500px;
        }}
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 13px;
            transition: transform 0.2s;
        }}
        .btn:hover {{
            transform: translateY(-2px);
        }}
        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}
        .legend-box {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Solvent Accessible Surface Area Analysis</h1>
    </div>
    
    <div class="dashboard-grid">
        <!-- Summary Statistics Panel -->
        <div class="panel panel-full">
            <h2> Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Residues</div>
                    <div class="stat-value">{total_residues}</div>
                </div>
                <div class="stat-card red">
                    <div class="stat-label">Interface Residues</div>
                    <div class="stat-value">{interface_residues}</div>
                </div>
                <div class="stat-card orange">
                    <div class="stat-label">Highly Buried</div>
                    <div class="stat-value">{highly_buried}</div>
                </div>
                <div class="stat-card blue">
                    <div class="stat-label">Total Buried SASA</div>
                    <div class="stat-value">{total_buried_sasa:.0f} (Å²)</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-label">Avg Burial</div>
                    <div class="stat-value">{avg_burial:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Chains Analyzed</div>
                    <div class="stat-value">{num_chains}</div>
                </div>
            </div>
        </div>
        
        <!-- Burial Distribution Panel -->
        <div class="panel">
            <h2> Burial Distribution</h2>
            <div class="controls">
                <button class="btn" onclick="toggleBurialPlotType()">Toggle View</button>
                <button class="btn" onclick="downloadChart('burialChart', 'burial_distribution.png')">💾 Save</button>
            </div>
            <div class="chart-container tall">
                <canvas id="burialChart"></canvas>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background: #e74c3c;"></div>
                    <span>Highly buried (≥75%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #e67e22;"></div>
                    <span>Moderate (50-75%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #f39c12;"></div>
                    <span>Slight (25-50%)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #3498db;"></div>
                    <span>Exposed (<25%)</span>
                </div>
            </div>
        </div>
        
        <!-- Top Interface Residues Panel -->
        <div class="panel">
            <h2> Top Interface Residues</h2>
            <div class="controls">
                <button class="btn" onclick="downloadChart('topResiduesChart', 'top_interface.png')">💾 Save</button>
            </div>
            <div class="chart-container tall">
                <canvas id="topResiduesChart"></canvas>
            </div>
        </div>
        
        <!-- Chain Comparison Panel -->
        <div class="panel panel-full">
            <h2> Chain-by-Chain Analysis</h2>
            <div class="controls">
                <button class="btn" onclick="downloadChart('chainChart', 'chain_analysis.png')">💾 Save</button>
            </div>
            <div class="chart-container">
                <canvas id="chainChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Data from Python
        const dataPoints = {json.dumps(data_points)};
        const topInterface = {json.dumps(top_interface)};
        const chainData = {json.dumps(dict(chain_data))};
        
        let burialChart, topResiduesChart, chainChart;
        let burialPlotType = 'histogram';
        
        function getColorByBurial(burial) {{
            if (burial >= 0.75) return 'rgba(231, 76, 60, 0.8)';
            if (burial >= 0.50) return 'rgba(230, 126, 34, 0.8)';
            if (burial >= 0.25) return 'rgba(243, 156, 18, 0.8)';
            return 'rgba(52, 152, 219, 0.8)';
        }}
        
        function createBurialChart() {{
            const ctx = document.getElementById('burialChart').getContext('2d');
            if (burialChart) burialChart.destroy();
            
            if (burialPlotType === 'histogram') {{
                const bins = new Array(20).fill(0);
                dataPoints.forEach(p => {{
                    const binIndex = Math.min(19, Math.floor(p.burial * 20));
                    bins[binIndex]++;
                }});
                
                const labels = [];
                const colors = [];
                for (let i = 0; i < 20; i++) {{
                    labels.push(`${{i * 5}}-${{(i + 1) * 5}}%`);
                    colors.push(getColorByBurial(i * 0.05));
                }}
                
                burialChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: [{{
                            data: bins,
                            backgroundColor: colors,
                            borderColor: colors.map(c => c.replace('0.8', '1')),
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ title: {{ display: true, text: 'Burial Fraction (%)', font: {{ size: 14, weight: 'bold' }} }} }},
                            y: {{ title: {{ display: true, text: 'Number of Residues', font: {{ size: 14, weight: 'bold' }} }}, beginAtZero: true }}
                        }}
                    }}
                }});
            }} else {{
                const scatterData = dataPoints.map((p, idx) => ({{ x: idx, y: p.burial * 100, residue: p.residue, delta: p.delta }}));
                const colors = scatterData.map(p => getColorByBurial(p.y / 100));
                
                burialChart = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [
                            {{ data: scatterData, backgroundColor: colors, pointRadius: 4 }},
                            {{ label: 'Interface (50%)', data: [{{x: 0, y: 50}}, {{x: dataPoints.length, y: 50}}], borderColor: 'red', borderDash: [10, 5], pointRadius: 0, type: 'line' }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: true }} }},
                        scales: {{
                            x: {{ title: {{ display: true, text: 'Residue Index', font: {{ size: 14, weight: 'bold' }} }} }},
                            y: {{ title: {{ display: true, text: 'Burial %', font: {{ size: 14, weight: 'bold' }} }}, min: 0, max: 100 }}
                        }}
                    }}
                }});
            }}
        }}
        
        function createTopResiduesChart() {{
            const ctx = document.getElementById('topResiduesChart').getContext('2d');
            if (topResiduesChart) topResiduesChart.destroy();
            
            const labels = topInterface.map(r => `${{r.chain}}:${{r.resnum}}`);
            
            topResiduesChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [
                        {{ label: 'Unbound', data: topInterface.map(r => r.unbound), backgroundColor: 'rgba(52, 152, 219, 0.8)' }},
                        {{ label: 'Bound', data: topInterface.map(r => r.bound), backgroundColor: 'rgba(230, 126, 34, 0.8)' }}
                    ]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'SASA (Å²)', font: {{ size: 14, weight: 'bold' }} }} }}
                    }}
                }}
            }});
        }}
        
        function createChainChart() {{
            const ctx = document.getElementById('chainChart').getContext('2d');
            if (chainChart) chainChart.destroy();
            
            const chains = Object.keys(chainData).sort();
            const interfaceCounts = chains.map(c => chainData[c].filter(r => r.burial >= 0.5).length);
            const avgBurials = chains.map(c => {{
                const avg = chainData[c].reduce((sum, r) => sum + r.burial, 0) / chainData[c].length;
                return avg * 100;
            }});
            
            chainChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: chains,
                    datasets: [
                        {{ label: 'Interface Residues', data: interfaceCounts, backgroundColor: 'rgba(231, 76, 60, 0.8)', yAxisID: 'y' }},
                        {{ label: 'Avg Burial %', data: avgBurials, backgroundColor: 'rgba(52, 152, 219, 0.8)', yAxisID: 'y1', type: 'line' }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ type: 'linear', position: 'left', title: {{ display: true, text: 'Interface Residues' }} }},
                        y1: {{ type: 'linear', position: 'right', min: 0, max: 100, title: {{ display: true, text: 'Avg Burial %' }}, grid: {{ drawOnChartArea: false }} }}
                    }}
                }}
            }});
        }}
        
        function toggleBurialPlotType() {{
            burialPlotType = burialPlotType === 'histogram' ? 'scatter' : 'histogram';
            createBurialChart();
        }}

        function downloadChart(chartId, filename) {{
            const canvas = document.getElementById(chartId);
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}
        
        // Initialize
        createBurialChart();
        createTopResiduesChart();
        createChainChart();

    </script>
</body>
</html>
"""
    
    return html

def generate_levy_classification(
    sasa_results: Dict[str, Dict[str, float]],
    title: str = "Levy (2010) Interface Classification"
) -> str:
    """
    Generate a scatter plot of RSA_bound vs RSA_unbound, coloured by Levy (2010) class.

    Residues are classified using Miller et al. max ASA values:
        Core    : RSAu >= 0.25 and RSAb < 0.25
        Support : RSAu <  0.25 and RSAb < 0.25
        Rim     : RSAu >= 0.25 and RSAb >= 0.25

    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        title: Page title

    Returns:
        HTML string (complete page via html_page())
    """

    _MILLER_MAX_ASA = {
        'ALA': 113, 'ARG': 241, 'ASN': 158, 'ASP': 151, 'CYS': 140,
        'GLN': 189, 'GLU': 183, 'GLY': 85,  'HIS': 194, 'ILE': 182,
        'LEU': 180, 'LYS': 211, 'MET': 204, 'PHE': 218, 'PRO': 143,
        'SER': 122, 'THR': 146, 'TRP': 259, 'TYR': 229, 'VAL': 160,
    }

    data_points = []
    counts = {'core': 0, 'support': 0, 'rim': 0, 'other': 0}

    for res_id, values in sasa_results.items():
        resname, chain, resnum = _parse_residue_id(res_id)
        _max_asa = _MILLER_MAX_ASA.get(resname.upper())
        if _max_asa and _max_asa > 0:
            rsa_u = values['unbound'] / _max_asa
            rsa_b = values['bound']   / _max_asa
            if rsa_u >= 0.25 and rsa_b < 0.25:
                levy = 'core'
            elif rsa_u < 0.25 and rsa_b < 0.25:
                levy = 'support'
            elif rsa_u >= 0.25 and rsa_b >= 0.25:
                levy = 'rim'
            else:
                levy = 'other'
        else:
            rsa_u = None
            rsa_b = None
            levy  = 'other'

        if rsa_u is not None:
            counts[levy] = counts.get(levy, 0) + 1
            data_points.append({
                'residue':   res_id,
                'resname':   resname,
                'chain':     chain,
                'resnum':    resnum,
                'rsa_bound':   round(rsa_b, 4),
                'rsa_unbound': round(rsa_u, 4),
                'levy_class':  levy,
                'burial':      round(values['buried_fraction'], 4),
            })

    total = len(data_points)

    extra_css = """
        .levy-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin: 18px 0;
        }
        .levy-card {
            padding: 14px;
            border-radius: 8px;
            color: white;
            text-align: center;
        }
        .levy-card.core    { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
        .levy-card.support { background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); }
        .levy-card.rim     { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
        .levy-card.other   { background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%); }
        .levy-label { font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; }
        .levy-value { font-size: 26px; font-weight: bold; margin-top: 4px; }
        .controls {
            margin: 16px 0;
            padding: 14px 18px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 13px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .chart-container {
            position: relative;
            height: 520px;
            margin: 16px 0;
        }
        .legend {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 14px;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 8px;
            align-items: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .legend-box {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            flex-shrink: 0;
        }
        .quadrant-desc {
            font-size: 12px;
            color: #666;
            margin: 6px 0 0 0;
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="levy-stats">
            <div class="levy-card core">
                <div class="levy-label">Core</div>
                <div class="levy-value">{counts['core']}</div>
            </div>
            <div class="levy-card support">
                <div class="levy-label">Support</div>
                <div class="levy-value">{counts['support']}</div>
            </div>
            <div class="levy-card rim">
                <div class="levy-label">Rim</div>
                <div class="levy-value">{counts['rim']}</div>
            </div>
            <div class="levy-card other">
                <div class="levy-label">Other / No RSA</div>
                <div class="levy-value">{counts.get('other', 0)}</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn" onclick="downloadSnapshot()">📸 Save PNG</button>
            <button class="btn" onclick="downloadData()">💾 Download CSV</button>
        </div>

        <div class="chart-container">
            <canvas id="levyChart"></canvas>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-box" style="background: rgba(231,76,60,0.85);"></div>
                <span><strong>Core</strong> — RSA<sub>u</sub> ≥ 0.25, RSA<sub>b</sub> &lt; 0.25</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: rgba(155,89,182,0.85);"></div>
                <span><strong>Support</strong> — RSA<sub>u</sub> &lt; 0.25, RSA<sub>b</sub> &lt; 0.25</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: rgba(46,204,113,0.85);"></div>
                <span><strong>Rim</strong> — RSA<sub>u</sub> ≥ 0.25, RSA<sub>b</sub> ≥ 0.25</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: rgba(149,165,166,0.85);"></div>
                <span><strong>Other</strong> — RSA<sub>u</sub> &lt; 0.25, RSA<sub>b</sub> ≥ 0.25</span>
            </div>
        </div>

        <div class="info-box">
            <p><strong>RSA</strong> = SASA / max SASA (Miller et al. 1987 values)</p>
            <p>Levy E.D. (2010) <em>J. Mol. Biol.</em> 403(4): 660–670. DOI:10.1016/j.jmb.2010.09.028</p>
            <p>Total residues with RSA data: {total}</p>
        </div>
    </div>

    <script>
        const dataPoints = {json.dumps(data_points)};
        let levyChart = null;

        const classColors = {{
            core:    'rgba(231, 76,  60,  0.85)',
            support: 'rgba(155, 89,  182, 0.85)',
            rim:     'rgba(46,  204, 113, 0.85)',
            other:   'rgba(149, 165, 166, 0.85)'
        }};
        const classLabels = {{
            core: 'Core', support: 'Support', rim: 'Rim', other: 'Other'
        }};

        function buildChart() {{
            const ctx = document.getElementById('levyChart').getContext('2d');
            if (levyChart) levyChart.destroy();

            const groups = {{}};
            dataPoints.forEach(p => {{
                const cls = p.levy_class || 'other';
                if (!groups[cls]) groups[cls] = [];
                groups[cls].push({{
                    x: p.rsa_bound,
                    y: p.rsa_unbound,
                    residue: p.residue,
                    resname: p.resname,
                    chain: p.chain,
                    resnum: p.resnum
                }});
            }});

            const classOrder = ['core', 'support', 'rim', 'other'];
            const datasets = classOrder
                .filter(cls => groups[cls] && groups[cls].length > 0)
                .map(cls => ({{
                    label: classLabels[cls] + ' (' + groups[cls].length + ')',
                    data: groups[cls],
                    backgroundColor: classColors[cls],
                    borderColor: classColors[cls].replace('0.85', '1'),
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 8
                }}));

            levyChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true, position: 'top',
                            labels: {{ font: {{ size: 13 }}, padding: 18 }} }},
                        tooltip: {{
                            callbacks: {{
                                label: (item) => {{
                                    const pt = item.raw;
                                    return `${{pt.residue}}: RSAb=${{item.parsed.x.toFixed(3)}}, RSAu=${{item.parsed.y.toFixed(3)}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'RSA bound (RSAb)', font: {{ size: 14, weight: 'bold' }} }},
                            min: 0, suggestedMax: 1.2,
                            ticks: {{ callback: v => v.toFixed(2) }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'RSA unbound (RSAu)', font: {{ size: 14, weight: 'bold' }} }},
                            min: 0, suggestedMax: 1.2,
                            ticks: {{ callback: v => v.toFixed(2) }}
                        }}
                    }}
                }}
            }});
        }}

        function downloadSnapshot() {{
            const canvas = document.getElementById('levyChart');
            const link = document.createElement('a');
            link.download = 'levy_classification.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function downloadData() {{
            const csv = 'residue,resname,chain,resnum,rsa_unbound,rsa_bound,levy_class,burial_fraction\\n' +
                dataPoints.map(p =>
                    `${{p.residue}},${{p.resname}},${{p.chain}},${{p.resnum}},${{p.rsa_unbound}},${{p.rsa_bound}},${{p.levy_class}},${{p.burial}}`
                ).join('\\n');
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const link = document.createElement('a');
            link.download = 'levy_classification.csv';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        buildChart();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"])


def visualize_sasa(
    sasa_results: Dict[str, Dict[str, float]],
    viz_type: str = 'all',
    output_file: str = 'sasa_analysis'
) -> Dict[str, str]:
    """
    Generate SASA visualizations (main function for app.py integration).
    
    This is the main entry point for SASA visualization, following the same
    pattern as visualize_geometry() in geometry_viz.py.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
            Expected format: {
                'residue_id': {
                    'unbound': float,
                    'bound': float,
                    'delta': float,
                    'buried_fraction': float
                }
            }
        viz_type: Type of visualization to generate:
            - 'all': Generate all visualizations + dashboard
            - 'burial': Only burial distribution plot
            - 'comparison': Only bound vs unbound comparison
            - 'heatmap': Only chain-by-chain heatmap
            - 'table': Only interface residues table
            - 'dashboard': Only dashboard with all plots
        output_file: Base path for output files (without extension)
    
    Returns:
        Dictionary mapping visualization type to file path
    
    Example:
        >>> from sasa_analysis import compare_bound_unbound_sasa
        >>> from sasa_viz import visualize_sasa
        >>> 
        >>> # Run SASA analysis
        >>> sasa_results = compare_bound_unbound_sasa(complex, ["chain A", "chain B"])
        >>> 
        >>> # Generate all visualizations
        >>> files = visualize_sasa(sasa_results, viz_type='all', 
        ...                        output_file='/tmp/results/sasa')
        >>> 
        >>> print(files)
        >>> # {
        >>> #     'dashboard': '/tmp/results/sasa.html',
        >>> #     'burial': '/tmp/results/sasa_burial.html',
        >>> #     'comparison': '/tmp/results/sasa_comparison.html',
        >>> #     'heatmap': '/tmp/results/sasa_heatmap.html',
        >>> #     'table': '/tmp/results/sasa_table.html'
        >>> # }
        >>> 
        >>> # Or generate just the dashboard
        >>> dashboard_files = visualize_sasa(
        ...     sasa_results,
        ...     viz_type='dashboard',
        ...     output_file='results/sasa_dashboard'
        >>> )
    """
    
    result_files = {}
    
    # Generate requested visualizations
    if viz_type == 'all' or viz_type == 'dashboard':
        # Generate comprehensive dashboard
        dashboard_html = generate_sasa_dashboard(
            sasa_results,
            title="SASA Analysis Dashboard"
        )
        dashboard_path = f"{output_file}.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        result_files['dashboard'] = dashboard_path
    
    if viz_type == 'all' or viz_type == 'burial':
        # Generate burial plot
        burial_html = generate_burial_plot(
            sasa_results,
            title="Residue Burial Distribution"
        )
        burial_path = f"{output_file}_burial.html"
        with open(burial_path, 'w') as f:
            f.write(burial_html)
        result_files['burial'] = burial_path
    
    if viz_type == 'all' or viz_type == 'comparison':
        # Generate comparison plot
        comparison_html = generate_sasa_comparison(
            sasa_results,
            title="Bound vs Unbound SASA Comparison",
            top_n=30
        )
        comparison_path = f"{output_file}_comparison.html"
        with open(comparison_path, 'w') as f:
            f.write(comparison_html)
        result_files['comparison'] = comparison_path
    
    if viz_type == 'all' or viz_type == 'heatmap':
        # Generate heatmap
        heatmap_html = generate_chain_heatmap(
            sasa_results,
            title="Residue Burial Heatmap by Chain"
        )
        heatmap_path = f"{output_file}_heatmap.html"
        with open(heatmap_path, 'w') as f:
            f.write(heatmap_html)
        result_files['heatmap'] = heatmap_path
    
    if viz_type == 'all' or viz_type == 'table':
        # Generate interface table
        table_html = generate_interface_table(
            sasa_results,
            title="SASA Data Table",
            burial_threshold=0.5
        )
        table_path = f"{output_file}_table.html"
        with open(table_path, 'w') as f:
            f.write(table_html)
        result_files['table'] = table_path

    if viz_type == 'all' or viz_type == 'levy':
        # Generate Levy (2010) classification scatter plot
        levy_html = generate_levy_classification(
            sasa_results,
            title="Levy (2010) Interface Classification"
        )
        levy_path = f"{output_file}_levy.html"
        with open(levy_path, 'w') as f:
            f.write(levy_html)
        result_files['levy'] = levy_path

    return result_files


# Example usage function
def example_usage():
    """
    Demonstrate the complete SASA visualization workflow.
    
    This example shows how to use sasa_viz.py in a typical workflow.
    """
    # Note: These imports would be at the top of your actual script
    from ..pdbparser import parse_pdb
    from ..sasa_analysis import compare_bound_unbound_sasa
    
    # Load structure
    cx = parse_pdb("complex.pdb")
    
    # Run SASA analysis
    sasa_results = compare_bound_unbound_sasa(
        cx,
        chain_exprs=["A", "B"]
    )
    
    # Generate all visualizations
    files = visualize_sasa(
        sasa_results,
        viz_type='all',
        output_file='results/sasa_analysis'
    )
    
    print("Generated files:")
    for viz_type, path in files.items():
        print(f"  {viz_type}: {path}")
    
    # Or generate just the dashboard
    dashboard_files = visualize_sasa(
        sasa_results,
        viz_type='dashboard',
        output_file='results/sasa_dashboard'
    )
    
    # Or generate specific plots
    specific_files = visualize_sasa(
        sasa_results,
        viz_type='comparison',
        output_file='results/sasa_comparison'
    )
    
    return files

