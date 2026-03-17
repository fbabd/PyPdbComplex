from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import json
import math

from ..models import Complex
from ..geometry import (
    BackboneAngles,
    SidechainChi,
    BendAngle,
    compute_backbone_angles,
    compute_sidechain_chis,
    compute_ca_bend_angles,
    analyze_geometry
)
from ._base import html_page


def generate_ramachandran_plot(
    backbone_angles: List,  # List[BackboneAngles]
    title: str = "Ramachandran Plot",
    highlight_outliers: bool = True,
    show_regions: bool = True
) -> str:
    """
    Generate interactive Ramachandran plot (φ vs ψ) with CORRECTED region definitions.
    
    Args:
        backbone_angles: List of BackboneAngles
        title: Plot title
        highlight_outliers: Highlight unusual conformations
        show_regions: Show allowed/favored region overlays
    
    Returns:
        HTML string with interactive plot
    
    """
    
    # Filter valid phi/psi pairs
    valid_points = []
    for ang in backbone_angles:
        if ang.phi is not None and ang.psi is not None:
            valid_points.append({
                "phi": ang.phi,
                "psi": ang.psi,
                "residue": ang.residue_id,
                "resname": ang.resname,
                "chain": ang.chain_id,
                "resseq": ang.resseq
            })
    
    extra_css = """
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
        .btn:hover {
            transform: translateY(-2px);
        }
        .chart-container {
            position: relative;
            margin: 20px 0;
        }
        #ramaChart {
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
                <div class="stat-value">{len(valid_points)}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-label">α-Helix</div>
                <div class="stat-value" id="alphaCount">-</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="stat-label">β-Sheet</div>
                <div class="stat-value" id="betaCount">-</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="stat-label">Other</div>
                <div class="stat-value" id="otherCount">-</div>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <input type="checkbox" id="showRegions" checked>
                <label for="showRegions">Show allowed regions</label>
            </div>

            <div class="control-group">
                <input type="checkbox" id="colorByChain">
                <label for="colorByChain">Color by chain</label>
            </div>

            <div class="size-controls">
                <label for="sizeSlider">Plot Size:</label>
                <input type="range" id="sizeSlider" min="400" max="1000" value="700" step="50">
                <span id="sizeValue">700px</span>
            </div>

            <button class="btn" onclick="downloadSnapshot()">📸 Save Snapshot</button>
            <button class="btn" onclick="downloadData()">💾 Export Data</button>
        </div>

        <div class="legend">
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(255, 0, 0, 0.3);"></div>
                    <span>α-Helix region</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(0, 0, 255, 0.3);"></div>
                    <span>β-Sheet region</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(0, 255, 0, 0.3);"></div>
                    <span>Left-handed helix</span>
                </div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="ramaChart"></canvas>
        </div>

        <div class="info-box">
            <p><strong>α-Helix:</strong> φ = [-90, -30], ψ = [-70, 10]</p>
            <p><strong>β-Sheet:</strong> φ = [-180°, -90°], ψ = [90°, 180°]</p>
            <p><strong>Left-handed helix:</strong> φ = [30, 100], ψ = [0, 80]</p>
        </div>

        <div>
        <p><strong> References: </strong> </p>
            <ul>
               <li> G.N. Ramachandran, V. Sasisekharan. Conformation of polypeptides and proteins. (Advances in Protein Chemistry, 1968) </li>
                <li> Davis et al. MolProbity: Nucleic Acids Research, 2007.</li>
            </ul>
        </div>
    </div>

    <script>
        const dataPoints = {json.dumps(valid_points)};
        let currentChart = null;
        let chartSize = 700;

        // Chain colors
        const chainColors = {{
            'H': 'rgba(231, 76, 60, 0.6)',
            'L': 'rgba(52, 152, 219, 0.6)',
            'A': 'rgba(46, 204, 113, 0.6)',
            'B': 'rgba(241, 196, 15, 0.6)',
            'C': 'rgba(155, 89, 182, 0.6)',
        }};

        // FIXED: Proper range-based classification
        function classifyResidue(phi, psi) {{
            // Alpha helix region: φ ∈ [-90, -30], ψ ∈ [-70, 10]
            if (phi >= -90 && phi <= -30 && psi >= -70 && psi <= 10) {{
                return 'alpha';
            }}
            // Beta sheet region: φ ∈ [-180, -90], ψ ∈ [90, 180]
            else if (phi >= -180 && phi <= -90 && psi >= 90 && psi <= 180) {{
                return 'beta';
            }}
            // Left-handed helix (rare, mostly Gly): φ ∈ [30, 100], ψ ∈ [0, 80]
            else if (phi >= 30 && phi <= 100 && psi >= 0 && psi <= 80) {{
                return 'left_helix';
            }}
            return 'other';
        }}

        function updateStats() {{
            let alpha = 0, beta = 0, other = 0;
            dataPoints.forEach(p => {{
                const type = classifyResidue(p.phi, p.psi);
                if (type === 'alpha') alpha++;
                else if (type === 'beta') beta++;
                else other++;
            }});
            document.getElementById('alphaCount').textContent = alpha;
            document.getElementById('betaCount').textContent = beta;
            document.getElementById('otherCount').textContent = other;
        }}

        function createChart() {{
            const ctx = document.getElementById('ramaChart').getContext('2d');
            const showRegions = document.getElementById('showRegions').checked;
            const colorByChain = document.getElementById('colorByChain').checked;

            if (currentChart) {{
                currentChart.destroy();
            }}

            // Set canvas size
            const canvas = document.getElementById('ramaChart');
            canvas.width = chartSize;
            canvas.height = chartSize;
            pointRad = parseInt(chartSize/150);

            // Prepare data
            const scatterData = dataPoints.map(p => ({{
                x: p.phi,
                y: p.psi,
                residue: p.residue,
                chain: p.chain
            }}));

            // Color points
            const pointColors = colorByChain
                ? scatterData.map(p => chainColors[p.chain] || 'rgba(100, 100, 100, 0.6)')
                : 'rgba(52, 152, 219, 0.6)';

            const datasets = [{{
                label: 'Residues',
                data: scatterData,
                backgroundColor: pointColors,
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1,
                pointRadius: pointRad,
                pointHoverRadius: 6
            }}];

            // Add region boxes if enabled - FIXED COORDINATES
            if (showRegions) {{
                // Alpha helix - CORRECTED
                datasets.push({{
                    label: 'α-Helix',
                    data: [
                        {{x: -90, y: -70}},
                        {{x: -30, y: -70}},
                        {{x: -30, y: -10}},
                        {{x: -90, y: -10}},
                        {{x: -90, y: -70}}
                    ],
                    showLine: true,
                    fill: true,
                    backgroundColor: 'rgba(255, 0, 0, 0.1)',
                    borderColor: 'rgba(255, 0, 0, 0.5)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0
                }});

                // Beta sheet - CORRECTED
                datasets.push({{
                    label: 'β-Sheet',
                    data: [
                        {{x: -180, y: 90}},
                        {{x: -90, y: 90}},
                        {{x: -90, y: 180}},
                        {{x: -180, y: 180}},
                        {{x: -180, y: 90}}
                    ],
                    showLine: true,
                    fill: true,
                    backgroundColor: 'rgba(0, 0, 255, 0.1)',
                    borderColor: 'rgba(0, 0, 255, 0.5)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0
                }});

                // Left-handed helix (optional, rare)
                datasets.push({{
                    label: 'Left-handed helix',
                    data: [
                        {{x: 30, y: 0}},
                        {{x: 100, y: 0}},
                        {{x: 100, y: 80}},
                        {{x: 30, y: 80}},
                        {{x: 30, y: 0}}
                    ],
                    showLine: true,
                    fill: true,
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderColor: 'rgba(0, 255, 0, 0.5)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0
                }});
            }}

            currentChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets }},
                options: {{
                    responsive: false,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    if (context.dataset.label === 'Residues') {{
                                        const point = context.raw;
                                        return [
                                            point.residue,
                                            `φ: ${{point.x.toFixed(1)}}°`,
                                            `ψ: ${{point.y.toFixed(1)}}°`
                                        ];
                                    }}
                                    return context.dataset.label;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'φ (phi) angle (°)',
                                font: {{ size: 14, weight: 'bold' }}
                            }},
                            min: -180,
                            max: 180,
                            ticks: {{ stepSize: 30 }},
                            grid: {{ color: 'rgba(0,0,0,0.1)' }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'ψ (psi) angle (°)',
                                font: {{ size: 14, weight: 'bold' }}
                            }},
                            min: -180,
                            max: 180,
                            ticks: {{ stepSize: 30 }},
                            grid: {{ color: 'rgba(0,0,0,0.1)' }}
                        }}
                    }}
                }}
            }});
        }}

        function downloadSnapshot() {{
            const canvas = document.getElementById('ramaChart');
            const link = document.createElement('a');
            link.download = 'ramachandran_plot.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function downloadData() {{
            const csv = 'residue,chain,resseq,phi,psi\\n' +
                dataPoints.map(p =>
                    `${{p.residue}},${{p.chain}},${{p.resseq}},${{p.phi}},${{p.psi}}`
                ).join('\\n');

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const link = document.createElement('a');
            link.download = 'ramachandran_data.csv';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        // Event listeners
        document.getElementById('showRegions').addEventListener('change', createChart);
        document.getElementById('colorByChain').addEventListener('change', createChart);

        document.getElementById('sizeSlider').addEventListener('input', (e) => {{
            chartSize = parseInt(e.target.value);
            document.getElementById('sizeValue').textContent = chartSize + 'px';
            createChart();
        }});

        // Initial render
        updateStats();
        createChart();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"])


def generate_chi_distribution(
    chi_angles: List[SidechainChi],
    title: str = "Sidechain Chi Angle Distribution"
) -> str:
    """
    Generate chi angle distribution histograms.
    
    Args:
        chi_angles: List of SidechainChi
        title: Plot title
    
    Returns:
        HTML string with interactive histograms
    """
    
    # Organize chi angles by type
    chi_data = {"chi1": [], "chi2": [], "chi3": [], "chi4": []}
    
    for chi in chi_angles:
        for chi_name, angle in chi.chis.items():
            if angle is not None:
                chi_data[chi_name].append({
                    "angle": angle,
                    "residue": chi.residue_id,
                    "resname": chi.resname
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
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
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
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <div class="size-controls">
                <label>Chart Height:</label>
                <input type="range" id="heightSlider" min="200" max="600" value="300" step="50">
                <span id="heightValue">300px</span>
            </div>
            <button class="btn" onclick="downloadAllSnapshots()">📸 Save All Plots</button>
        </div>

        <div class="info-box">
                <p><strong>Chi Angles:</strong> Sidechain dihedral angles that define rotamer conformations</p>
                <p>• Preferred values cluster around -180°, -60°, 60°, and 180° (gauche+, gauche-, trans)</p>
            </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h3>χ1 Distribution</h3>
                <div class="chart-container">
                    <canvas id="chi1Chart"></canvas>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #666;">
                    {len(chi_data['chi1'])} residues
                </div>
            </div>

            <div class="chart-card">
                <h3>χ2 Distribution</h3>
                <div class="chart-container">
                    <canvas id="chi2Chart"></canvas>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #666;">
                    {len(chi_data['chi2'])} residues
                </div>
            </div>

            <div class="chart-card">
                <h3>χ3 Distribution</h3>
                <div class="chart-container">
                    <canvas id="chi3Chart"></canvas>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #666;">
                    {len(chi_data['chi3'])} residues
                </div>
            </div>

            <div class="chart-card">
                <h3>χ4 Distribution</h3>
                <div class="chart-container">
                    <canvas id="chi4Chart"></canvas>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #666;">
                    {len(chi_data['chi4'])} residues
                </div>
            </div>



            </div>
        </div>
    </div>

    <script>
        const chiData = {json.dumps(chi_data)};
        let charts = {{}};
        let chartHeight = 300;

        function createHistogram(canvasId, data, label, color) {{
            const ctx = document.getElementById(canvasId).getContext('2d');

            if (charts[canvasId]) {{
                charts[canvasId].destroy();
            }}

            // Create bins
            const bins = new Array(36).fill(0);  // 36 bins of 10° each
            data.forEach(item => {{
                const angle = item.angle;
                const binIndex = Math.floor((angle + 180) / 10);
                bins[Math.max(0, Math.min(35, binIndex))]++;
            }});

            const labels = [];
            for (let i = 0; i < 36; i++) {{
                const angle = -180 + i * 10;
                labels.push(`${{angle}}°`);
            }}

            const canvas = document.getElementById(canvasId);
            canvas.height = chartHeight;

            charts[canvasId] = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: label,
                        data: bins,
                        backgroundColor: color,
                        borderColor: color.replace('0.6', '1'),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    return context[0].label;
                                }},
                                label: function(context) {{
                                    return `Count: ${{context.parsed.y}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Angle (°)' }},
                            ticks: {{
                                maxRotation: 45,
                                minRotation: 45,
                                autoSkip: true,
                                maxTicksLimit: 12
                            }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Count' }},
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}

        function renderAllCharts() {{
            createHistogram('chi1Chart', chiData.chi1, 'χ1', 'rgba(231, 76, 60, 0.6)');
            createHistogram('chi2Chart', chiData.chi2, 'χ2', 'rgba(52, 152, 219, 0.6)');
            createHistogram('chi3Chart', chiData.chi3, 'χ3', 'rgba(46, 204, 113, 0.6)');
            createHistogram('chi4Chart', chiData.chi4, 'χ4', 'rgba(155, 89, 182, 0.6)');
        }}

        function downloadAllSnapshots() {{
            ['chi1Chart', 'chi2Chart', 'chi3Chart', 'chi4Chart'].forEach(id => {{
                const canvas = document.getElementById(id);
                const link = document.createElement('a');
                link.download = `${{id}}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            }});
        }}

        document.getElementById('heightSlider').addEventListener('input', (e) => {{
            chartHeight = parseInt(e.target.value);
            document.getElementById('heightValue').textContent = chartHeight + 'px';
            renderAllCharts();
        }});

        // Initial render
        renderAllCharts();
    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"],
                     max_width=1600)


def generate_bend_angle_plot(
    bend_angles: List[BendAngle],
    title: str = "CA Trace Bend Angles"
) -> str:
    """
    Generate CA bend angle visualization along sequence.
    
    Args:
        bend_angles: List of BendAngle
        title: Plot title
    
    Returns:
        HTML string with interactive line plot
    """
    
    # Organize by chain
    chains_data = {}
    for bend in bend_angles:
        if bend.angle_deg is not None:
            if bend.chain_id not in chains_data:
                chains_data[bend.chain_id] = []
            chains_data[bend.chain_id].append({
                "resseq": bend.resseq_center,
                "angle": bend.angle_deg,
                "residue": bend.residue_id,
                "resname": bend.resname_center
            })
    
    # Sort by resseq
    for chain_id in chains_data:
        chains_data[chain_id].sort(key=lambda x: x["resseq"])
    
    extra_css = """
        .controls {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
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
        }
        .chart-container {
            position: relative;
            margin: 20px 0;
        }
        #bendChart {
            max-width: 100%;
        }
    """

    body = f"""
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <div class="control-group">
                <input type="checkbox" id="showBentLine" checked>
                <label for="showBentLine">Show bend threshold (140°)</label>
            </div>
            <div class="control-group">
                <input type="checkbox" id="showStraightLine" checked>
                <label for="showStraightLine">Show straight threshold (160°)</label>
            </div>

            <div class="size-controls">
                <label>Plot Height:</label>
                <input type="range" id="heightSlider" min="300" max="800" value="500" step="50">
                <span id="heightValue">500px</span>
            </div>

            <button class="btn" onclick="downloadSnapshot()">📸 Save Snapshot</button>
        </div>

        <div class="chart-container">
            <canvas id="bendChart"></canvas>
        </div>

        <div class="info-box">
                <p><strong>Interpretation:</strong></p>
                <p>• &lt;140°: Significantly bent region</p>
                <p>• 140-160°: Moderate curvature</p>
                <p>• &gt;160°: Extended/straight region</p>
            </div>

    </div>

    <script>
        const chainsData = {json.dumps(chains_data)};
        let currentChart = null;
        let chartHeight = 500;

        const chainColors = {{
            'H': 'rgba(231, 76, 60, 0.8)',
            'L': 'rgba(52, 152, 219, 0.8)',
            'A': 'rgba(46, 204, 113, 0.8)',
            'B': 'rgba(241, 196, 15, 0.8)',
            'C': 'rgba(155, 89, 182, 0.8)',
        }};

        function createChart() {{
            const ctx = document.getElementById('bendChart').getContext('2d');
            const showBentLine = document.getElementById('showBentLine').checked;
            const showStraightLine = document.getElementById('showStraightLine').checked;

            if (currentChart) {{
                currentChart.destroy();
            }}

            const canvas = document.getElementById('bendChart');
            canvas.height = chartHeight;

            // Create datasets for each chain
            const datasets = [];
            Object.keys(chainsData).forEach((chainId, idx) => {{
                const data = chainsData[chainId];
                datasets.push({{
                    label: `Chain ${{chainId}}`,
                    data: data.map(d => ({{ x: d.resseq, y: d.angle, residue: d.residue }})),
                    borderColor: chainColors[chainId] || `hsl(${{idx * 60}}, 70%, 50%)`,
                    backgroundColor: chainColors[chainId] || `hsl(${{idx * 60}}, 70%, 50%, 0.1)`,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    tension: 0.1,
                    fill: false
                }});
            }});

            // Add threshold lines
            if (showBentLine || showStraightLine) {{
                const allResseqs = [];
                Object.values(chainsData).forEach(chain => {{
                    chain.forEach(d => allResseqs.push(d.resseq));
                }});
                const minRes = Math.min(...allResseqs);
                const maxRes = Math.max(...allResseqs);

                if (showBentLine) {{
                    datasets.push({{
                        label: 'Bent threshold (140°)',
                        data: [{{x: minRes, y: 140}}, {{x: maxRes, y: 140}}],
                        borderColor: 'rgba(255, 0, 0, 0.5)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }});
                }}

                if (showStraightLine) {{
                    datasets.push({{
                        label: 'Straight threshold (160°)',
                        data: [{{x: minRes, y: 160}}, {{x: maxRes, y: 160}}],
                        borderColor: 'rgba(0, 255, 0, 0.5)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }});
                }}
            }}

            currentChart = new Chart(ctx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const point = context.raw;
                                    if (point.residue) {{
                                        return [
                                            point.residue,
                                            `Angle: ${{point.y.toFixed(1)}}°`
                                        ];
                                    }}
                                    return `${{context.dataset.label}}: ${{point.y}}°`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Residue Number',
                                font: {{ size: 14, weight: 'bold' }}
                            }},
                            type: 'linear'
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Bend Angle (°)',
                                font: {{ size: 14, weight: 'bold' }}
                            }},
                            min: 0,
                            max: 180,
                            ticks: {{ stepSize: 20 }}
                        }}
                    }}
                }}
            }});
        }}

        function downloadSnapshot() {{
            const canvas = document.getElementById('bendChart');
            const link = document.createElement('a');
            link.download = 'bend_angles.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        document.getElementById('showBentLine').addEventListener('change', createChart);
        document.getElementById('showStraightLine').addEventListener('change', createChart);
        document.getElementById('heightSlider').addEventListener('input', (e) => {{
            chartHeight = parseInt(e.target.value);
            document.getElementById('heightValue').textContent = chartHeight + 'px';
            createChart();
        }});

        createChart();
    </script>
"""
    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"],
                     max_width=1600)
        

def generate_geometry_dashboard(
    backbone_angles: List[BackboneAngles],
    sidechain_chis: List[SidechainChi],
    bend_angles: List[BendAngle],
    title: str = "Geometry Analysis Dashboard"
) -> str:
    """
    Generate a comprehensive dashboard with all geometry visualizations.
    
    Args:
        backbone_angles: List of BackboneAngles
        sidechain_chis: List of SidechainChi
        bend_angles: List of BendAngle
        title: Dashboard title
    
    Returns:
        HTML string with complete dashboard
    """
    
    # Prepare data for Ramachandran plot
    rama_points = []
    for ang in backbone_angles:
        if ang.phi is not None and ang.psi is not None:
            rama_points.append({
                "phi": ang.phi,
                "psi": ang.psi,
                "residue": ang.residue_id,
                "resname": ang.resname,
                "chain": ang.chain_id,
                "resseq": ang.resseq
            })
    
    # Prepare data for bend angles by chain
    bend_by_chain = {}
    for bend in bend_angles:
        if bend.angle_deg is not None:
            if bend.chain_id not in bend_by_chain:
                bend_by_chain[bend.chain_id] = []
            bend_by_chain[bend.chain_id].append({
                "resseq": bend.resseq_center,
                "angle": bend.angle_deg,
                "residue": bend.residue_id
            })
    
    # Prepare chi angle distributions
    chi_data = {"chi1": [], "chi2": [], "chi3": [], "chi4": [], "chi5": []}
    for chi in sidechain_chis:
        for chi_name, chi_val in chi.chis.items():
            if chi_val is not None and chi_name in chi_data:
                chi_data[chi_name].append(chi_val)
    
    # Count secondary structures (simplified)
    alpha_count = sum(1 for p in rama_points 
                  if -90 <= p["phi"] <= -30 and -75 <= p["psi"] <= -20)
    beta_count = sum(1 for p in rama_points 
                    if -180 <= p["phi"] <= -90 and 90 <= p["psi"] <= 180)
    other_count = len(rama_points) - alpha_count - beta_count
    
    # Count bent regions
    bent_count = sum(1 for bend in bend_angles if bend.angle_deg and bend.angle_deg < 140)
    straight_count = sum(1 for bend in bend_angles if bend.angle_deg and bend.angle_deg > 160)

    # --- Build per-residue table data ---
    _bend_lookup = {(b.chain_id, b.resseq_center): b.angle_deg for b in bend_angles if b.angle_deg is not None}
    _chi_lookup  = {(c.chain_id, c.resseq): c.chis for c in sidechain_chis}

    _geo_table = []
    for ang in backbone_angles:
        phi, psi = ang.phi, ang.psi
        if phi is not None and psi is not None:
            if   -90  <= phi <= -30  and -75 <= psi <= -20:  ss = "α-Helix"
            elif -180 <= phi <= -90  and  90 <= psi <= 180:  ss = "β-Sheet"
            elif  30  <= phi <= 100  and   0 <= psi <= 80:   ss = "Left Helix"
            else:                                              ss = "Other"
        else:
            ss = "—"
        _chis  = _chi_lookup.get((ang.chain_id, ang.resseq), {})
        _bend  = _bend_lookup.get((ang.chain_id, ang.resseq))
        _geo_table.append({
            "chain":    ang.chain_id,
            "resseq":   ang.resseq,
            "resname":  ang.resname,
            "phi":      round(phi,       1) if phi       is not None else None,
            "psi":      round(psi,       1) if psi       is not None else None,
            "omega":    round(ang.omega, 1) if ang.omega is not None else None,
            "ss":       ss,
            "bend_deg": round(_bend, 1) if _bend is not None else None,
            "chi1":     round(_chis.get("chi1"), 1) if _chis.get("chi1") is not None else None,
            "chi2":     round(_chis.get("chi2"), 1) if _chis.get("chi2") is not None else None,
            "chi3":     round(_chis.get("chi3"), 1) if _chis.get("chi3") is not None else None,
            "chi4":     round(_chis.get("chi4"), 1) if _chis.get("chi4") is not None else None,
        })
    geo_table_json = json.dumps(_geo_table)

    extra_css = """
        body {
            background: #ffffff;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 36px;
            margin: 0 0 10px 0;
            color: black;
        }
        .header p {
            font-size: 16px;
            opacity: 0.95;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .panel {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        .panel-full {
            grid-column: 1 / -1;
        }
        .panel h2 {
            color: #2c3e50;
            margin: 0 0 20px 0;
            font-size: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .panel h2::before {
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card.red {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .stat-card.blue {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .stat-card.green {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }
        .stat-card.orange {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        .stat-card {
            text-align: center;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin-top: 15px;
        }
        .chart-container.tall {
            height: 500px;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
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
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .legend {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
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
        }
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Tab navigation */
        .geo-tabs {
            display: flex;
            border-bottom: 2px solid #dee2e6;
            margin-bottom: 24px;
        }
        .geo-tab {
            padding: 10px 30px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            font-size: 15px;
            font-weight: 600;
            color: #6c757d;
            cursor: pointer;
            border-radius: 6px 6px 0 0;
            transition: color 0.15s, border-color 0.15s;
        }
        .geo-tab.active  { color: #667eea; border-bottom-color: #667eea; background: #f4f6ff; }
        .geo-tab:hover:not(.active) { color: #495057; background: #f8f9fa; }
        .geo-pane { display: none; }
        .geo-pane.active { display: block; }

        /* Data table */
        .dt-toolbar {
            display: flex; align-items: center; gap: 12px;
            padding: 0 0 14px 0; flex-wrap: wrap;
        }
        .dt-toolbar input {
            padding: 7px 13px; border: 1px solid #ced4da; border-radius: 6px;
            font-size: 13px; width: 240px; outline: none;
        }
        .dt-toolbar input:focus { border-color: #667eea; box-shadow: 0 0 0 2px rgba(102,126,234,.15); }
        .dt-dl-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; padding: 7px 18px; border-radius: 6px;
            font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity .15s;
        }
        .dt-dl-btn:hover { opacity: 0.88; }
        .dt-scroll {
            overflow-x: auto; border-radius: 8px; border: 1px solid #e9ecef;
            max-height: 600px; overflow-y: auto;
        }
        .dt-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .dt-table thead th {
            background: #f4f6ff; color: #2c3e50; font-weight: 700;
            padding: 10px 14px; text-align: left; position: sticky; top: 0; z-index: 1;
            border-bottom: 2px solid #dee2e6; white-space: nowrap;
            cursor: pointer; user-select: none;
        }
        .dt-table thead th:hover { background: #e8ecff; }
        .dt-table thead th .si { color: #aaa; margin-left: 4px; font-size: 11px; }
        .dt-table tbody tr:nth-child(even) { background: #f9f9fb; }
        .dt-table tbody tr:hover { background: #eef0ff; }
        .dt-table tbody td { padding: 8px 14px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }
        .dt-no-rows { padding: 24px; text-align: center; color: #888; font-style: italic; }
        .ss-badge {
            display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 700; color: white;
        }
        .ss-alpha   { background: #e74c3c; }
        .ss-beta    { background: #3498db; }
        .ss-left    { background: #2ecc71; }
        .ss-other   { background: #95a5a6; }
        .ss-dash    { background: #bdc3c7; color: #555; }
    """

    body = f"""
    <div class="header">
        <h1> {title}</h1>
    </div>

    <div class="geo-tabs">
        <button class="geo-tab active" id="geo-tab-charts" onclick="showGeoTab('charts')">📊 Charts</button>
        <button class="geo-tab" id="geo-tab-table"  onclick="showGeoTab('table')">📋 Data Table</button>
    </div>

    <!-- Charts pane -->
    <div id="geo-pane-charts" class="geo-pane active">
    <div class="dashboard-grid">
        <!-- Summary Statistics Panel -->
        <div class="panel panel-full">
            <h2> Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card ">
                    <div class="stat-label">Chains</div>
                    <div class="stat-value">{len(bend_by_chain)}</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-label">Total Residues</div>
                    <div class="stat-value">{len(rama_points)}</div>
                </div>
                <div class="stat-card red">
                    <div class="stat-label">α-Helix</div>
                    <div class="stat-value">{alpha_count}</div>
                </div>
                <div class="stat-card blue">
                    <div class="stat-label">β-Sheet</div>
                    <div class="stat-value">{beta_count}</div>
                </div>
                <!--div class="stat-card green">
                    <div class="stat-label">Other</div>
                    <div class="stat-value">{other_count}</div>
                </div-->
                <div class="stat-card orange">
                    <div class="stat-label">Bent Regions</div>
                    <div class="stat-value">{bent_count}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Straight Regions</div>
                    <div class="stat-value">{straight_count}</div>
                </div>
                <div class="stat-card blue">
                    <div class="stat-label">Chi Angles</div>
                    <div class="stat-value">{len(sidechain_chis)}</div>
                </div>

            </div>
        </div>

        <!-- Ramachandran Plot Panel -->
        <div class="panel">
            <h2> Ramachandran Plot</h2>
            <div class="controls">
                <button class="btn" onclick="toggleRamaRegions()">Toggle Regions</button>
                <button class="btn" onclick="downloadChart('ramaChart', 'ramachandran.png')">💾 Save</button>
            </div>
            <div class="chart-container tall">
                <canvas id="ramaChart"></canvas>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(255, 0, 0, 0.3);"></div>
                    <span>α-Helix</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(0, 0, 255, 0.3);"></div>
                    <span>β-Sheet</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: rgba(0, 255, 0, 0.3);"></div>
                    <span>Left Helix</span>
                </div>
            </div>
        </div>

        <!-- Bend Angles Panel -->
        <div class="panel">
            <h2> Backbone Bend Angles</h2>
            <div class="controls">
                <button class="btn" onclick="toggleBendThresholds()">Toggle Thresholds</button>
                <button class="btn" onclick="downloadChart('bendChart', 'bend_angles.png')">💾 Save</button>
            </div>
            <div class="chart-container tall">
                <canvas id="bendChart"></canvas>
            </div>
            <div class="info-box">
                <p><strong>Interpretation:</strong></p>
                <p>• &lt;140°: Significantly bent region</p>
                <p>• 140-160°: Moderate curvature</p>
                <p>• &gt;160°: Extended/straight region</p>
            </div>
        </div>

        <!-- Chi Angle Distribution Panel -->
        <div class="panel panel-full">
            <h2> Sidechain Chi Angle Distribution</h2>
            <div class="controls">
                <button class="btn" onclick="downloadChart('chiChart', 'chi_distribution.png')">💾 Save</button>
            </div>
            <div class="chart-container">
                <canvas id="chiChart"></canvas>
            </div>
            <div class="info-box">
                <p><strong>Chi Angles:</strong> Sidechain dihedral angles that define rotamer conformations</p>
                <p>• Preferred values cluster around -180°, -60°, 60°, and 180° (gauche+, gauche-, trans)</p>
            </div>
        </div>
    </div>
    </div> <!-- end geo-pane-charts -->

    <!-- Data Table pane -->
    <div id="geo-pane-table" class="geo-pane">
        <div class="panel panel-full">
            <div class="dt-toolbar">
                <input type="text" id="dt-search" placeholder="🔍 Filter by chain, residue, or SS…"
                       oninput="filterDt()">
                <button class="dt-dl-btn" onclick="downloadDt()">⬇ Download CSV</button>
                <span id="dt-count" style="color:#888; font-size:12px;"></span>
            </div>
            <div class="dt-scroll">
                <table class="dt-table" id="dt-table">
                    <thead><tr>
                        <th onclick="sortDt(0)">Chain <span class="si">⇅</span></th>
                        <th onclick="sortDt(1)">ResSeq <span class="si">⇅</span></th>
                        <th onclick="sortDt(2)">Residue <span class="si">⇅</span></th>
                        <th onclick="sortDt(3)">Phi (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(4)">Psi (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(5)">Omega (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(6)">Structure <span class="si">⇅</span></th>
                        <th onclick="sortDt(7)">Bend (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(8)">χ1 (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(9)">χ2 (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(10)">χ3 (°) <span class="si">⇅</span></th>
                        <th onclick="sortDt(11)">χ4 (°) <span class="si">⇅</span></th>
                    </tr></thead>
                    <tbody id="dt-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const ramaData = {json.dumps(rama_points)};
        const bendData = {json.dumps(bend_by_chain)};
        const chiData  = {json.dumps(chi_data)};
        const geoTableData = {geo_table_json};

        let ramaChart, bendChart, chiChart;
        let showRamaRegions = true;
        let showBendThresholds = true;

        // Chain colors
        const chainColors = {{
            'H': 'rgba(231, 76, 60, 0.8)',
            'L': 'rgba(52, 152, 219, 0.8)',
            'A': 'rgba(46, 204, 113, 0.8)',
            'B': 'rgba(241, 196, 15, 0.8)',
            'C': 'rgba(155, 89, 182, 0.8)',
        }};

        // Initialize Ramachandran Plot
        function createRamachandranPlot() {{
            const ctx = document.getElementById('ramaChart').getContext('2d');

            if (ramaChart) ramaChart.destroy();

            const datasets = [];

            // Add region backgrounds if enabled
            if (showRamaRegions) {{
                // Alpha helix region
                datasets.push({{
                    label: 'α-Helix region',
                    data: [
                        {{x: -90, y: -75}}, {{x: -30, y: -75}},
                        {{x: -30, y: -20}}, {{x: -90, y: -20}},
                        {{x: -90, y: -75}}
                    ],
                    backgroundColor: 'rgba(255, 0, 0, 0.15)',
                    borderColor: 'rgba(255, 0, 0, 0.3)',
                    borderWidth: 1,
                    pointRadius: 0,
                    showLine: true,
                    fill: true
                }});

                // Beta sheet region
                datasets.push({{
                    label: 'β-Sheet region',
                    data: [
                        {{x: -180, y: 90}}, {{x: -90, y: 90}},
                        {{x: -90, y: 180}}, {{x: -180, y: 180}},
                        {{x: -180, y: 90}}
                    ],
                    backgroundColor: 'rgba(0, 0, 255, 0.15)',
                    borderColor: 'rgba(0, 0, 255, 0.3)',
                    borderWidth: 1,
                    pointRadius: 0,
                    showLine: true,
                    fill: true
                }});
            }}

            // Add data points
            datasets.push({{
                label: 'Residues',
                data: ramaData.map(d => ({{x: d.phi, y: d.psi, residue: d.residue}})),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1,
                pointRadius: 4,
                pointHoverRadius: 6
            }});

            ramaChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const point = context.raw;
                                    if (point.residue) {{
                                        return [
                                            point.residue,
                                            `φ: ${{point.x.toFixed(1)}}°`,
                                            `ψ: ${{point.y.toFixed(1)}}°`
                                        ];
                                    }}
                                    return '';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'φ (Phi) [°]', font: {{ size: 14, weight: 'bold' }} }},
                            min: -180,
                            max: 180,
                            ticks: {{ stepSize: 60 }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'ψ (Psi) [°]', font: {{ size: 14, weight: 'bold' }} }},
                            min: -180,
                            max: 180,
                            ticks: {{ stepSize: 60 }}
                        }}
                    }}
                }}
            }});
        }}

        // Initialize Bend Angle Plot
        function createBendPlot() {{
            const ctx = document.getElementById('bendChart').getContext('2d');

            if (bendChart) bendChart.destroy();

            const datasets = [];

            // Add data for each chain
            Object.keys(bendData).forEach((chainId, idx) => {{
                const data = bendData[chainId];
                datasets.push({{
                    label: `Chain ${{chainId}}`,
                    data: data.map(d => ({{x: d.resseq, y: d.angle, residue: d.residue}})),
                    borderColor: chainColors[chainId] || `hsl(${{idx * 60}}, 70%, 50%)`,
                    backgroundColor: chainColors[chainId] || `hsl(${{idx * 60}}, 70%, 50%)`,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    tension: 0.1
                }});
            }});

            // Add threshold lines
            if (showBendThresholds && Object.keys(bendData).length > 0) {{
                const allResseqs = [];
                Object.values(bendData).forEach(chain => {{
                    chain.forEach(d => allResseqs.push(d.resseq));
                }});
                const minRes = Math.min(...allResseqs);
                const maxRes = Math.max(...allResseqs);

                datasets.push({{
                    label: 'Bent (140°)',
                    data: [{{x: minRes, y: 140}}, {{x: maxRes, y: 140}}],
                    borderColor: 'rgba(255, 0, 0, 0.5)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }});

                datasets.push({{
                    label: 'Straight (160°)',
                    data: [{{x: minRes, y: 160}}, {{x: maxRes, y: 160}}],
                    borderColor: 'rgba(0, 255, 0, 0.5)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }});
            }}

            bendChart = new Chart(ctx, {{
                type: 'line',
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
                                    if (point && point.residue) {{
                                        return [
                                            point.residue,
                                            `Angle: ${{point.y.toFixed(1)}}°`
                                        ];
                                    }}
                                    return `${{context.dataset.label}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Residue Number', font: {{ size: 14, weight: 'bold' }} }},
                            type: 'linear'
                        }},
                        y: {{
                            title: {{ display: true, text: 'Bend Angle (°)', font: {{ size: 14, weight: 'bold' }} }},
                            min: 0,
                            max: 180,
                            ticks: {{ stepSize: 30 }}
                        }}
                    }}
                }}
            }});
        }}

        // Initialize Chi Distribution Plot (FIXED VERSION)
        function createChiPlot() {{
            const ctx = document.getElementById('chiChart').getContext('2d');

            if (chiChart) chiChart.destroy();

            // Define bin parameters
            const binSize = 30;
            const binCenters = [];
            for (let i = -180; i < 180; i += binSize) {{
                binCenters.push(i + binSize/2);
            }}

            // Create datasets for each chi angle type
            const datasets = [];
            const colors = [
                'rgba(231, 76, 60, 0.7)',
                'rgba(52, 152, 219, 0.7)',
                'rgba(46, 204, 113, 0.7)',
                'rgba(241, 196, 15, 0.7)',
                'rgba(155, 89, 182, 0.7)'
            ];

            // Process each chi angle type
            let hasData = false;
            Object.keys(chiData).forEach((chiName, idx) => {{
                const values = chiData[chiName];
                if (values.length === 0) return;

                hasData = true;

                // Create histogram bins
                const counts = new Array(binCenters.length).fill(0);
                values.forEach(v => {{
                    const binIndex = Math.floor((v + 180) / binSize);
                    if (binIndex >= 0 && binIndex < counts.length) {{
                        counts[binIndex]++;
                    }}
                }});

                // Format chi name for display (use Greek letter)
                const displayName = chiName.replace('chi', 'χ');

                datasets.push({{
                    label: displayName,
                    data: counts,
                    backgroundColor: colors[idx % colors.length],
                    borderColor: colors[idx % colors.length].replace('0.7', '1'),
                    borderWidth: 1,
                    barPercentage: 0.9,
                    categoryPercentage: 1.0
                }});
            }});

            // Create labels for x-axis (show major tick marks)
            const labels = binCenters.map(c => {{
                if (c === -165 || c === -90 || c === 0 || c === 90 || c === 165) {{
                    return `${{c}}°`;
                }}
                return c;
            }});

            // Create the chart
            if (hasData) {{
                chiChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: labels,
                        datasets: datasets
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false
                        }},
                        plugins: {{
                            legend: {{
                                display: true,
                                position: 'top',
                                labels: {{
                                    usePointStyle: true,
                                    padding: 15,
                                    font: {{
                                        size: 12,
                                        weight: 'bold'
                                    }}
                                }}
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(context) {{
                                        const binCenter = binCenters[context[0].dataIndex];
                                        const binStart = binCenter - binSize/2;
                                        const binEnd = binCenter + binSize/2;
                                        return `Range: ${{binStart.toFixed(0)}}° to ${{binEnd.toFixed(0)}}°`;
                                    }},
                                    label: function(context) {{
                                        return `${{context.dataset.label}}: ${{context.parsed.y}} residues`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Chi Angle (°)',
                                    font: {{ size: 14, weight: 'bold' }}
                                }},
                                ticks: {{
                                    maxRotation: 45,
                                    minRotation: 0,
                                    autoSkip: true,
                                    maxTicksLimit: 12
                                }},
                                grid: {{
                                    display: true,
                                    color: 'rgba(0, 0, 0, 0.05)'
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Number of Residues',
                                    font: {{ size: 14, weight: 'bold' }}
                                }},
                                beginAtZero: true,
                                ticks: {{
                                    stepSize: 1,
                                    precision: 0
                                }},
                                grid: {{
                                    display: true,
                                    color: 'rgba(0, 0, 0, 0.1)'
                                }}
                            }}
                        }}
                    }}
                }});
            }} else {{
                // No chi data available
                ctx.font = '16px Arial';
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                ctx.fillText('No chi angle data available', ctx.canvas.width / 2, ctx.canvas.height / 2);
            }}
        }}

        // Toggle functions
        function toggleRamaRegions() {{
            showRamaRegions = !showRamaRegions;
            createRamachandranPlot();
        }}

        function toggleBendThresholds() {{
            showBendThresholds = !showBendThresholds;
            createBendPlot();
        }}

        // Download function
        function downloadChart(chartId, filename) {{
            const canvas = document.getElementById(chartId);
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        // Initialize all charts
        createRamachandranPlot();
        createBendPlot();
        createChiPlot();

        // ── Tab switching ─────────────────────────────────────────────
        function showGeoTab(tab) {{
            ['charts', 'table'].forEach(t => {{
                document.getElementById('geo-pane-' + t).classList.toggle('active', t === tab);
                document.getElementById('geo-tab-'  + t).classList.toggle('active', t === tab);
            }});
        }}

        // ── Data Table ────────────────────────────────────────────────
        const _ssBadge = {{
            'α-Helix':    '<span class="ss-badge ss-alpha">α-Helix</span>',
            'β-Sheet':    '<span class="ss-badge ss-beta">β-Sheet</span>',
            'Left Helix': '<span class="ss-badge ss-left">Left Helix</span>',
            'Other':      '<span class="ss-badge ss-other">Other</span>',
            '—':          '<span class="ss-badge ss-dash">—</span>',
        }};

        const _dtKeys = ['chain','resseq','resname','phi','psi','omega','ss','bend_deg',
                         'chi1','chi2','chi3','chi4'];

        let _dtRows   = geoTableData.slice();
        let _dtSortCol = 1, _dtSortAsc = true;   // default: sort by resseq

        function _fmt(v) {{ return (v === null || v === undefined) ? '—' : v; }}

        function _renderDt() {{
            const tb = document.getElementById('dt-body');
            if (!_dtRows.length) {{
                tb.innerHTML = '<tr><td colspan="12" class="dt-no-rows">No residues match the filter.</td></tr>';
            }} else {{
                tb.innerHTML = _dtRows.map(r => `<tr>
                    <td>${{r.chain}}</td>
                    <td>${{r.resseq}}</td>
                    <td>${{r.resname}}</td>
                    <td>${{_fmt(r.phi)}}</td>
                    <td>${{_fmt(r.psi)}}</td>
                    <td>${{_fmt(r.omega)}}</td>
                    <td>${{_ssBadge[r.ss] || r.ss}}</td>
                    <td>${{_fmt(r.bend_deg)}}</td>
                    <td>${{_fmt(r.chi1)}}</td>
                    <td>${{_fmt(r.chi2)}}</td>
                    <td>${{_fmt(r.chi3)}}</td>
                    <td>${{_fmt(r.chi4)}}</td>
                </tr>`).join('');
            }}
            const el = document.getElementById('dt-count');
            if (el) el.textContent = _dtRows.length + ' residue' + (_dtRows.length !== 1 ? 's' : '');
        }}

        function filterDt() {{
            const q = (document.getElementById('dt-search').value || '').toLowerCase();
            _dtRows = q
                ? geoTableData.filter(r =>
                    String(r.chain).toLowerCase().includes(q) ||
                    String(r.resseq).includes(q) ||
                    String(r.resname).toLowerCase().includes(q) ||
                    String(r.ss).toLowerCase().includes(q))
                : geoTableData.slice();
            _applySortDt();
            _renderDt();
        }}

        function _applySortDt() {{
            const key = _dtKeys[_dtSortCol];
            _dtRows.sort((a, b) => {{
                const av = a[key], bv = b[key];
                const nullA = av === null || av === undefined;
                const nullB = bv === null || bv === undefined;
                if (nullA && nullB) return 0;
                if (nullA) return 1;
                if (nullB) return -1;
                const n = (typeof av === 'number' && typeof bv === 'number')
                    ? av - bv : String(av).localeCompare(String(bv), undefined, {{numeric: true}});
                return _dtSortAsc ? n : -n;
            }});
        }}

        function sortDt(col) {{
            _dtSortAsc = (_dtSortCol === col) ? !_dtSortAsc : true;
            _dtSortCol = col;
            _applySortDt();
            _renderDt();
        }}

        function _escCsv(v) {{
            const s = (v === null || v === undefined) ? '' : String(v);
            return s.includes(',') || s.includes('"') || s.includes('\\n')
                ? '"' + s.replace(/"/g, '""') + '"' : s;
        }}

        function downloadDt() {{
            const headers = ['chain','resseq','resname','phi_deg','psi_deg','omega_deg',
                             'secondary_structure','bend_deg','chi1_deg','chi2_deg','chi3_deg','chi4_deg'];
            const lines = [headers.join(',')];
            geoTableData.forEach(r => {{
                lines.push([r.chain, r.resseq, r.resname, r.phi, r.psi, r.omega,
                            r.ss, r.bend_deg, r.chi1, r.chi2, r.chi3, r.chi4]
                    .map(_escCsv).join(','));
            }});
            const blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href = url; a.download = 'geometry_per_residue.csv'; a.click();
            URL.revokeObjectURL(url);
        }}

        // Initial table render
        _applySortDt();
        _renderDt();

    </script>
"""

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"])


def visualize_geometry(
    cx: Complex,
    chains: Optional[List[str]] = None,
    viz_type: str = 'all',
    output_file: str = 'geometry_analysis'
) -> Dict[str, str]:
    """
    Generate geometry visualizations for the specified chains.
    
    This is the main function called by app.py for geometry analysis.
    
    Args:
        cx: Complex structure
        chains: List of chain IDs to analyze (e.g., ['H', 'L']). If None, analyze all chains.
        viz_type: Type of visualization to generate:
            - 'all': Generate all visualizations + dashboard
            - 'ramachandran': Only Ramachandran plot
            - 'chi': Only chi angle distribution
            - 'bends': Only bend angle plot
            - 'dashboard': Only dashboard with all plots
        output_file: Base path for output files (without extension)
    
    Returns:
        Dictionary mapping visualization type to file path
    
    Example:
        >>> files = visualize_geometry(cx, chains=['H', 'L'], viz_type='all', 
        ...                            output_file='/tmp/results/geometry')
        >>> print(files)
        {
            'dashboard': '/tmp/results/geometry.html',
            'ramachandran': '/tmp/results/geometry_ramachandran.html',
            'chi': '/tmp/results/geometry_chi.html',
            'bends': '/tmp/results/geometry_bends.html'
        }
    """
    from ..selection import select
    from ..geometry import analyze_geometry
    
    # Build selection string
    if chains is None or len(chains) == 0:
        # Analyze all chains
        selection = "*"
    else:
        # Analyze specified chains
        selection = list(chains)
    
    # Perform comprehensive geometry analysis
    geom = analyze_geometry(
        cx, 
        selection,
        compute_backbone=True,
        compute_sidechains=True,
        compute_bends=True
    )
    
    result_files = {}
    
    # Generate requested visualizations
    if viz_type == 'all' or viz_type == 'dashboard':
        # Generate dashboard with all plots
        dashboard_html = generate_geometry_dashboard(
            geom.backbone,
            geom.sidechains,
            geom.bends,
            title=f"Geometry Analysis - Chains {', '.join(chains) if chains else 'All'}"
        )
        dashboard_path = f"{output_file}.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        result_files['dashboard'] = dashboard_path
    
    if viz_type == 'all' or viz_type == 'ramachandran':
        # Generate Ramachandran plot
        rama_html = generate_ramachandran_plot(
            geom.backbone,
            title=f"Ramachandran Plot - Chains {', '.join(chains) if chains else 'All'}",
            highlight_outliers=True,
            show_regions=True
        )
        rama_path = f"{output_file}_ramachandran.html"
        with open(rama_path, 'w') as f:
            f.write(rama_html)
        result_files['ramachandran'] = rama_path
    
    if viz_type == 'all' or viz_type == 'chi':
        # Generate chi distribution plot
        chi_html = generate_chi_distribution(
            geom.sidechains,
            title=f"Chi Angle Distribution - Chains {', '.join(chains) if chains else 'All'}"
        )
        chi_path = f"{output_file}_chi.html"
        with open(chi_path, 'w') as f:
            f.write(chi_html)
        result_files['chi'] = chi_path
    
    if viz_type == 'all' or viz_type == 'bends':
        # Generate bend angle plot
        bend_html = generate_bend_angle_plot(
            geom.bends,
            title=f"Backbone Bend Angles - Chains {', '.join(chains) if chains else 'All'}"
        )
        bend_path = f"{output_file}_bends.html"
        with open(bend_path, 'w') as f:
            f.write(bend_html)
        result_files['bends'] = bend_path
    
    return result_files





# # Example usage
# def example_usage():
#     """Demonstrate the complete geometry visualization workflow."""
#     from .pdbparser import parse_pdb
    
#     # Load structure
#     cx = parse_pdb("antibody.pdb")
    
#     # Generate all visualizations
#     files = visualize_geometry(
#         cx, 
#         chains=['H', 'L'],
#         viz_type='all',
#         output_file='results/antibody_geometry'
#     )
    
#     print("Generated files:")
#     for viz_type, path in files.items():
#         print(f"  {viz_type}: {path}")
    
#     # Or generate just the dashboard
#     dashboard_files = visualize_geometry(
#         cx,
#         chains=['H', 'L'],
#         viz_type='dashboard',
#         output_file='results/antibody_dashboard'
#     )
    
#     # Or generate just Ramachandran plot
#     rama_files = visualize_geometry(
#         cx,
#         chains=['H'],
#         viz_type='ramachandran',
#         output_file='results/heavy_chain_rama'
#     )

