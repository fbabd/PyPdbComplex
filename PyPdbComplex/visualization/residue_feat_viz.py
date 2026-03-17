"""
Interactive HTML Visualization for Residue Features

This script creates an interactive HTML dashboard to explore residue-level features
extracted from protein complexes. Features include:
- Interactive 3D scatter plots of residue positions colored by properties
- Sortable/filterable data tables
- Distribution plots for various features
- Interface and hotspot highlighting
- Export capabilities

Usage:
    python residue_features_viz.py <features.csv> [--output dashboard.html]
    
Or in code:
    from residue_features_viz import create_interactive_dashboard
    create_interactive_dashboard(features_dict, output_path="dashboard.html")
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import csv


def load_features_from_csv(csv_path: str) -> List[Dict]:
    """Load features from CSV file into list of dictionaries."""
    features = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['resseq', 'num_heavy_atoms', 'num_all_atoms', 
                       'interface_contact_count', 'num_partner_residues',
                       'num_hbonds', 'num_salt_bridges', 'num_hydrophobic',
                       'num_pi_stacking', 'num_disulfides', 'total_interactions']:
                if key in row and row[key]:
                    row[key] = int(row[key])
            
            for key in ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4',
                       'bend_angle', 'unbound_sasa', 'bound_sasa', 'buried_fraction',
                       'min_dist_to_partner', 'ca_dist_to_partner', 
                       'avg_dist_to_neighbors', 'vdw_energy', 'avg_bfactor',
                       'max_bfactor', 'min_bfactor']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            
            # Convert boolean fields
            for key in ['is_interface', 'is_charged', 'is_polar', 'is_hydrophobic',
                       'is_aromatic', 'is_positive', 'is_negative']:
                if key in row:
                    row[key] = row[key].lower() in ['true', '1', 'yes']
            
            features.append(row)
    
    return features


def features_dict_to_list(features: Dict[str, Any]) -> List[Dict]:
    """
    Convert features dictionary to list of dictionaries.
    Handles ResidueFeatures objects by calling their to_dict() method.
    """
    result = []
    for res_id, feat in features.items():
        # Check if it's a ResidueFeatures object with to_dict method
        if hasattr(feat, 'to_dict') and callable(feat.to_dict):
            result.append(feat.to_dict())
        # Otherwise assume it's already a dict
        elif isinstance(feat, dict):
            result.append(feat)
        else:
            raise TypeError(f"Feature must have to_dict() method or be a dict, got {type(feat)}")
    
    return result


def create_interactive_dashboard(
    features: Union[List[Dict], Dict[str, Any], str],
    output_path: str = "residue_features_dashboard.html",
    title: str = "Residue Features Interactive Dashboard",
    include_3d: bool = True,
) -> None:
    """
    Create an interactive HTML dashboard for residue features.
    
    Args:
        features: Can be:
            - Dict[str, ResidueFeatures]: Dictionary mapping residue IDs to ResidueFeatures objects
            - List[Dict]: List of feature dictionaries
            - str: Path to CSV file containing features
        output_path: Path for output HTML file
        title: Dashboard title
        include_3d: Whether to include 3D visualization (requires coordinates)
    
    Example:
        # From extract_residue_features() output
        features = extract_residue_features(cx, "chain A", "chain B")
        create_interactive_dashboard(features, "my_analysis.html")
    """
    # Load/convert features to list of dicts
    if isinstance(features, str):
        features_list = load_features_from_csv(features)
    elif isinstance(features, dict):
        features_list = features_dict_to_list(features)
    else:
        features_list = features
    
    if not features_list:
        raise ValueError("No features provided")
    
    # Convert to JSON for embedding
    features_json = json.dumps(features_list, indent=2)
    
    # Determine available features
    sample = features_list[0]
    has_sasa = 'bound_sasa' in sample and sample['bound_sasa'] is not None
    has_geometry = 'phi' in sample and sample['phi'] is not None
    has_vdw = 'vdw_energy' in sample and sample['vdw_energy'] is not None
    has_interface = any(f.get('is_interface', False) for f in features_list)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            padding: 0 40px;
            overflow-x: auto;
        }}
        
        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #6c757d;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }}
        
        .tab:hover {{
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: 600;
        }}
        
        .tab-content {{
            display: none;
            padding: 40px;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .plot-container {{
            margin-bottom: 40px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .plot-title {{
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2c3e50;
        }}
        
        .controls {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #495057;
        }}
        
        .control-group select, .control-group input {{
            width: 100%;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }}
        
        .control-group select:focus, .control-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }}
        
        table.dataTable {{
            width: 100% !important;
            margin-top: 20px !important;
        }}
        
        .dataTables_wrapper {{
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        
        .button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s;
            margin: 5px;
        }}
        
        .button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .filter-section {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .filter-section h3 {{
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        
        .checkbox-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .checkbox-group label {{
            display: flex;
            align-items: center;
            cursor: pointer;
        }}
        
        .checkbox-group input {{
            margin-right: 5px;
            width: auto;
        }}
        
        @media (max-width: 768px) {{
            .grid-2, .grid-3 {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 {title}</h1>
            <p>Interactive analysis of protein residue features</p>
        </div>
        
        <div class="stats-grid" id="statsGrid"></div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab" onclick="showTab('interface')">🔗 Interface</button>
            <button class="tab" onclick="showTab('interactions')">⚡ Interactions</button>
            <button class="tab" onclick="showTab('properties')">🧪 Properties</button>
            <button class="tab" onclick="showTab('geometry')">📐 Geometry</button>
            <button class="tab" onclick="showTab('table')">📋 Data Table</button>
        </div>
        
        <div id="overview" class="tab-content active">
            <div class="plot-container">
                <div class="plot-title">Residue Distribution by Chain</div>
                <div id="chainDistribution"></div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="plot-title">Chemical Property Distribution</div>
                    <div id="chemicalPie"></div>
                </div>
                
                <div class="plot-container">
                    <div class="plot-title">Interaction Count Distribution</div>
                    <div id="interactionHist"></div>
                </div>
            </div>
        </div>
        
        <div id="interface" class="tab-content">
            <div class="filter-section">
                <h3>Filter Options</h3>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="showInterfaceOnly" onchange="updateInterfacePlots()"> Interface Only</label>
                    <label><input type="checkbox" id="showHotspotsOnly" onchange="updateInterfacePlots()"> Hotspots Only</label>
                </div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Interface Residues by Position</div>
                <div id="interfaceScatter"></div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="plot-title">Contact Count Distribution</div>
                    <div id="contactHist"></div>
                </div>
                
                <div class="plot-container">
                    <div class="plot-title">Distance to Partner</div>
                    <div id="distanceHist"></div>
                </div>
            </div>
        </div>
        
        <div id="interactions" class="tab-content">
            <div class="plot-container">
                <div class="plot-title">Interaction Types Distribution</div>
                <div id="interactionTypes"></div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="plot-title">H-bonds vs Salt Bridges</div>
                    <div id="hbondSaltScatter"></div>
                </div>
                
                <div class="plot-container">
                    <div class="plot-title">Total Interactions by Residue Type</div>
                    <div id="interactionsByType"></div>
                </div>
            </div>
            
            {"<div class='plot-container'><div class='plot-title'>Van der Waals Energy Distribution</div><div id='vdwHist'></div></div>" if has_vdw else ""}
        </div>
        
        <div id="properties" class="tab-content">
            {"<div class='plot-container'><div class='plot-title'>SASA: Bound vs Unbound</div><div id='sasaScatter'></div></div>" if has_sasa else ""}
            
            {"<div class='plot-container'><div class='plot-title'>Buried Fraction Distribution</div><div id='buriedHist'></div></div>" if has_sasa else ""}
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="plot-title">B-factor Distribution</div>
                    <div id="bfactorBox"></div>
                </div>
                
                <div class="plot-container">
                    <div class="plot-title">Atom Count Distribution</div>
                    <div id="atomHist"></div>
                </div>
            </div>
        </div>
        
        <div id="geometry" class="tab-content">
            {"<div class='plot-container'><div class='plot-title'>Ramachandran Plot (Phi vs Psi)</div><div id='ramachandran'></div></div>" if has_geometry else "<p>No geometry data available</p>"}
            
            {"<div class='grid-2'><div class='plot-container'><div class='plot-title'>Phi Angle Distribution</div><div id='phiHist'></div></div><div class='plot-container'><div class='plot-title'>Psi Angle Distribution</div><div id='psiHist'></div></div></div>" if has_geometry else ""}
            
            {"<div class='plot-container'><div class='plot-title'>Chi1 Angle Distribution</div><div id='chi1Hist'></div></div>" if has_geometry else ""}
        </div>
        
        <div id="table" class="tab-content">
            <div style="margin-bottom: 20px;">
                <button class="button" onclick="exportFilteredCSV()">📥 Export Filtered Data</button>
                <button class="button" onclick="exportFullCSV()">📥 Export All Data</button>
            </div>
            
            <table id="featuresTable" class="display" style="width:100%">
                <thead></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script>
        // Feature data
        const features = {features_json};
        
        // Calculate statistics
        const stats = {{
            total: features.length,
            interface: features.filter(f => f.is_interface).length,
            charged: features.filter(f => f.is_charged).length,
            hydrophobic: features.filter(f => f.is_hydrophobic).length,
            aromatic: features.filter(f => f.is_aromatic).length,
            hotspots: features.filter(f => f.is_interface && f.total_interactions >= 3).length,
            avgInteractions: (features.reduce((sum, f) => sum + (f.total_interactions || 0), 0) / features.length).toFixed(2),
            totalHbonds: features.reduce((sum, f) => sum + (f.num_hbonds || 0), 0),
            totalSaltBridges: features.reduce((sum, f) => sum + (f.num_salt_bridges || 0), 0)
        }};
        
        // Display statistics
        document.getElementById('statsGrid').innerHTML = `
            <div class="stat-card"><div class="stat-value">${{stats.total}}</div><div class="stat-label">Total Residues</div></div>
            <div class="stat-card"><div class="stat-value">${{stats.interface}}</div><div class="stat-label">Interface</div></div>
            <div class="stat-card"><div class="stat-value">${{stats.hotspots}}</div><div class="stat-label">Hotspots</div></div>
            <div class="stat-card"><div class="stat-value">${{stats.totalHbonds}}</div><div class="stat-label">H-bonds</div></div>
            <div class="stat-card"><div class="stat-value">${{stats.totalSaltBridges}}</div><div class="stat-label">Salt Bridges</div></div>
            <div class="stat-card"><div class="stat-value">${{stats.avgInteractions}}</div><div class="stat-label">Avg Interactions</div></div>
        `;
        
        // Tab switching
        function showTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Trigger plot updates when switching tabs
            if (tabName === 'interface') updateInterfacePlots();
        }}
        
        // ==================== OVERVIEW TAB ====================
        
        // Chain distribution
        const chainCounts = {{}};
        features.forEach(f => {{
            chainCounts[f.chain_id] = (chainCounts[f.chain_id] || 0) + 1;
        }});
        
        Plotly.newPlot('chainDistribution', [{{
            x: Object.keys(chainCounts),
            y: Object.values(chainCounts),
            type: 'bar',
            marker: {{color: '#667eea'}}
        }}], {{
            xaxis: {{title: 'Chain ID'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        
        // Chemical properties pie chart
        Plotly.newPlot('chemicalPie', [{{
            labels: ['Charged', 'Polar', 'Hydrophobic', 'Aromatic', 'Other'],
            values: [
                stats.charged,
                features.filter(f => f.is_polar).length,
                stats.hydrophobic,
                stats.aromatic,
                features.filter(f => !f.is_charged && !f.is_polar && !f.is_hydrophobic && !f.is_aromatic).length
            ],
            type: 'pie',
            marker: {{colors: ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#95a5a6']}}
        }}], {{
            margin: {{t: 20}}
        }});
        
        // Interaction histogram
        const interactions = features.map(f => f.total_interactions || 0);
        Plotly.newPlot('interactionHist', [{{
            x: interactions,
            type: 'histogram',
            marker: {{color: '#764ba2'}},
            nbinsx: 20
        }}], {{
            xaxis: {{title: 'Number of Interactions'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        
        // ==================== INTERFACE TAB ====================
        
        function updateInterfacePlots() {{
            let data = features;
            
            if (document.getElementById('showInterfaceOnly')?.checked) {{
                data = data.filter(f => f.is_interface);
            }}
            
            if (document.getElementById('showHotspotsOnly')?.checked) {{
                data = data.filter(f => f.is_interface && f.total_interactions >= 3);
            }}
            
            // Interface scatter by position
            const chains = [...new Set(data.map(f => f.chain_id))];
            const traces = chains.map(chain => {{
                const chainData = data.filter(f => f.chain_id === chain);
                return {{
                    x: chainData.map(f => f.resseq),
                    y: chainData.map(f => f.total_interactions || 0),
                    text: chainData.map(f => `${{f.resname}} ${{f.chain_id}}:${{f.resseq}}<br>Interactions: ${{f.total_interactions}}`),
                    name: `Chain ${{chain}}`,
                    mode: 'markers',
                    marker: {{
                        size: chainData.map(f => f.is_interface ? 10 : 5),
                        opacity: 0.7
                    }}
                }};
            }});
            
            Plotly.newPlot('interfaceScatter', traces, {{
                xaxis: {{title: 'Residue Number'}},
                yaxis: {{title: 'Total Interactions'}},
                hovermode: 'closest',
                margin: {{t: 20}}
            }});
            
            // Contact count histogram
            const contacts = data.map(f => f.interface_contact_count || 0);
            Plotly.newPlot('contactHist', [{{
                x: contacts,
                type: 'histogram',
                marker: {{color: '#667eea'}},
                nbinsx: 20
            }}], {{
                xaxis: {{title: 'Contact Count'}},
                yaxis: {{title: 'Number of Residues'}},
                margin: {{t: 20}}
            }});
            
            // Distance histogram
            const distances = data.filter(f => f.min_dist_to_partner != null).map(f => f.min_dist_to_partner);
            Plotly.newPlot('distanceHist', [{{
                x: distances,
                type: 'histogram',
                marker: {{color: '#764ba2'}},
                nbinsx: 30
            }}], {{
                xaxis: {{title: 'Distance to Partner (Å)'}},
                yaxis: {{title: 'Count'}},
                margin: {{t: 20}}
            }});
        }}
        
        updateInterfacePlots();
        
        // ==================== INTERACTIONS TAB ====================
        
        // Interaction types bar chart
        Plotly.newPlot('interactionTypes', [{{
            x: ['H-bonds', 'Salt Bridges', 'Hydrophobic', 'Pi-Stacking', 'Disulfides'],
            y: [
                stats.totalHbonds,
                stats.totalSaltBridges,
                features.reduce((sum, f) => sum + (f.num_hydrophobic || 0), 0),
                features.reduce((sum, f) => sum + (f.num_pi_stacking || 0), 0),
                features.reduce((sum, f) => sum + (f.num_disulfides || 0), 0)
            ],
            type: 'bar',
            marker: {{color: ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']}}
        }}], {{
            xaxis: {{title: 'Interaction Type'}},
            yaxis: {{title: 'Total Count'}},
            margin: {{t: 20}}
        }});
        
        // H-bonds vs salt bridges scatter
        const interfaceRes = features.filter(f => f.is_interface);
        Plotly.newPlot('hbondSaltScatter', [{{
            x: interfaceRes.map(f => f.num_hbonds || 0),
            y: interfaceRes.map(f => f.num_salt_bridges || 0),
            text: interfaceRes.map(f => `${{f.resname}} ${{f.chain_id}}:${{f.resseq}}`),
            mode: 'markers',
            marker: {{
                size: interfaceRes.map(f => (f.total_interactions || 0) * 3),
                color: interfaceRes.map(f => f.total_interactions || 0),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{title: 'Total<br>Interactions'}}
            }}
        }}], {{
            xaxis: {{title: 'Number of H-bonds'}},
            yaxis: {{title: 'Number of Salt Bridges'}},
            hovermode: 'closest',
            margin: {{t: 20}}
        }});
        
        // Interactions by residue type
        const resTypes = {{}};
        features.forEach(f => {{
            if (!resTypes[f.resname]) resTypes[f.resname] = [];
            resTypes[f.resname].push(f.total_interactions || 0);
        }});
        
        const boxData = Object.entries(resTypes).map(([name, values]) => ({{
            y: values,
            name: name,
            type: 'box',
            boxmean: true
        }}));
        
        Plotly.newPlot('interactionsByType', boxData, {{
            yaxis: {{title: 'Total Interactions'}},
            xaxis: {{title: 'Residue Type'}},
            margin: {{t: 20}},
            showlegend: false
        }});
        
        {f'''
        // VdW energy histogram
        const vdwEnergies = features.filter(f => f.vdw_energy != null).map(f => f.vdw_energy);
        Plotly.newPlot('vdwHist', [{{
            x: vdwEnergies,
            type: 'histogram',
            marker: {{color: '#e74c3c'}},
            nbinsx: 30
        }}], {{
            xaxis: {{title: 'VdW Energy (kcal/mol)'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        ''' if has_vdw else ''}
        
        // ==================== PROPERTIES TAB ====================
        
        {f'''
        // SASA scatter
        const sasaData = features.filter(f => f.bound_sasa != null && f.unbound_sasa != null);
        Plotly.newPlot('sasaScatter', [{{
            x: sasaData.map(f => f.unbound_sasa),
            y: sasaData.map(f => f.bound_sasa),
            text: sasaData.map(f => `${{f.resname}} ${{f.chain_id}}:${{f.resseq}}`),
            mode: 'markers',
            marker: {{
                size: 8,
                color: sasaData.map(f => f.buried_fraction || 0),
                colorscale: 'RdYlBu',
                showscale: true,
                colorbar: {{title: 'Buried<br>Fraction'}}
            }}
        }}], {{
            xaxis: {{title: 'Unbound SASA (Ų)'}},
            yaxis: {{title: 'Bound SASA (Ų)'}},
            hovermode: 'closest',
            margin: {{t: 20}}
        }});
        
        // Buried fraction histogram
        const buriedFractions = features.filter(f => f.buried_fraction != null).map(f => f.buried_fraction);
        Plotly.newPlot('buriedHist', [{{
            x: buriedFractions,
            type: 'histogram',
            marker: {{color: '#3498db'}},
            nbinsx: 20
        }}], {{
            xaxis: {{title: 'Buried Fraction'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        ''' if has_sasa else ''}
        
        // B-factor box plots by chain
        const bfactorByChain = {{}};
        features.forEach(f => {{
            if (f.avg_bfactor != null) {{
                if (!bfactorByChain[f.chain_id]) bfactorByChain[f.chain_id] = [];
                bfactorByChain[f.chain_id].push(f.avg_bfactor);
            }}
        }});
        
        const bfactorTraces = Object.entries(bfactorByChain).map(([chain, values]) => ({{
            y: values,
            name: `Chain ${{chain}}`,
            type: 'box',
            boxmean: true
        }}));
        
        Plotly.newPlot('bfactorBox', bfactorTraces, {{
            yaxis: {{title: 'Average B-factor'}},
            margin: {{t: 20}}
        }});
        
        // Atom count histogram
        const atomCounts = features.map(f => f.num_heavy_atoms || 0);
        Plotly.newPlot('atomHist', [{{
            x: atomCounts,
            type: 'histogram',
            marker: {{color: '#9b59b6'}},
            nbinsx: 20
        }}], {{
            xaxis: {{title: 'Number of Heavy Atoms'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        
        // ==================== GEOMETRY TAB ====================
        
        {f'''
        // Ramachandran plot
        const ramaData = features.filter(f => f.phi != null && f.psi != null);
        Plotly.newPlot('ramachandran', [{{
            x: ramaData.map(f => f.phi),
            y: ramaData.map(f => f.psi),
            text: ramaData.map(f => `${{f.resname}} ${{f.chain_id}}:${{f.resseq}}`),
            mode: 'markers',
            marker: {{
                size: 5,
                color: ramaData.map(f => f.chain_id),
                opacity: 0.6
            }}
        }}], {{
            xaxis: {{title: 'Phi (°)', range: [-180, 180]}},
            yaxis: {{title: 'Psi (°)', range: [-180, 180]}},
            hovermode: 'closest',
            margin: {{t: 20}}
        }});
        
        // Phi histogram
        const phiAngles = features.filter(f => f.phi != null).map(f => f.phi);
        Plotly.newPlot('phiHist', [{{
            x: phiAngles,
            type: 'histogram',
            marker: {{color: '#e74c3c'}},
            nbinsx: 36
        }}], {{
            xaxis: {{title: 'Phi Angle (°)'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        
        // Psi histogram
        const psiAngles = features.filter(f => f.psi != null).map(f => f.psi);
        Plotly.newPlot('psiHist', [{{
            x: psiAngles,
            type: 'histogram',
            marker: {{color: '#3498db'}},
            nbinsx: 36
        }}], {{
            xaxis: {{title: 'Psi Angle (°)'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        
        // Chi1 histogram
        const chi1Angles = features.filter(f => f.chi1 != null).map(f => f.chi1);
        Plotly.newPlot('chi1Hist', [{{
            x: chi1Angles,
            type: 'histogram',
            marker: {{color: '#f39c12'}},
            nbinsx: 36
        }}], {{
            xaxis: {{title: 'Chi1 Angle (°)'}},
            yaxis: {{title: 'Count'}},
            margin: {{t: 20}}
        }});
        ''' if has_geometry else ''}
        
        // ==================== DATA TABLE ====================
        
        // Prepare table columns
        const columns = Object.keys(features[0]).map(key => ({{
            title: key,
            data: key
        }}));
        
        // Initialize DataTable
        const table = $('#featuresTable').DataTable({{
            data: features,
            columns: columns,
            pageLength: 25,
            scrollX: true,
            order: [[0, 'asc']],
            dom: 'Bfrtip',
            initComplete: function() {{
                // Add column filters
                this.api().columns().every(function() {{
                    const column = this;
                    const header = $(column.header());
                    const select = $('<select><option value=""></option></select>')
                        .appendTo(header)
                        .on('change', function() {{
                            const val = $.fn.dataTable.util.escapeRegex($(this).val());
                            column.search(val ? '^' + val + '$' : '', true, false).draw();
                        }});
                    
                    column.data().unique().sort().each(function(d) {{
                        if (d != null && d !== '') {{
                            select.append('<option value="' + d + '">' + d + '</option>');
                        }}
                    }});
                }});
            }}
        }});
        
        // Export functions
        function exportFilteredCSV() {{
            const filtered = table.rows({{search: 'applied'}}).data().toArray();
            downloadCSV(filtered, 'filtered_features.csv');
        }}
        
        function exportFullCSV() {{
            downloadCSV(features, 'all_features.csv');
        }}
        
        function downloadCSV(data, filename) {{
            const headers = Object.keys(data[0]);
            const csv = [
                headers.join(','),
                ...data.map(row => headers.map(h => {{
                    const val = row[h];
                    return typeof val === 'string' && val.includes(',') ? `"${{val}}"` : val;
                }}).join(','))
            ].join('\\n');
            
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Interactive dashboard created: {output_path}")
    print(f"📊 Total residues visualized: {len(features_list)}")
    print(f"🔗 Interface residues: {sum(1 for f in features_list if f.get('is_interface', False))}")
    print(f"\n🌐 Open in browser: file://{Path(output_path).absolute()}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create interactive HTML visualization of residue features"
    )
    parser.add_argument(
        'input',
        help='Path to CSV file containing residue features'
    )
    parser.add_argument(
        '-o', '--output',
        default='residue_features_dashboard.html',
        help='Output HTML file path (default: residue_features_dashboard.html)'
    )
    parser.add_argument(
        '-t', '--title',
        default='Residue Features Interactive Dashboard',
        help='Dashboard title'
    )
    
    args = parser.parse_args()
    
    create_interactive_dashboard(
        args.input,
        output_path=args.output,
        title=args.title
    )


if __name__ == '__main__':
    main()