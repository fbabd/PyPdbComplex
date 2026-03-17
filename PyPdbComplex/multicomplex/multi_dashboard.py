from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
import base64
from pathlib import Path
import io



def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for embedding in HTML"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')


def df_to_html_table(df: pd.DataFrame, table_id: str = "data-table", 
                     max_rows: int = 1000) -> str:
    """
    Convert DataFrame to interactive HTML table with search and sort.
    
    Args:
        df: pandas DataFrame
        table_id: HTML element ID for the table
        max_rows: Maximum rows to display (for performance)
    
    Returns:
        HTML string with DataTables.js interactive table
    """
    
    # Limit rows for performance
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        truncated_msg = f"<p><em>Showing first {max_rows} of {len(df)} rows</em></p>"
    else:
        df_display = df
        truncated_msg = ""
    
    # Convert to HTML with pandas
    table_html = df_display.to_html(
        index=False,
        classes='display compact',
        table_id=table_id,
        border=0
    )
    
    # Add DataTables & FixedColumns
    html = truncated_msg + table_html 
    return html 


def create_interactive_dashboard(
    results: Dict[str, Dict],
    pdb_files: List[Tuple[str, str]],
    output_file: Optional[str] = None,  # Changed default to None
    title: str = 'Multiple Complex Analysis',
    wt_name: str = 'WT',
    chains: Optional[List[str]] = None,
) -> str:  # Returns HTML string directly
    """
    Create a comprehensive interactive HTML dashboard with all analysis results.
    No file system operations - all data embedded in HTML.

    Args:
        results: Analysis results dictionary from MultiComplexComparison
        pdb_files: List of (variant_name, pdb_path) tuples
        output_file: Optional path to save HTML (if None, returns HTML string only)
        title: Dashboard title
        wt_name: Wild-type variant name
        chains: List of chains to analyze (None = all chains)
    
    Returns:
        HTML string of the complete dashboard
    """
    
    # Import required modules
    from .multi_complex_tabular_analysis import (
        extract_residue_features,
        generate_publication_figures_for_web,  # Use web version only
        calculate_interface_statistics,
        compare_interface_residues,
    )
    from .multi_analysis_viz import (
        generate_3d_structure_comparison,
        generate_3d_sidebyside_comparison
    )
    
    # =========================================================================
    # 1. Extract and prepare data
    # =========================================================================
    print("  - Extracting residue features...")
    df_features = extract_residue_features(results)
    
    print("  - Calculating interface statistics...")
    df_stats = calculate_interface_statistics(df_features)
    
    print("  - Comparing interface residues...")
    df_interface_comparison = compare_interface_residues(
        df_features, 
        wt_name=wt_name,
        output_file=None
    )
    
    # =========================================================================
    # 2. Generate publication figures as base64
    # =========================================================================
    print("  - Generating publication figures...")
    figures_web = generate_publication_figures_for_web(
        results,
        wt_name=wt_name,
        chains=chains
    )
    
    # =========================================================================
    # 3. Generate 3D visualizations
    # =========================================================================
    print("  - Generating 3D structure comparison...")
    html_3d_single = generate_3d_structure_comparison(
        results,
        pdb_files=pdb_files,
        output_file=None   
    )
    
    print("  - Generating side-by-side 3D comparison...")
    html_3d_sidebyside = generate_3d_sidebyside_comparison(
        results,
        pdb_files=pdb_files,
        output_file=None   
    )
    
    # =========================================================================
    # 4. Build the HTML dashboard
    # =========================================================================
    print("  - Building HTML dashboard...")
    
    # Build complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- CSS Libraries -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .dashboard-header {{
            background: linear-gradient(183deg, #b6cbdb 0%, #e3f2fd 100%);  
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .dashboard-header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 600;
            color: black;
        }}
        
        .dashboard-header p {{
            font-size: 16px;
            opacity: 0.95;
        }}
        
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
            flex-wrap: wrap;
        }}
        
        .tab-button {{
            padding: 16px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            white-space: nowrap;
        }}
        
        .tab-button:hover {{
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }}
        
        .tab-button.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}
        
        .tab-content {{
            display: none;
            padding: 40px;
            animation: fadeIn 0.3s;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section-subtitle {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin: 30px 0 15px 0;
        }}
        
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 16px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        
        .info-box p {{
            margin: 5px 0;
            color: #1565c0;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        table.dataTable {{
            width: 100% !important;
            border-collapse: collapse;
        }}
        
        table.dataTable thead th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        table.dataTable tbody td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        table.dataTable tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .figure-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .figure-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .figure-title {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin: 15px 0 10px 0;
        }}
        
        .figure-caption {{
            font-size: 14px;
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
        }}
        
        .grid-3col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }}
        
        .grid-1col {{
            margin: 20px 0;
        }}
        
        .download-btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            margin: 10px 5px;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }}
        
        .download-btn:hover {{
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0;
        }}
        
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .iframe-wrapper {{
            width: 100%;
            margin: 20px 0;
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #999;
        }}
        
        .loading i {{
            font-size: 48px;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="dashboard-header">
            <h1> Multi-Complex Analysis </h1>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-button active" onclick="openTab(event, 'overview')">
                <i class="fas fa-home"></i> Overview
            </button>
            <button class="tab-button" onclick="openTab(event, 'data-tables')">
                <i class="fas fa-table"></i> Data Tables
            </button>
            <button class="tab-button" onclick="openTab(event, 'statistics')">
                <i class="fas fa-chart-bar"></i> Statistics
            </button>
            <button class="tab-button" onclick="openTab(event, 'figures')">
                <i class="fas fa-chart-line"></i> Figures
            </button>
            <button class="tab-button" onclick="openTab(event, '3d-single')">
                <i class="fas fa-cube"></i> 3D Viewer
            </button>
            <button class="tab-button" onclick="openTab(event, '3d-sidebyside')">
                <i class="fas fa-cubes"></i> 3D Side-by-Side
            </button>
        </div>
        
        <!-- Tab Contents -->
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <h2 class="section-title">Analysis Overview</h2>
            
            <div class="info-box">
                <p><strong>Analysis Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Variants Analyzed:</strong> {', '.join(results.keys())}</p>
                <p><strong>Wild-type Reference:</strong> {wt_name}</p>
               
            </div>
            
            
            <h3 class="section-subtitle">Dashboard Contents</h3>
            <p>This interactive dashboard provides comprehensive analysis results organized in tabs:</p>
            <ul style="margin: 15px 0 15px 30px; line-height: 2;">
                <li><strong>Data Tables:</strong> Interactive, searchable tables of all residue features and comparisons</li>
                <li><strong>Statistics:</strong> Summary statistics and interface residue comparisons</li>
                <li><strong>Publication Figures:</strong> High-quality plots for manuscript inclusion</li>
                <li><strong>3D Viewer:</strong> Interactive molecular visualization with property coloring</li>
                <li><strong>3D Side-by-Side:</strong> Compare multiple variants simultaneously</li>
            </ul>
            
            <h3 class="section-subtitle">Quick Downloads</h3>
            <div style="margin: 20px 0;">
                <button onclick="downloadCSV('features')" class="download-btn">
                    <i class="fas fa-download"></i> Download Features CSV
                </button>
                <button onclick="downloadCSV('statistics')" class="download-btn">
                    <i class="fas fa-download"></i> Download Statistics CSV
                </button>
                <button onclick="downloadCSV('interface')" class="download-btn">
                    <i class="fas fa-download"></i> Download Interface Comparison CSV
                </button>
            </div>
        </div>
        
        <!-- Data Tables Tab -->
        <div id="data-tables" class="tab-content">
            <h2 class="section-title">Data Tables</h2>
            
            <h3 class="section-subtitle">Residue Features</h3>
            <p>Complete per-residue features across all variants. Use the search box to filter by residue, chain, or variant.</p>
            <div class="table-container">
                {df_to_html_table(df_features, 'features-table', max_rows=2000)}
            </div>
        </div>
        
        <!-- Statistics Tab -->
        <div id="statistics" class="tab-content">
            <h2 class="section-title">Statistical Analysis</h2>
            
            <h3 class="section-subtitle">Interface Statistics by Variant</h3>
            <p>Summary statistics for interface residues in each variant.</p>
            <div class="table-container">
                {df_to_html_table(df_stats, 'stats-table')}
            </div>
            
            <h3 class="section-subtitle">Interface Residue Comparison</h3>
            <p>Detailed comparison of interface residues across variants.</p>
            <div class="table-container">
                {df_to_html_table(df_interface_comparison, 'interface-table', max_rows=1000)}
            </div>
        </div>
        
        <!-- Publication Figures Tab -->
        <div id="figures" class="tab-content">
            <h2 class="section-title">Publication-Quality Figures</h2>
            <p>All figures are high-resolution and can be right-clicked to save.</p>
"""
    
    # Add all publication figures using base64 data
    figure_info = {
        'interface_burial': {
            'title': 'Figure 2: Interface Residue Burial Comparison',
            'caption': 'Comparison of SASA burial fraction for top 20 interface residues across variants.'
        },
        'vdw_heatmap': {
            'title': 'Figure 3: Van der Waals Energy Heatmap',
            'caption': 'Heatmap showing VdW energies for top 10 interface residues. Blue = favorable, Red = unfavorable.'
        },
        'contact_distribution': {
            'title': 'Figure 4: Interaction Type Distribution',
            'caption': 'Stacked bar chart showing distribution of interaction types across variants.'
        },
        'mutation_impact': {
            'title': 'Figure 5: Mutation Impact Analysis',
            'caption': 'Scatter plots showing ΔVdW energy vs ΔSASA burial for each variant compared to wild-type.'
        },
    }
    
    # Handle property_profiles_combined (single combined image)
    if 'property_profiles_combined' in figures_web and figures_web['property_profiles_combined']:
        html += """
            <h3 class="section-subtitle">Figure 1: Residue Property Profiles (All Chains)</h3>
            <p class="figure-caption">Combined property profiles showing burial, energy, and contacts across all chains.</p>
            <div class="figure-container">
                <img src="{}" alt="Property Profiles - All Chains Combined" />
            </div>
    """.format(figures_web['property_profiles_combined'])
        # Handle regular figures (single image each)
        for fig_name, info in figure_info.items():
            if fig_name in figures_web and figures_web[fig_name]:
                html += f"""
                <h3 class="section-subtitle">{info['title']}</h3>
                <p class="figure-caption">{info['caption']}</p>
                <div class="figure-container">
                    <img src="{figures_web[fig_name]}" alt="{info['title']}" />
                </div>
    """
        
    
    
    html += """
        </div>
        
        <!-- 3D Single Viewer Tab -->
        <div id="3d-single" class="tab-content">
            <h2 class="section-title">Interactive 3D Structure Viewer</h2>
            <p>Explore protein structures with interactive property-based coloring. Click residues for details.</p>
            <div class="iframe-wrapper">
"""
    
    # Embed the 3D single viewer
    if html_3d_single:
        html += f"""
                <iframe srcdoc="{_escape_html(html_3d_single)}" allow="downloads" style="width:100%;min-height:550px;border:1px solid #dee2e6;border-radius:8px;display:block;"></iframe>
"""
    else:
        html += """
                <div class="loading">
                    <i class="fas fa-spinner"></i>
                    <p>3D Viewer loading...</p>
                </div>
"""

    html += """
            </div>
        </div>

        <!-- 3D Side-by-Side Tab -->
        <div id="3d-sidebyside" class="tab-content">
            <h2 class="section-title">Side-by-Side 3D Comparison</h2>
            <p>Compare multiple variants side-by-side with synchronized controls and property coloring.</p>
            <div class="iframe-wrapper">
"""

    if html_3d_sidebyside:
        html += f"""
                <iframe srcdoc="{_escape_html(html_3d_sidebyside)}" allow="downloads" style="width:100%;min-height:550px;border:1px solid #dee2e6;border-radius:8px;display:block;"></iframe>
"""
    else:
        html += """
                <div class="loading">
                    <i class="fas fa-spinner"></i>
                    <p>3D Viewer loading...</p>
                </div>
"""

    html += f"""
            </div>
        </div>
        
    </div>
    
    <!-- JavaScript Libraries -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    
    <script>
        // Tab switching
        function openTab(evt, tabName) {{
            var i, tabcontent, tabbuttons;
            
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("active");
            }}
            
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {{
                tabbuttons[i].classList.remove("active");
            }}
            
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
        
        // Initialize DataTables
        $(document).ready(function() {{
            $('#features-table').DataTable({{
                pageLength: 25,
                order: [[0, 'asc']],
                scrollX: true
            }});
            
            $('#stats-table').DataTable({{
                pageLength: 10,
                scrollX: true
            }});
            
            $('#interface-table').DataTable({{
                pageLength: 25,
                scrollX: true
            }});
        }});
        
        // Data for downloads
        const csvData = {{
            features: {df_features.to_json(orient='records')},
            statistics: {df_stats.to_json(orient='records')},
            interface: {df_interface_comparison.to_json(orient='records')}
        }};
        
        // Download CSV function
        function downloadCSV(dataType) {{
            const items = csvData[dataType];  // already a JS array, no JSON.parse needed
            if (!items || items.length === 0) return;

            const headers = Object.keys(items[0]);
            let csv = headers.join(',') + '\\n';

            items.forEach(item => {{
                const values = headers.map(header => {{
                    const value = item[header];
                    if (value === null || value === undefined) return '';
                    const strValue = String(value);
                    if (strValue.includes(',') || strValue.includes('"') || strValue.includes('\\n')) {{
                        return '"' + strValue.replace(/"/g, '""') + '"';
                    }}
                    return strValue;
                }});
                csv += values.join(',') + '\\n';
            }});

            const filename = dataType + '_data.csv';

            // Safari does not honour the download attribute on blob URLs —
            // use a data: URI instead, which works across all browsers.
            const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
            if (isSafari) {{
                window.open('data:text/csv;charset=utf-8,' + encodeURIComponent(csv));
            }} else {{
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }}
        }}
    </script>
</body>
</html>
"""
    
    # Optionally save to file if path provided
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"\n✓ Interactive dashboard saved to: {output_file}")
        except Exception as e:
            print(f"\nWarning: Could not save to file: {e}")
            print("  Returning HTML string instead")
    
    print(f"✓ Dashboard generated successfully")
    
    return html


def _escape_html(html_content: str) -> str:
    """Escape HTML for embedding in iframe srcdoc attribute"""
    return (html_content
            .replace('&', '&amp;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


