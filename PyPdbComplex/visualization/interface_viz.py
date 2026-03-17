from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import json

# Try relative imports first (when used as part of abag2 package)
# Fall back to absolute imports (when used standalone)
try:
    from ..models import Complex
    from ..interface import InterfaceAnalysis, ResidueInterfaceData, compute_interface
except ImportError:
    # Standalone usage - expect types to be passed in
    # Type hints will be strings to avoid import errors
    Complex = 'Complex'
    InterfaceAnalysis = 'InterfaceAnalysis'
    ResidueInterfaceData = 'ResidueInterfaceData'
    compute_interface = None

# ContactAnalysis is optional — used for typed contact coloring in 3D view
try:
    from ..contacts import ContactAnalysis
except ImportError:
    try:
        from ..bonds import ContactAnalysis
    except ImportError:
        ContactAnalysis = None

def generate_interface_html(
    analysis: InterfaceAnalysis,
    cx: Complex,
    title: str = "Protein Interface Analysis",
    color_scheme: str = "default",
    box_size: int = 50 
) -> str:
    """
    Generate an interactive HTML visualization of interface contacts.
    
    Args:
        analysis: InterfaceAnalysis result
        cx: Complex structure
        title: Page title
        color_scheme: 'default', 'heatmap', or 'grouped'
    
    Returns:
        HTML string ready to save or display
    """
    
    # Color schemes
    schemes = {
        "default": {
            "interface": "#ff6b6b",
            "non_interface": "#e8e8e8",
            "group_A": "#4ecdc4",
            "group_B": "#ffe66d",
        },
        "heatmap": {
            # Will use gradient based on contact count
            "low": "#fff5f0",
            "high": "#7f0000",
            "non_interface": "#f0f0f0",
        },
        "grouped": {
            "group_A_interface": "#3498db",
            "group_B_interface": "#e74c3c",
            "both_groups": "#9b59b6",
            "non_interface": "#ecf0f1",
        }
    }
    
    colors = schemes.get(color_scheme, schemes["default"])
    
    # Build HTML sections
    html_chains = []
    
    for chain_id in sorted(analysis.chain_data.keys()):
        res_data_list = analysis.chain_data[chain_id]
        
        if not res_data_list:
            continue
        
        is_group_A = chain_id in analysis.group_A_chains
        is_group_B = chain_id in analysis.group_B_chains
        
        # Determine chain label
        if is_group_A and is_group_B:
            group_label = "Groups A & B"
        elif is_group_A:
            group_label = "Group A"
        elif is_group_B:
            group_label = "Group B"
        else:
            group_label = "Other"
        
        html_chains.append(
            _render_chain_table(
                chain_id, 
                res_data_list, 
                group_label,
                is_group_A,
                is_group_B,
                color_scheme,
                colors
            )
        )
    
    # Summary statistics
    summary_A = analysis.get_group_summary("A")
    summary_B = analysis.get_group_summary("B")
    
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
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .chain-section {{
            margin: 30px 0;
        }}
        .chain-header {{
            background: #34495e;
            color: white;
            padding: 12px 15px;
            border-radius: 6px 6px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .chain-header h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .chain-header .badge {{
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: normal;
        }}
        .residue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax({box_size}px, 1fr));
            gap: 4px;
            padding: 15px;
            background: #fafafa;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 6px 6px;
        }}
        .residue-cell {{
            aspect-ratio: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 11px;
            padding: 4px;
            position: relative;
        }}
        .residue-cell:hover {{
            transform: scale(1.15);
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .residue-cell .resname {{
            font-weight: bold;
            font-size: 12px;
        }}
        .residue-cell .resseq {{
            font-size: 10px;
            opacity: 0.8;
        }}
        .residue-cell .contacts {{
            position: absolute;
            top: 2px;
            right: 2px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            max-width: 300px;
        }}
        .tooltip.show {{
            display: block;
        }}
        .contact-list {{
            margin-top: 5px;
            padding-top: 5px;
            border-top: 1px solid rgba(255,255,255,0.3);
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Group A Chains</h3>
                <div class="value">{', '.join(summary_A['chains'])}</div>
                <div class="label">{summary_A['interface_residues']} interface residues ({summary_A['interface_percentage']:.1f}%)</div>
            </div>
            <div class="summary-card">
                <h3>Group B Chains</h3>
                <div class="value">{', '.join(summary_B['chains'])}</div>
                <div class="label">{summary_B['interface_residues']} interface residues ({summary_B['interface_percentage']:.1f}%)</div>
            </div>
            <div class="summary-card">
                <h3>Total Contacts</h3>
                <div class="value">{analysis.total_contacts}</div>
                <div class="label">Atom-atom contacts within {analysis.cutoff} Å</div>
            </div>
            {f'''<div class="summary-card">
                <h3>Buried Surface Area</h3>
                <div class="value">{analysis.bsa_total:.0f} Ų</div>
                <div class="label">{analysis.note}</div>
            </div>''' if analysis.bsa_total else ''}
        </div>
        
        <div class="legend">
            <strong>Legend:</strong>
            <div class="legend-item">
                <div class="legend-box" style="background: {colors.get('interface', colors.get('group_A_interface', '#ff6b6b'))}"></div>
                <span>Interface Residue</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background: {colors.get('non_interface', '#e8e8e8')}"></div>
                <span>Non-Interface</span>
            </div>
        </div>
        
        <div style="margin: 25px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <label style="font-weight: bold; color: #2c3e50;">Box Size:</label>
                <input type="range" id="boxSizeSlider" min="30" max="100" value="{box_size}" 
                       style="flex: 1; min-width: 200px;">
                <span id="boxSizeValue" style="font-weight: bold; color: #3498db; min-width: 60px;">{box_size}px</span>
                <button onclick="resetBoxSize()" 
                        style="background: #3498db; color: white; border: none; padding: 8px 16px; 
                               border-radius: 6px; cursor: pointer; font-weight: 600;">
                    Reset
                </button>
            </div>
        </div>
        
        {''.join(html_chains)}
        
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        const tooltip = document.getElementById('tooltip');
        
        // Tooltip functionality
        document.querySelectorAll('.residue-cell').forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const info = JSON.parse(cell.dataset.info);
                let html = `
                    <div><strong>${{info.resname}} ${{info.chain}}:${{info.resseq}}</strong></div>
                    <div>Interface: ${{info.is_interface ? 'Yes' : 'No'}}</div>
                    <div>Contacts: ${{info.contact_count}}</div>
                `;
                
                if (info.partners && info.partners.length > 0) {{
                    html += `<div class="contact-list">Partners: ${{info.partners.join(', ')}}</div>`;
                }}

                if (info.sasa_details) {{
                    html += `<div class="contact-list">
                        <div style="font-weight:bold; margin-bottom:3px;">SASA (Å²):</div>
                        <div>Unbound: &nbsp;${{info.sasa_details.unbound.toFixed(1)}}</div>
                        <div>Bound: &nbsp;&nbsp;&nbsp;${{info.sasa_details.bound.toFixed(1)}}</div>
                        <div>Buried: &nbsp;&nbsp;${{info.sasa_details.delta.toFixed(1)}}</div>
                        <div>Buried frac: ${{(info.sasa_details.buried_fraction * 100).toFixed(1)}}%</div>
                    </div>`;
                }}

                tooltip.innerHTML = html;
                tooltip.classList.add('show');
            }});

            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }});

            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});

        // Box size control functionality
        const boxSizeSlider = document.getElementById('boxSizeSlider');
        const boxSizeValue = document.getElementById('boxSizeValue');
        const residueGrids = document.querySelectorAll('.residue-grid');
        
        function updateBoxSize(size) {{
            residueGrids.forEach(grid => {{
                grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${{size}}px, 1fr))`;
            }});
            boxSizeValue.textContent = size + 'px';
        }}
        
        if (boxSizeSlider) {{
            boxSizeSlider.addEventListener('input', (e) => {{
                updateBoxSize(e.target.value);
            }});
        }}
        
        function resetBoxSize() {{
            const defaultSize = {box_size};
            if (boxSizeSlider) {{
                boxSizeSlider.value = defaultSize;
                updateBoxSize(defaultSize);
            }}
        }}
    </script>
</body>
</html>
"""
    
    return html


def _render_chain_table(
    chain_id: str,
    res_data_list: List[ResidueInterfaceData],
    group_label: str,
    is_group_A: bool,
    is_group_B: bool,
    color_scheme: str,
    colors: Dict
) -> str:
    """Render a single chain's residue grid."""
    
    # Calculate statistics
    total = len(res_data_list)
    interface_count = sum(1 for rd in res_data_list if rd.is_interface)
    total_contacts = sum(rd.contact_count for rd in res_data_list)
    
    # Build residue cells
    cells = []
    for rd in res_data_list:
        # Determine color
        if color_scheme == "heatmap":
            if rd.is_interface:
                # Gradient from low to high
                max_contacts = max((r.contact_count for r in res_data_list if r.is_interface), default=1)
                intensity = rd.contact_count / max_contacts if max_contacts > 0 else 0
                # Interpolate between low and high
                color = _interpolate_color(colors["low"], colors["high"], intensity)
            else:
                color = colors["non_interface"]
        elif color_scheme == "grouped":
            if rd.is_interface:
                if is_group_A and is_group_B:
                    color = colors["both_groups"]
                elif is_group_A:
                    color = colors["group_A_interface"]
                elif is_group_B:
                    color = colors["group_B_interface"]
                else:
                    color = "#95a5a6"
            else:
                color = colors["non_interface"]
        else:  # default
            color = colors["interface"] if rd.is_interface else colors["non_interface"]
        
        # Text color (dark on light backgrounds, light on dark)
        text_color = "#000" if _is_light_color(color) else "#fff"
        
        # Build cell data
        info = {
            "resname": rd.residue.resname,
            "chain": chain_id,
            "resseq": f"{rd.residue.resseq}{rd.residue.icode}".strip(),
            "is_interface": rd.is_interface,
            "contact_count": rd.contact_count,
            "partners": rd.partner_residues[:5],  # Limit to first 5 for tooltip
            "sasa_details": rd.sasa_details,
        }

        import json
        cell_html = f"""
        <div class="residue-cell" 
             style="background: {color}; color: {text_color};"
             data-info='{json.dumps(info)}'>
            <div class="resname">{rd.residue.resname}</div>
            <div class="resseq">{info['resseq']}</div>
            {f'<div class="contacts">{rd.contact_count}</div>' if rd.is_interface else ''}
        </div>
        """
        cells.append(cell_html)
    
    return f"""
    <div class="chain-section">
        <div class="chain-header">
            <h2>Chain {chain_id}</h2>
            <div>
                <span class="badge">{group_label}</span>
                <span class="badge">{interface_count}/{total} interface ({100*interface_count/total:.0f}%)</span>
                <span class="badge">{total_contacts} contacts</span>
            </div>
        </div>
        <div class="residue-grid">
            {''.join(cells)}
        </div>
    </div>
    """


def _interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    # Parse hex colors
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    # Interpolate
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    
    return f"#{r:02x}{g:02x}{b:02x}"


def _is_light_color(hex_color: str) -> bool:
    """Determine if a color is light (for contrast)."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    # Calculate perceived brightness
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 128


# ============ Alternative: Simple Table Format ============

def generate_interface_table(
    analysis: InterfaceAnalysis,
    chains: Optional[List[str]] = None,
    show_non_interface: bool = False
) -> str:
    """
    Generate a simple text table of interface residues.
    
    Args:
        analysis: InterfaceAnalysis result
        chains: Specific chains to show (None = all)
        show_non_interface: Include non-interface residues
    
    Returns:
        Formatted text table
    """
    chains = chains or sorted(analysis.chain_data.keys())
    
    lines = []
    lines.append("=" * 90)
    lines.append(f"{'Chain':<8} {'Residue':<12} {'Type':<10} {'Contacts':<10} {'Partners':<40}")
    lines.append("=" * 90)
    
    for chain_id in chains:
        if chain_id not in analysis.chain_data:
            continue
        
        for rd in analysis.chain_data[chain_id]:
            if not show_non_interface and not rd.is_interface:
                continue
            
            status = "Interface" if rd.is_interface else "-"
            partners = ", ".join(rd.partner_residues[:3])
            if len(rd.partner_residues) > 3:
                partners += f" +{len(rd.partner_residues)-3} more"
            
            lines.append(
                f"{chain_id:<8} {rd.id_str:<12} {status:<10} {rd.contact_count:<10} {partners:<40}"
            )
    
    lines.append("=" * 90)
    return "\n".join(lines)



def generate_interface_html_with_3d(
    cx: Complex,
    analysis: InterfaceAnalysis,
    title: str = "Protein Interface Analysis with 3D Structure",
    color_scheme: str = "default",
    box_size: int = 50,
    show_3d: bool = True,
    cutoff_min: float = 3.0,
    cutoff_max: float = 8.0,
    cutoff_step: float = 0.5,
    contact_analysis=None,
) -> str:
    """
    Generate an interactive HTML visualization with both 2D grid and 3D structure views.
    
    Args:
        cx: Complex structure
        analysis: InterfaceAnalysis result
        title: Page title
        color_scheme: 'default', 'heatmap', or 'grouped'
        box_size: Initial size of residue boxes in 2D grid
        show_3d: Include 3D structure viewer
        cutoff_min: Minimum cutoff distance for slider
        cutoff_max: Maximum cutoff distance for slider
        cutoff_step: Step size for cutoff slider
    
    Returns:
        HTML string with interactive visualization
    """
    
    # Color schemes
    schemes = {
        "default": {
            "interface": "#ff6b6b",
            "non_interface": "#e8e8e8",
            "group_A": "#4ecdc4",
            "group_B": "#ffe66d",
            "group_A_3d": "0x4ecdc4",
            "group_B_3d": "0xffe66d",
            "interface_3d": "0xff6b6b",
        },
        "heatmap": {
            "low": "#fff5f0",
            "high": "#7f0000",
            "non_interface": "#f0f0f0",
            "interface_3d": "0xff0000",
            "interface": "#ff6b6b",
            "non_interface": "#e8e8e8",
        },
        "grouped": {
            "group_A_interface": "#3498db",
            "group_B_interface": "#e74c3c",
            "both_groups": "#9b59b6",
            "non_interface": "#ecf0f1",
            "group_A_3d": "0x3498db",
            "group_B_3d": "0xe74c3c",
            "interface": "#ff6b6b",
            "non_interface": "#e8e8e8",
        }
    }
    
    colors = schemes.get(color_scheme, schemes["default"])
    
    # Get PDB content for 3D viewer with fallback
    pdb_content = ""
    if hasattr(cx, 'to_pdb_string'):
        try:
            pdb_content = cx.to_pdb_string()
        except Exception as e:
            print(f"Warning: Could not generate PDB string: {e}")
    
    # If still no PDB content, try to generate minimal PDB from Complex object
    if not pdb_content:
        try:
            pdb_lines = []
            pdb_lines.append("HEADER    GENERATED STRUCTURE")
            atom_num = 1
            for chain_id, chain in cx.chains.items():
                for residue in chain.iter_residues():
                    for atom in residue.iter_atoms():
                        coord = atom.coord
                        pdb_lines.append(
                            f"ATOM  {atom_num:5d}  {atom.name:<4s}{residue.resname:>3s} "
                            f"{chain_id}{residue.resseq:4d}{residue.icode}   "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           "
                            f"{atom.element:>2s}"
                        )
                        atom_num += 1
            pdb_lines.append("END")
            pdb_content = "\n".join(pdb_lines)
        except Exception as e:
            print(f"Warning: Could not generate minimal PDB: {e}")
            pdb_content = "HEADER    NO STRUCTURE DATA\nEND\n"
    
    # Build HTML sections for 2D grid
    html_chains = []
    for chain_id in sorted(analysis.chain_data.keys()):
        res_data_list = analysis.chain_data[chain_id]
        
        if not res_data_list:
            continue
        
        is_group_A = chain_id in analysis.group_A_chains
        is_group_B = chain_id in analysis.group_B_chains
        
        # Determine chain label
        if is_group_A and is_group_B:
            group_label = "Groups A & B"
        elif is_group_A:
            group_label = "Group A"
        elif is_group_B:
            group_label = "Group B"
        else:
            group_label = "Other"
        
        html_chains.append(
            _render_chain_table(
                chain_id, 
                res_data_list, 
                group_label,
                is_group_A,
                is_group_B,
                color_scheme,
                colors
            )
        )
    
    # Summary statistics
    summary_A = analysis.get_group_summary("A")
    summary_B = analysis.get_group_summary("B")
    
    # Prepare interface residue data for 3D highlighting
    interface_residues_json = _prepare_interface_residues_for_3d(analysis)

    # Prepare contact pairs for 3D dashed-line drawing (with optional typed contact info)
    interface_contacts_json = _prepare_interface_contacts_for_3d(analysis, cx, contact_analysis)

    # Contact type colors (same palette as contacts_viz.py)
    _contact_type_colors = {
        "hydrogen_bond": "#3498db",
        "salt_bridge":   "#e74c3c",
        "disulfide":     "#f39c12",
        "hydrophobic":   "#2ecc71",
        "pi_stacking":   "#9b59b6",
    }
    contact_type_colors_json = json.dumps(_contact_type_colors)
    contact_type_labels_json = json.dumps({
        "hydrogen_bond": "H-Bond",
        "salt_bridge":   "Salt Bridge",
        "disulfide":     "Disulfide",
        "hydrophobic":   "Hydrophobic",
        "pi_stacking":   "π-Stacking",
    })
    contact_bond_radius_json = json.dumps({
        "hydrogen_bond": 0.12,
        "salt_bridge":   0.18,
        "disulfide":     0.25,
        "hydrophobic":   0.08,
        "pi_stacking":   0.15,
    })
    has_typed_contacts = contact_analysis is not None and hasattr(contact_analysis, 'contacts') and len(contact_analysis.contacts) > 0

    # --- Build CSV data for Data Tables section ---
    _iface_residue_csv = []
    for _chain_id in sorted(analysis.chain_data.keys()):
        _is_A = _chain_id in analysis.group_A_chains
        _is_B = _chain_id in analysis.group_B_chains
        _group = "A & B" if (_is_A and _is_B) else ("A" if _is_A else ("B" if _is_B else "—"))
        for _rd in analysis.chain_data[_chain_id]:
            _res = _rd.residue
            _sasa = _rd.sasa_details or {}
            _iface_residue_csv.append({
                "chain":              _chain_id,
                "resseq":             f"{_res.resseq}{getattr(_res, 'icode', '').strip()}",
                "resname":            _res.resname,
                "group":              _group,
                "is_interface":       _rd.is_interface,
                "contact_count":      _rd.contact_count,
                "partner_residues":   "; ".join(sorted(_rd.partner_residues)),
                "sasa_unbound_A2":    round(_sasa["unbound"], 2) if "unbound" in _sasa else "",
                "sasa_bound_A2":      round(_sasa["bound"],   2) if "bound"   in _sasa else "",
                "sasa_buried_A2":     round(_sasa["delta"],   2) if "delta"   in _sasa else "",
                "buried_fraction_pct": round(_sasa["buried_fraction"] * 100, 1) if "buried_fraction" in _sasa else "",
            })
    iface_residue_csv_json = json.dumps(_iface_residue_csv)

    # Pairs CSV — typed if contact_analysis available, otherwise from partner_residues
    _iface_pairs_csv = []
    if has_typed_contacts:
        for _c in contact_analysis.contacts:
            _p1 = _c.residue1_id.split()
            _p2 = _c.residue2_id.split()
            def _pr(p):
                rn = p[0] if p else ""
                ch, rs = "", ""
                if len(p) > 1 and ":" in p[1]:
                    ch, rs = p[1].split(":", 1)
                return rn, ch, rs
            rn1, ch1, rs1 = _pr(_p1)
            rn2, ch2, rs2 = _pr(_p2)
            _iface_pairs_csv.append({
                "residue1": _c.residue1_id, "chain1": ch1, "resseq1": rs1, "resname1": rn1,
                "residue2": _c.residue2_id, "chain2": ch2, "resseq2": rs2, "resname2": rn2,
                "bond_type": _c.type.value,
                "distance_angstrom": round(_c.distance, 3),
            })
    else:
        # Derive untyped pairs from partner_residues (avoid duplicates)
        _seen_pairs = set()
        for _chain_id in sorted(analysis.chain_data.keys()):
            for _rd in analysis.chain_data[_chain_id]:
                if not _rd.is_interface:
                    continue
                _res = _rd.residue
                _rid = f"{_res.resname} {_chain_id}:{_res.resseq}{getattr(_res, 'icode', '').strip()}"
                for _partner in _rd.partner_residues:
                    _key = tuple(sorted([_rid, _partner]))
                    if _key not in _seen_pairs:
                        _seen_pairs.add(_key)
                        _iface_pairs_csv.append({
                            "residue1": _rid,
                            "residue2": _partner,
                            "bond_type": "—",
                            "distance_angstrom": "",
                        })
    iface_pairs_csv_json = json.dumps(_iface_pairs_csv)

    # Prepare complex data for recomputation
    complex_data = {
        "group_A_chains": analysis.group_A_chains,
        "group_B_chains": analysis.group_B_chains,
        "current_cutoff": analysis.cutoff
    }
    
    # Escape backticks in PDB content for JavaScript template literal
    pdb_content_escaped = pdb_content.replace('`', '\\`')
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1800px;
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
        
        /* 3D Viewer Section */
        .viewer-section {{
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        .viewer-controls {{
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .control-group label {{
            font-size: 12px;
            font-weight: 600;
            color: #2c3e50;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        #viewer-3d {{
            width: 100%;
            height: 600px;
            background: #000;
            border-radius: 6px;
            position: relative;
        }}
        .slider-container {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            min-width: 250px;
        }}
        .slider-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        input[type="range"] {{
            flex: 1;
            min-width: 150px;
        }}
        .slider-value {{
            min-width: 60px;
            font-weight: bold;
            color: #3498db;
            font-size: 14px;
        }}
        button {{
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #2980b9;
        }}
        button:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
        }}
        .spinner {{
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Residue info panel (click popup) */
        #residue-info-panel {{
            display: none;
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 280px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            padding: 16px;
            z-index: 1000;
            font-size: 13px;
        }}
        #residue-info-panel .rip-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
        #residue-info-panel .rip-title {{
            font-size: 15px;
            font-weight: 700;
            color: #2c3e50;
        }}
        #residue-info-panel .rip-close {{
            background: #e74c3c;
            color: white;
            border: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 14px;
            line-height: 1;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #residue-info-panel .rip-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        #residue-info-panel .rip-label {{
            font-weight: 600;
            color: #7f8c8d;
        }}
        #residue-info-panel .rip-value {{
            color: #2c3e50;
            font-weight: 500;
        }}
        #residue-info-panel .rip-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }}

        /* Summary Section */
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        /* Chain Grid Section */
        .chain-section {{
            margin: 30px 0;
        }}
        .chain-header {{
            background: #34495e;
            color: white;
            padding: 12px 15px;
            border-radius: 6px 6px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .chain-header h2 {{
            margin: 0;
            font-size: 18px;
        }}
        .chain-header .badge {{
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: normal;
        }}
        .residue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax({box_size}px, 1fr));
            gap: 4px;
            padding: 15px;
            background: #fafafa;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 6px 6px;
        }}
        .residue-cell {{
            aspect-ratio: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 11px;
            padding: 4px;
            position: relative;
        }}
        .residue-cell:hover {{
            transform: scale(1.15);
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .residue-cell .resname {{
            font-weight: bold;
            font-size: 12px;
        }}
        .residue-cell .resseq {{
            font-size: 10px;
            opacity: 0.8;
        }}
        .residue-cell .contacts {{
            position: absolute;
            top: 2px;
            right: 2px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            max-width: 300px;
        }}
        .tooltip.show {{
            display: block;
        }}
        .contact-list {{
            margin-top: 5px;
            padding-top: 5px;
            border-top: 1px solid rgba(255,255,255,0.3);
            font-size: 11px;
        }}
        
        .section-toggle {{
            background: #34495e;
            color: white;
            padding: 12px 15px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
        }}
        .section-toggle:hover {{
            background: #2c3e50;
        }}
        .section-content {{
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .section-content.collapsed {{
            max-height: 0;
        }}

        /* Data Tables section */
        .dt-tabs {{
            display: flex; gap: 8px; margin: 18px 0 0 0;
            border-bottom: 2px solid #dee2e6; padding-bottom: 0;
        }}
        .dt-tab {{
            background: none; border: none; border-bottom: 3px solid transparent;
            padding: 8px 22px; font-size: 14px; font-weight: 600; color: #6c757d;
            cursor: pointer; margin-bottom: -2px; border-radius: 6px 6px 0 0;
            transition: color 0.15s, border-color 0.15s;
        }}
        .dt-tab.active {{ color: #34495e; border-bottom-color: #34495e; background: #f4f6f8; }}
        .dt-tab:hover:not(.active) {{ color: #495057; background: #f8f9fa; }}
        .dt-panel {{ display: none; }}
        .dt-panel.active {{ display: block; }}
        .dt-toolbar {{
            display: flex; align-items: center; gap: 12px;
            padding: 12px 0 10px 0; flex-wrap: wrap;
        }}
        .dt-toolbar input {{
            padding: 6px 12px; border: 1px solid #ced4da; border-radius: 6px;
            font-size: 13px; width: 220px; outline: none;
        }}
        .dt-toolbar input:focus {{ border-color: #34495e; box-shadow: 0 0 0 2px rgba(52,73,94,0.15); }}
        .dt-dl-btn {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white; border: none; padding: 6px 16px; border-radius: 6px;
            font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity 0.15s;
        }}
        .dt-dl-btn:hover {{ opacity: 0.88; }}
        .dt-scroll {{
            overflow-x: auto; border-radius: 8px; border: 1px solid #e9ecef;
            max-height: 480px; overflow-y: auto;
        }}
        .dt-table {{
            width: 100%; border-collapse: collapse; font-size: 13px;
        }}
        .dt-table thead th {{
            background: #f4f6f8; color: #2c3e50; font-weight: 700;
            padding: 10px 14px; text-align: left; position: sticky; top: 0; z-index: 1;
            border-bottom: 2px solid #dee2e6; white-space: nowrap;
            cursor: pointer; user-select: none;
        }}
        .dt-table thead th:hover {{ background: #e8ecf0; }}
        .dt-table thead th .sort-icon {{ color: #aaa; margin-left: 4px; font-size: 11px; }}
        .dt-table tbody tr:nth-child(even) {{ background: #f9f9fb; }}
        .dt-table tbody tr:hover {{ background: #edf0f3; }}
        .dt-table tbody td {{
            padding: 8px 14px; border-bottom: 1px solid #f0f0f0; white-space: nowrap;
        }}
        .dt-table .partners-cell {{ white-space: normal; max-width: 260px; font-size: 12px; color: #555; }}
        .dt-badge {{
            display: inline-block; padding: 2px 9px; border-radius: 12px;
            font-size: 11px; font-weight: 700; color: white;
            text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .dt-no-rows {{ padding: 24px; text-align: center; color: #888; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Contacts</h3>
                <div class="value" id="total-contacts">{analysis.total_contacts}</div>
                <div class="label">Atom-atom contacts at {analysis.cutoff}Å cutoff</div>
            </div>
            
            <div class="summary-card">
                <h3>Group A Interface</h3>
                <div class="value">{summary_A['interface_residues']}</div>
                <div class="label">
                    {summary_A['interface_percentage']:.1f}% of {summary_A['total_residues']} residues<br>
                    Chains: {', '.join(summary_A['chains'])}
                </div>
            </div>
            
            <div class="summary-card">
                <h3>Group B Interface</h3>
                <div class="value">{summary_B['interface_residues']}</div>
                <div class="label">
                    {summary_B['interface_percentage']:.1f}% of {summary_B['total_residues']} residues<br>
                    Chains: {', '.join(summary_B['chains'])}
                </div>
            </div>
            
            {f'''<div class="summary-card">
                <h3>Buried Surface Area</h3>
                <div class="value">{analysis.bsa_total:.0f}</div>
                <div class="label">Å² ({analysis.note})</div>
            </div>''' if analysis.bsa_total else ''}
        </div>

        {_render_3d_viewer_section(cutoff_min, cutoff_max, cutoff_step, analysis.cutoff) if show_3d else ''}
        
        <div class="section-toggle" onclick="toggleSection('grid-section')">
            <h2 style="margin: 0;">2D Residue Grid View</h2>
            <span id="grid-toggle-icon">▼</span>
        </div>
        
        <div id="grid-section" class="section-content">
            <div class="legend">
                <div style="font-weight: 600; margin-right: 10px;">Legend:</div>
                <div class="legend-item">
                    <div class="legend-box" style="background: {colors['interface']};"></div>
                    <span>Interface Residue</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: {colors['non_interface']};"></div>
                    <span>Non-Interface</span>
                </div>
            </div>
            
            <div style="margin: 20px 0;">
                <label style="font-weight: 600; margin-right: 10px;">Box Size:</label>
                <input type="range" id="boxSizeSlider" min="30" max="100" value="{box_size}" 
                       style="width: 200px; vertical-align: middle;">
                <span id="boxSizeValue" style="margin-left: 10px; font-weight: bold;">{box_size}px</span>
                <button onclick="resetBoxSize()" 
                        style="margin-left: 15px;">
                    Reset
                </button>
            </div>
            
            {''.join(html_chains)}
        </div>

        <div class="section-toggle" onclick="toggleSection('data-section')">
            <h2 style="margin: 0;">📊 Data Tables</h2>
            <span id="data-toggle-icon">▼</span>
        </div>
        <div id="data-section" class="section-content">
            <div class="dt-tabs">
                <button class="dt-tab active" id="dt-tab-residues"
                        onclick="showDtTab('residues')">Interface Residues</button>
                <button class="dt-tab" id="dt-tab-pairs"
                        onclick="showDtTab('pairs')">Contact Pairs</button>
            </div>

            <!-- Residues panel -->
            <div id="dt-panel-residues" class="dt-panel active">
                <div class="dt-toolbar">
                    <input type="text" id="dt-residues-search" placeholder="🔍 Filter residues…"
                           oninput="filterDt('residues')">
                    <button class="dt-dl-btn" onclick="downloadDt('residues')">⬇ Download CSV</button>
                    <span id="dt-residues-count" style="color:#888; font-size:12px;"></span>
                </div>
                <div class="dt-scroll">
                    <table class="dt-table" id="dt-residues-table">
                        <thead><tr>
                            <th onclick="sortDt('residues',0)">Chain <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',1)">ResSeq <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',2)">Residue <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',3)">Group <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',4)">Interface <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',5)">Contacts <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',6)">SASA Unbound (Å²) <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',7)">SASA Bound (Å²) <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',8)">Buried (Å²) <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('residues',9)">Buried % <span class="sort-icon">⇅</span></th>
                            <th>Partner Residues</th>
                        </tr></thead>
                        <tbody id="dt-residues-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- Pairs panel -->
            <div id="dt-panel-pairs" class="dt-panel">
                <div class="dt-toolbar">
                    <input type="text" id="dt-pairs-search" placeholder="🔍 Filter pairs…"
                           oninput="filterDt('pairs')">
                    <button class="dt-dl-btn" onclick="downloadDt('pairs')">⬇ Download CSV</button>
                    <span id="dt-pairs-count" style="color:#888; font-size:12px;"></span>
                </div>
                <div class="dt-scroll">
                    <table class="dt-table" id="dt-pairs-table">
                        <thead><tr>
                            <th onclick="sortDt('pairs',0)">Residue 1 <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('pairs',1)">Residue 2 <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('pairs',2)">Bond Type <span class="sort-icon">⇅</span></th>
                            <th onclick="sortDt('pairs',3)">Distance (Å) <span class="sort-icon">⇅</span></th>
                        </tr></thead>
                        <tbody id="dt-pairs-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="tooltip" id="tooltip"></div>
    </div>

    <script>
        // Global variables
        let viewer3d = null;
        let pdbData = `{pdb_content_escaped}`;
        let interfaceResidues = {interface_residues_json};
        let complexData = {json.dumps(complex_data)};
        let currentCutoff = {analysis.cutoff};
        let interfaceContacts = {interface_contacts_json};
        let showInterfaceContacts = true;
        let interfaceContactShapes = [];
        const interfaceContactColors = {contact_type_colors_json};
        const interfaceContactLabels = {contact_type_labels_json};
        const interfaceContactRadius = {contact_bond_radius_json};
        const hasTypedContacts = {'true' if has_typed_contacts else 'false'};
        
        // Initialize 3D viewer
        {_render_3d_viewer_javascript() if show_3d else ''}
        
        // Tooltip functionality for 2D grid
        const tooltip = document.getElementById('tooltip');
        
        document.querySelectorAll('.residue-cell').forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const info = JSON.parse(cell.dataset.info);
                let html = `
                    <div><strong>${{info.resname}} ${{info.chain}}:${{info.resseq}}</strong></div>
                    <div>Interface: ${{info.is_interface ? 'Yes' : 'No'}}</div>
                    <div>Contacts: ${{info.contact_count}}</div>
                `;
                
                if (info.partners && info.partners.length > 0) {{
                    html += `<div class="contact-list">Partners: ${{info.partners.join(', ')}}</div>`;
                }}

                if (info.sasa_details) {{
                    html += `<div class="contact-list">
                        <div style="font-weight:bold; margin-bottom:3px;">SASA (Å²):</div>
                        <div>Unbound: &nbsp;${{info.sasa_details.unbound.toFixed(1)}}</div>
                        <div>Bound: &nbsp;&nbsp;&nbsp;${{info.sasa_details.bound.toFixed(1)}}</div>
                        <div>Buried: &nbsp;&nbsp;${{info.sasa_details.delta.toFixed(1)}}</div>
                        <div>Buried frac: ${{(info.sasa_details.buried_fraction * 100).toFixed(1)}}%</div>
                    </div>`;
                }}

                tooltip.innerHTML = html;
                tooltip.classList.add('show');
            }});

            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }});

            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});
        
        // Box size control for 2D grid
        const boxSizeSlider = document.getElementById('boxSizeSlider');
        const boxSizeValue = document.getElementById('boxSizeValue');
        const residueGrids = document.querySelectorAll('.residue-grid');
        
        function updateBoxSize(size) {{
            residueGrids.forEach(grid => {{
                grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${{size}}px, 1fr))`;
            }});
            boxSizeValue.textContent = size + 'px';
        }}
        
        if (boxSizeSlider) {{
            boxSizeSlider.addEventListener('input', (e) => {{
                updateBoxSize(e.target.value);
            }});
        }}
        
        function resetBoxSize() {{
            const defaultSize = {box_size};
            if (boxSizeSlider) {{
                boxSizeSlider.value = defaultSize;
                updateBoxSize(defaultSize);
            }}
        }}

        // Section toggle functionality
        function toggleSection(sectionId) {{
            const section = document.getElementById(sectionId);
            const icon = document.getElementById(sectionId.replace('-section', '-toggle-icon'));

            if (section.classList.contains('collapsed')) {{
                section.classList.remove('collapsed');
                icon.textContent = '▼';
            }} else {{
                section.classList.add('collapsed');
                icon.textContent = '▶';
            }}
        }}

        // ── Data Tables ──────────────────────────────────────────────
        const _dtResidueData = {iface_residue_csv_json};
        const _dtPairsData   = {iface_pairs_csv_json};

        const _dtContactColors = {json.dumps(_contact_type_colors)};
        const _dtContactLabels = {{
            'hydrogen_bond': 'H-Bond', 'salt_bridge': 'Salt Bridge',
            'disulfide': 'Disulfide', 'hydrophobic': 'Hydrophobic', 'pi_stacking': 'π-Stacking',
        }};

        const _dtState = {{
            residues: {{ rows: _dtResidueData.slice(), sortCol: -1, sortAsc: true }},
            pairs:    {{ rows: _dtPairsData.slice(),   sortCol: -1, sortAsc: true }},
        }};

        function _badge(type) {{
            const c = _dtContactColors[type];
            const l = _dtContactLabels[type] || type;
            return c ? `<span class="dt-badge" style="background:${{c}}">${{l}}</span>` : (type || '—');
        }}

        function _escCsv(v) {{
            const s = (v === null || v === undefined) ? '' : String(v);
            return (s.includes(',') || s.includes('"') || s.includes('\\n'))
                ? '"' + s.replace(/"/g, '""') + '"' : s;
        }}
        function _toCsv(rows) {{
            if (!rows.length) return '';
            const h = Object.keys(rows[0]);
            return [h.join(','), ...rows.map(r => h.map(k => _escCsv(r[k])).join(','))].join('\\n');
        }}
        function _dlBlob(content, filename) {{
            const blob = new Blob([content], {{type: 'text/csv'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = filename; a.click();
            URL.revokeObjectURL(url);
        }}

        function _renderResidues(rows) {{
            const tb = document.getElementById('dt-residues-body');
            if (!rows.length) {{
                tb.innerHTML = '<tr><td colspan="11" class="dt-no-rows">No residues match the filter.</td></tr>';
            }} else {{
                tb.innerHTML = rows.map(r => `<tr>
                    <td>${{r.chain}}</td>
                    <td>${{r.resseq}}</td>
                    <td>${{r.resname}}</td>
                    <td>${{r.group}}</td>
                    <td>${{r.is_interface ? '✔' : ''}}</td>
                    <td>${{r.contact_count}}</td>
                    <td>${{r.sasa_unbound_A2 !== '' ? r.sasa_unbound_A2 : '—'}}</td>
                    <td>${{r.sasa_bound_A2   !== '' ? r.sasa_bound_A2   : '—'}}</td>
                    <td>${{r.sasa_buried_A2  !== '' ? r.sasa_buried_A2  : '—'}}</td>
                    <td>${{r.buried_fraction_pct !== '' ? r.buried_fraction_pct + '%' : '—'}}</td>
                    <td class="partners-cell">${{r.partner_residues || ''}}</td>
                </tr>`).join('');
            }}
            const el = document.getElementById('dt-residues-count');
            if (el) el.textContent = rows.length + ' row' + (rows.length !== 1 ? 's' : '');
        }}

        function _renderPairs(rows) {{
            const tb = document.getElementById('dt-pairs-body');
            if (!rows.length) {{
                tb.innerHTML = '<tr><td colspan="4" class="dt-no-rows">No pairs match the filter.</td></tr>';
            }} else {{
                tb.innerHTML = rows.map(r => `<tr>
                    <td>${{r.residue1}}</td>
                    <td>${{r.residue2}}</td>
                    <td>${{_badge(r.bond_type)}}</td>
                    <td>${{r.distance_angstrom !== '' ? (typeof r.distance_angstrom === 'number' ? r.distance_angstrom.toFixed(3) : r.distance_angstrom) : '—'}}</td>
                </tr>`).join('');
            }}
            const el = document.getElementById('dt-pairs-count');
            if (el) el.textContent = rows.length + ' row' + (rows.length !== 1 ? 's' : '');
        }}

        function _render(panel) {{
            if (panel === 'residues') _renderResidues(_dtState.residues.rows);
            else _renderPairs(_dtState.pairs.rows);
        }}

        function filterDt(panel) {{
            const q = (document.getElementById('dt-' + panel + '-search').value || '').toLowerCase();
            const src = panel === 'residues' ? _dtResidueData : _dtPairsData;
            _dtState[panel].rows = q
                ? src.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)))
                : src.slice();
            _dtState[panel].sortCol = -1;
            _render(panel);
        }}

        function sortDt(panel, col) {{
            const st = _dtState[panel];
            st.sortAsc = (st.sortCol === col) ? !st.sortAsc : true;
            st.sortCol = col;
            const resKeys  = ['chain','resseq','resname','group','is_interface','contact_count',
                              'sasa_unbound_A2','sasa_bound_A2','sasa_buried_A2','buried_fraction_pct'];
            const pairKeys = ['residue1','residue2','bond_type','distance_angstrom'];
            const key = panel === 'residues' ? resKeys[col] : pairKeys[col];
            if (!key) return;
            st.rows.sort((a, b) => {{
                const av = a[key], bv = b[key];
                const n = (typeof av === 'number' && typeof bv === 'number') ? av - bv
                        : String(av).localeCompare(String(bv), undefined, {{numeric: true}});
                return st.sortAsc ? n : -n;
            }});
            _render(panel);
        }}

        function showDtTab(panel) {{
            ['residues','pairs'].forEach(p => {{
                document.getElementById('dt-panel-' + p).classList.toggle('active', p === panel);
                document.getElementById('dt-tab-'   + p).classList.toggle('active', p === panel);
            }});
        }}

        function downloadDt(panel) {{
            if (panel === 'residues')
                _dlBlob(_toCsv(_dtResidueData), 'interface_residues.csv');
            else
                _dlBlob(_toCsv(_dtPairsData), 'interface_pairs.csv');
        }}

        // Initial render
        _renderResidues(_dtResidueData);
        _renderPairs(_dtPairsData);

    </script>
</body>
</html>
"""
    
    return html


def _render_3d_viewer_section(cutoff_min: float, cutoff_max: float, cutoff_step: float, current_cutoff: float) -> str:
    """Render the 3D viewer section HTML"""
    return f"""
        <div class="viewer-section">
            <h2 style="margin-top: 0;">3D Structure Viewer</h2>

            <div class="viewer-controls">

                <div class="control-group">
                    <label>View Style</label>
                    <div style="display: flex; gap: 10px;">
                        <button onclick="setStyle('cartoon')">Cartoon</button>
                        <button onclick="setStyle('stick')">Stick</button>
                        <button onclick="setStyle('sphere')">Sphere</button>
                    </div>
                </div>

                <div class="control-group">
                    <label>Actions</label>
                    <div style="display: flex; gap: 10px;">
                        <button onclick="resetView()">Reset View</button>
                        <button onclick="toggleSpin()">Toggle Spin</button>
                    </div>
                </div>

                <div class="control-group">
                    <label>Contacts</label>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <label style="display:flex; align-items:center; gap:5px; cursor:pointer; font-size:13px;">
                            <input type="checkbox" id="toggleContactsCheckbox" checked
                                   onchange="toggleInterfaceContacts(this.checked)"
                                   style="cursor:pointer;">
                            Show contact bonds
                        </label>
                    </div>
                </div>
            </div>

            <div id="contact-bond-legend" style="display:none; margin-bottom:12px; padding:8px 12px;
                 background:#f0f0f0; border-radius:6px; font-size:12px;">
                <span style="font-weight:600; margin-right:8px;">Bond types:</span>
                <span id="contact-legend-items"></span>
            </div>
            
            <div style="position: relative;">
                <div id="viewer-3d"></div>
                <div id="residue-info-panel">
                    <div class="rip-header">
                        <span class="rip-title" id="rip-title">Residue</span>
                        <button class="rip-close" onclick="closeResiduePanel()">&#x2715;</button>
                    </div>
                    <div id="rip-body"></div>
                </div>
            </div>
            <p style="font-size:12px; color:#888; margin-top:8px;">
                <i>Click any residue to see details. Interface residues show contact information.</i>
            </p>
        </div>
    """


def _render_3d_viewer_javascript() -> str:
    """Render the JavaScript code for 3D viewer functionality"""
    return """
        // Initialize 3D viewer when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            initViewer3D();
        });
        
        function initViewer3D() {
            console.log('Initializing 3D viewer...');
            const element = document.getElementById('viewer-3d');
            
            if (!element) {
                console.error('3D viewer element not found!');
                return;
            }
            
            if (typeof $3Dmol === 'undefined') {
                console.error('3Dmol.js library not loaded!');
                alert('Error: 3Dmol.js library failed to load. Check your internet connection.');
                return;
            }
            
            if (!pdbData || pdbData.trim() === '' || pdbData === 'HEADER    NO STRUCTURE DATA') {
                console.error('No PDB data available!');
                element.innerHTML = '<div style="color: white; padding: 20px; text-align: center;">' +
                                   'No structure data available<br>PDB file may be empty or failed to load</div>';
                return;
            }
            
            console.log('PDB data length:', pdbData.length, 'characters');
            console.log('Interface residues:', Object.keys(interfaceResidues).length, 'chains');
            
            const config = { backgroundColor: 'white' };
            viewer3d = $3Dmol.createViewer(element, config);
            
            // Load PDB structure
            try {
                viewer3d.addModel(pdbData, 'pdb');
                console.log('PDB model loaded successfully');
            } catch (error) {
                console.error('Error loading PDB model:', error);
                element.innerHTML = '<div style="color: white; padding: 20px; text-align: center;">' +
                                   'Error loading structure<br>' + error.message + '</div>';
                return;
            }
            
            // Apply initial styling
            applyStructureStyle('cartoon');

            // Register click handlers on all atoms
            registerAllClickable();

            viewer3d.zoomTo();
            viewer3d.render();
            drawInterfaceContacts3d();
            buildContactBondLegend();
            console.log('3D viewer initialized successfully');
        }

        function buildContactBondLegend() {
            if (!hasTypedContacts) return;
            // Collect which types actually appear in the data
            const usedTypes = new Set();
            (interfaceContacts || []).forEach(pair => {
                (pair.types || []).forEach(t => usedTypes.add(t));
            });
            if (!usedTypes.size) return;
            const legendEl = document.getElementById('contact-bond-legend');
            const itemsEl  = document.getElementById('contact-legend-items');
            if (!legendEl || !itemsEl) return;
            let html = '';
            usedTypes.forEach(t => {
                const color = interfaceContactColors[t] || '#aaa';
                const label = interfaceContactLabels[t] || t;
                html += `<span style="display:inline-flex; align-items:center; gap:4px; margin-right:12px;">
                    <span style="display:inline-block; width:22px; height:4px; background:${color}; border-radius:2px;"></span>
                    <span>${label}</span>
                </span>`;
            });
            itemsEl.innerHTML = html;
            legendEl.style.display = 'block';
        }
        
        function applyStructureStyle(style) {
            if (!viewer3d) return;
            
            viewer3d.setStyle({}, {});  // Clear all styles
            
            // Style for non-interface residues (gray)
            if (style === 'cartoon') {
                viewer3d.setStyle({}, {cartoon: {color: 'gray', opacity: 0.7}});
            } else if (style === 'stick') {
                viewer3d.setStyle({}, {stick: {color: 'gray', opacity: 0.7}});
            } else if (style === 'sphere') {
                viewer3d.setStyle({}, {sphere: {color: 'gray', opacity: 0.7, scale: 0.3}});
            }
            
            // Highlight interface residues
            highlightInterfaceResidues(style);
            
            viewer3d.render();
        }
        
        function highlightInterfaceResidues(style) {
            if (!viewer3d || !interfaceResidues) return;

            const styleMap = {
                'cartoon': (color) => ({cartoon: {color: color, opacity: 1.0}}),
                'stick': (color) => ({stick: {color: color, opacity: 1.0, radius: 0.25}}),
                'sphere': (color) => ({sphere: {color: color, opacity: 1.0, scale: 0.5}})
            };

            const getStyle = styleMap[style] || styleMap['cartoon'];

            // Highlight each interface residue and make it clickable
            Object.keys(interfaceResidues).forEach(chainId => {
                const residues = interfaceResidues[chainId];

                residues.forEach(res => {
                    const selection = {
                        chain: chainId,
                        resi: res.resseq
                    };

                    // Use different colors for different groups
                    let color = 'red';  // Default interface color
                    if (complexData.group_A_chains.includes(chainId)) {
                        color = 'cyan';
                    } else if (complexData.group_B_chains.includes(chainId)) {
                        color = 'yellow';
                    }

                    viewer3d.setStyle(selection, getStyle(color));

                    // Make each interface residue clickable
                    viewer3d.setClickable(selection, true, function(atom) {
                        showResiduePanel(atom, chainId, res);
                    });
                });
            });
        }

        // Build a fast lookup: "CHAIN:RESSEQ" -> residue data
        function buildInterfaceLookup() {
            const lookup = {};
            Object.keys(interfaceResidues).forEach(chainId => {
                interfaceResidues[chainId].forEach(res => {
                    lookup[chainId + ':' + res.resseq] = { chainId, res };
                });
            });
            return lookup;
        }

        // Make all atoms clickable for basic info; interface ones get enriched info
        function registerAllClickable() {
            if (!viewer3d) return;
            const lookup = buildInterfaceLookup();
            viewer3d.setClickable({}, true, function(atom) {
                const key = atom.chain + ':' + atom.resi;
                if (lookup[key]) {
                    showResiduePanel(atom, lookup[key].chainId, lookup[key].res);
                } else {
                    showResiduePanel(atom, atom.chain, null);
                }
            });
        }

        function showResiduePanel(atom, chainId, resData) {
            const isInterface = resData !== null;
            const isGroupA = complexData.group_A_chains.includes(chainId);
            const isGroupB = complexData.group_B_chains.includes(chainId);

            let groupLabel = 'Other';
            let groupColor = '#95a5a6';
            if (isGroupA) { groupLabel = 'Group A'; groupColor = '#00bcd4'; }
            else if (isGroupB) { groupLabel = 'Group B'; groupColor = '#f39c12'; }

            document.getElementById('rip-title').textContent =
                (atom.resn || '???') + ' ' + chainId + ':' + atom.resi;

            let rows = `
                <div class="rip-row">
                    <span class="rip-label">Residue</span>
                    <span class="rip-value">${atom.resn || 'N/A'}</span>
                </div>
                <div class="rip-row">
                    <span class="rip-label">Chain</span>
                    <span class="rip-value">${chainId}</span>
                </div>
                <div class="rip-row">
                    <span class="rip-label">Position</span>
                    <span class="rip-value">${atom.resi}</span>
                </div>
                <div class="rip-row">
                    <span class="rip-label">Group</span>
                    <span class="rip-value">
                        <span class="rip-badge" style="background:${groupColor};color:white;">${groupLabel}</span>
                    </span>
                </div>
                <div class="rip-row">
                    <span class="rip-label">Interface</span>
                    <span class="rip-value">
                        <span class="rip-badge" style="background:${isInterface ? '#27ae60' : '#bdc3c7'};color:white;">
                            ${isInterface ? 'Yes' : 'No'}
                        </span>
                    </span>
                </div>`;

            if (isInterface && resData) {
                rows += `
                <div class="rip-row">
                    <span class="rip-label">Contacts</span>
                    <span class="rip-value" style="font-weight:700;color:#e74c3c;">${resData.contact_count}</span>
                </div>`;
            }

            if (atom.b !== undefined && atom.b !== 0) {
                rows += `
                <div class="rip-row">
                    <span class="rip-label">B-factor</span>
                    <span class="rip-value">${atom.b.toFixed(2)}</span>
                </div>`;
            }

            document.getElementById('rip-body').innerHTML = rows;
            document.getElementById('residue-info-panel').style.display = 'block';
        }

        function closeResiduePanel() {
            document.getElementById('residue-info-panel').style.display = 'none';
        }

        function setStyle(style) {
            applyStructureStyle(style);
        }

        function resetView() {
            if (!viewer3d) return;
            viewer3d.zoomTo();
            viewer3d.render();
        }

        let isSpinning = false;
        function toggleSpin() {
            isSpinning = !isSpinning;
            if (isSpinning) {
                viewer3d.spin(true);
            } else {
                viewer3d.spin(false);
            }
        }

        // --- Interface contact bond drawing ---

        function addDashedLine(start, end, colors, radius, dashLen, gapLen) {
            if (!Array.isArray(colors)) colors = [colors];
            const dx = end.x - start.x, dy = end.y - start.y, dz = end.z - start.z;
            const totalLen = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (totalLen < 0.001) return [];
            const ux = dx/totalLen, uy = dy/totalLen, uz = dz/totalLen;
            const shapes = [];
            let pos = 0, dashIdx = 0;
            while (pos < totalLen) {
                const segEnd = Math.min(pos + dashLen, totalLen);
                const s = { x: start.x + ux*pos,   y: start.y + uy*pos,   z: start.z + uz*pos };
                const e = { x: start.x + ux*segEnd, y: start.y + uy*segEnd, z: start.z + uz*segEnd };
                shapes.push(viewer3d.addCylinder({
                    start: s, end: e,
                    radius: radius,
                    color: colors[dashIdx % colors.length],
                    fromCap: 1, toCap: 1
                }));
                pos += dashLen + gapLen;
                dashIdx++;
            }
            return shapes;
        }

        function drawInterfaceContacts3d() {
            interfaceContactShapes.forEach(s => { try { viewer3d.removeShape(s); } catch(e) {} });
            interfaceContactShapes.length = 0;
            if (!showInterfaceContacts || !interfaceContacts || !interfaceContacts.length) {
                viewer3d.render();
                return;
            }
            interfaceContacts.forEach(pair => {
                const c1 = pair.coord1, c2 = pair.coord2;
                if (!c1 || !c2) return;
                const start = { x: c1[0], y: c1[1], z: c1[2] };
                const end   = { x: c2[0], y: c2[1], z: c2[2] };

                let colors, radius;
                if (pair.types && pair.types.length > 0) {
                    // Color by bond type(s); cycle through colors for multi-type pairs
                    colors = pair.types.map(t => interfaceContactColors[t] || '#888888');
                    radius = Math.max(...pair.types.map(t => interfaceContactRadius[t] || 0.10));
                } else {
                    // No type info — visible gray
                    colors = ['#888888'];
                    radius = 0.12;
                }
                const newShapes = addDashedLine(start, end, colors, radius, 0.4, 0.25);
                newShapes.forEach(s => interfaceContactShapes.push(s));
            });
            viewer3d.render();
        }

        function toggleInterfaceContacts(checked) {
            showInterfaceContacts = checked;
            drawInterfaceContacts3d();
        }

    """


def _prepare_interface_contacts_for_3d(analysis: InterfaceAnalysis, cx: Complex, contact_analysis=None) -> str:
    """Prepare unique contact pairs with Cα coordinates for 3D dashed-line drawing.

    When contact_analysis is provided its contacts are used directly as the
    source of lines (so every bond type is guaranteed to appear).  The
    partner_residues from the interface analysis are used as a fallback.
    """
    from collections import defaultdict

    # Build Cα lookup: (chain_id, resseq_int) -> [x, y, z]
    ca_lookup = {}
    for chain_id, chain in cx.chains.items():
        for residue in chain.iter_residues():
            ca_coord = None
            for atom in residue.iter_atoms():
                if atom.name.strip() == 'CA':
                    ca_coord = list(atom.coord)
                    break
            if ca_coord is None:
                for atom in residue.iter_atoms():
                    ca_coord = list(atom.coord)
                    break
            if ca_coord is not None:
                ca_lookup[(chain_id, residue.resseq)] = ca_coord

    def parse_id_str(id_str):
        """Parse 'RESNAME CHAIN:RESSEQ[icode]' -> (chain, resseq_int) or (None, None)."""
        parts = id_str.split()
        if len(parts) >= 2 and ':' in parts[1]:
            chain, pos = parts[1].split(':', 1)
            digits = ''.join(filter(str.isdigit, pos))
            if digits:
                return chain, int(digits)
        return None, None

    contact_pairs = []

    if contact_analysis is not None and hasattr(contact_analysis, 'contacts') and contact_analysis.contacts:
        # Use typed contacts directly: group by residue pair, collect all types
        pair_types: dict = defaultdict(list)
        pair_coords: dict = {}
        for contact in contact_analysis.contacts:
            r1_chain, r1_resseq = parse_id_str(contact.residue1_id)
            r2_chain, r2_resseq = parse_id_str(contact.residue2_id)
            if r1_chain is None or r2_chain is None:
                continue
            coord1 = ca_lookup.get((r1_chain, r1_resseq))
            coord2 = ca_lookup.get((r2_chain, r2_resseq))
            if coord1 is None or coord2 is None:
                continue
            key = tuple(sorted([(r1_chain, r1_resseq), (r2_chain, r2_resseq)]))
            tval = contact.type.value if hasattr(contact.type, 'value') else str(contact.type)
            if tval not in pair_types[key]:
                pair_types[key].append(tval)
            if key not in pair_coords:
                pair_coords[key] = (coord1, coord2)
        for key, types in pair_types.items():
            coord1, coord2 = pair_coords[key]
            contact_pairs.append({"coord1": coord1, "coord2": coord2, "types": types})
    else:
        # Fallback: use partner_residues from interface analysis (untyped)
        pair_set: set = set()
        for chain_id, res_data_list in analysis.chain_data.items():
            for rd in res_data_list:
                if not rd.is_interface:
                    continue
                self_chain, self_resseq = parse_id_str(rd.id_str)
                if self_chain is None:
                    continue
                for partner_id in rd.partner_residues:
                    partner_chain, partner_resseq = parse_id_str(partner_id)
                    if partner_chain is None:
                        continue
                    pair_key = tuple(sorted([(self_chain, self_resseq), (partner_chain, partner_resseq)]))
                    if pair_key in pair_set:
                        continue
                    pair_set.add(pair_key)
                    coord1 = ca_lookup.get((self_chain, self_resseq))
                    coord2 = ca_lookup.get((partner_chain, partner_resseq))
                    if coord1 is None or coord2 is None:
                        continue
                    contact_pairs.append({"coord1": coord1, "coord2": coord2, "types": []})

    return json.dumps(contact_pairs)


def _prepare_interface_residues_for_3d(analysis: InterfaceAnalysis) -> str:
    """Prepare interface residue data for 3D visualization as JSON"""
    interface_data = {}
    
    for chain_id, res_data_list in analysis.chain_data.items():
        interface_residues = [
            {
                "resseq": rd.residue.resseq,
                "resname": rd.residue.resname,
                "icode": rd.residue.icode,
                "contact_count": rd.contact_count
            }
            for rd in res_data_list if rd.is_interface
        ]
        
        if interface_residues:
            interface_data[chain_id] = interface_residues
    
    return json.dumps(interface_data)


def _render_chain_table(
    chain_id: str,
    res_data_list: List[ResidueInterfaceData],
    group_label: str,
    is_group_A: bool,
    is_group_B: bool,
    color_scheme: str,
    colors: Dict
) -> str:
    """Render a single chain's residue grid (same as original)."""
    
    # Calculate statistics
    total = len(res_data_list)
    interface_count = sum(1 for rd in res_data_list if rd.is_interface)
    total_contacts = sum(rd.contact_count for rd in res_data_list)
    
    # Build residue cells
    cells = []
    for rd in res_data_list:
        # Determine color
        if color_scheme == "heatmap":
            if rd.is_interface:
                max_contacts = max((r.contact_count for r in res_data_list if r.is_interface), default=1)
                intensity = rd.contact_count / max_contacts if max_contacts > 0 else 0
                color = _interpolate_color(colors["low"], colors["high"], intensity)
            else:
                color = colors["non_interface"]
        elif color_scheme == "grouped":
            if rd.is_interface:
                if is_group_A and is_group_B:
                    color = colors["both_groups"]
                elif is_group_A:
                    color = colors["group_A_interface"]
                elif is_group_B:
                    color = colors["group_B_interface"]
                else:
                    color = "#95a5a6"
            else:
                color = colors["non_interface"]
        else:  # default
            color = colors["interface"] if rd.is_interface else colors["non_interface"]
        
        # Text color (dark on light backgrounds, light on dark)
        text_color = "#000" if _is_light_color(color) else "#fff"
        
        # Build cell data
        info = {
            "resname": rd.residue.resname,
            "chain": chain_id,
            "resseq": f"{rd.residue.resseq}{rd.residue.icode}".strip(),
            "is_interface": rd.is_interface,
            "contact_count": rd.contact_count,
            "partners": rd.partner_residues[:5],
            "sasa_details": rd.sasa_details,
        }

        cell_html = f"""
        <div class="residue-cell" 
             style="background: {color}; color: {text_color};"
             data-info='{json.dumps(info)}'>
            <div class="resname">{rd.residue.resname}</div>
            <div class="resseq">{info['resseq']}</div>
            {f'<div class="contacts">{rd.contact_count}</div>' if rd.is_interface else ''}
        </div>
        """
        cells.append(cell_html)
    
    return f"""
    <div class="chain-section">
        <div class="chain-header">
            <h2>Chain {chain_id}</h2>
            <div>
                <span class="badge">{group_label}</span>
                <span class="badge">{interface_count}/{total} interface ({100*interface_count/total:.0f}%)</span>
                <span class="badge">{total_contacts} contacts</span>
            </div>
        </div>
        <div class="residue-grid">
            {''.join(cells)}
        </div>
    </div>
    """


def _interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    
    return f"#{r:02x}{g:02x}{b:02x}"


def _is_light_color(hex_color: str) -> bool:
    """Determine if a color is light (for contrast)."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 128

