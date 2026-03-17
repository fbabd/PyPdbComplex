from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict
import json

from ..models import Complex
from ..contacts import ContactAnalysis, ContactType, Contact
from ..interface import InterfaceAnalysis
from ._base import html_page


def generate_contact_html(
    analysis: ContactAnalysis,
    cx: Complex,
    interface_analysis: Optional[InterfaceAnalysis] = None,
    title: str = "Protein Contact Analysis",
    color_scheme: str = "by_type",
    box_size: int = 60,
    show_3d: bool = True,
) -> str:
    """
    Generate interactive HTML visualization of molecular contacts.
    
    Args:
        analysis: ContactAnalysis result
        cx: Complex structure
        interface_analysis: Optional InterfaceAnalysis to highlight interface regions
        title: Page title
        color_scheme: 'by_type' (color by contact type) or 'by_count' (heatmap)
    
    Returns:
        HTML string ready to save or display
    """
    
    # Contact type colors
    contact_colors = {
        ContactType.HYDROGEN_BOND: "#3498db",      # Blue
        ContactType.SALT_BRIDGE: "#e74c3c",        # Red
        ContactType.DISULFIDE: "#f39c12",          # Orange
        ContactType.HYDROPHOBIC: "#2ecc71",        # Green
        ContactType.PI_STACKING: "#9b59b6",        # Purple
    }

    # --- 3D preparation ---
    pdb_content_3d = ""
    contact_3d_json = "null"
    if show_3d:
        if hasattr(cx, 'to_pdb_string'):
            try:
                pdb_content_3d = cx.to_pdb_string()
            except Exception:
                pass
        if not pdb_content_3d:
            try:
                lines = ["HEADER    GENERATED STRUCTURE"]
                atom_num = 1
                for chain_id, chain in cx.chains.items():
                    for residue in chain.iter_residues():
                        for atom in residue.iter_atoms():
                            coord = atom.coord
                            lines.append(
                                f"ATOM  {atom_num:5d}  {atom.name:<4s}{residue.resname:>3s} "
                                f"{chain_id}{residue.resseq:4d}{residue.icode}   "
                                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           "
                                f"{atom.element:>2s}"
                            )
                            atom_num += 1
                lines.append("END")
                pdb_content_3d = "\n".join(lines)
            except Exception:
                pdb_content_3d = "HEADER    NO STRUCTURE DATA\nEND\n"
        contact_3d_json = _prepare_contacts_for_3d(analysis, cx)
    
    # Get summaries
    residue_summary = analysis.get_residue_summary()
    contact_counts = analysis.get_contact_counts()

    # --- Build CSV data ---
    # Residue CSV: one row per residue that has contacts
    _ct_keys = [ct.value for ct in ContactType]
    _residue_csv = []
    for resid, counts in residue_summary.items():
        # resid format: "ASP H:32"  or  "ASP H:32A"
        parts = resid.split()
        _resname = parts[0] if len(parts) > 0 else ""
        _chain, _resseq = "", ""
        if len(parts) > 1:
            cr = parts[1]
            if ":" in cr:
                _chain, _resseq = cr.split(":", 1)
        row = {
            "chain": _chain,
            "resseq": _resseq,
            "resname": _resname,
            "total_contacts": counts.get("total", 0),
        }
        for ct in ContactType:
            row[ct.value] = counts.get(ct.value, 0)
        row["partner_residues"] = "; ".join(sorted(counts.get("partners", [])))
        _residue_csv.append(row)
    # Sort by chain then resseq
    _residue_csv.sort(key=lambda r: (r["chain"], r["resseq"]))
    contact_residue_csv_json = json.dumps(_residue_csv)

    # Pairs CSV: one row per contact
    _pairs_csv = []
    for contact in analysis.contacts:
        _p1 = contact.residue1_id.split()
        _p2 = contact.residue2_id.split()
        def _parse_resid(p):
            rn = p[0] if len(p) > 0 else ""
            ch, rs = "", ""
            if len(p) > 1 and ":" in p[1]:
                ch, rs = p[1].split(":", 1)
            return rn, ch, rs
        rn1, ch1, rs1 = _parse_resid(_p1)
        rn2, ch2, rs2 = _parse_resid(_p2)
        _pairs_csv.append({
            "residue1": contact.residue1_id,
            "chain1": ch1, "resseq1": rs1, "resname1": rn1,
            "residue2": contact.residue2_id,
            "chain2": ch2, "resseq2": rs2, "resname2": rn2,
            "contact_type": contact.type.value,
            "distance_angstrom": round(contact.distance, 3),
        })
    contact_pairs_csv_json = json.dumps(_pairs_csv)

    # Build chain sections
    html_chains = []
    
    for chain_id in sorted(cx.chains.keys()):
        chain = cx.chains[chain_id]
        residues = list(chain.iter_residues())
        
        if not residues:
            continue
        
        html_chains.append(
            _render_contact_chain(
                chain_id,
                residues,
                residue_summary,
                analysis,
                interface_analysis,
                color_scheme,
                contact_colors
            )
        )
    
    # Generate legend
    legend_html = _generate_legend(contact_colors, color_scheme) 
    
    size_control_html = f"""
        <div style="margin: 25px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;">
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <label style="font-weight: bold; color: #2c3e50;">Box Size:</label>
                <input type="range" id="boxSizeSlider" min="40" max="120" value="{box_size}" 
                    style="flex: 1; min-width: 200px;">
                <span id="boxSizeValue" style="font-weight: bold; color: #667eea; min-width: 60px;">{box_size}px</span>
                <button onclick="resetBoxSize()" 
                        style="background: #667eea; color: white; border: none; padding: 8px 16px; 
                            border-radius: 6px; cursor: pointer; font-weight: 600;">
                    Reset
                </button>
            </div>
        </div>
        """
    
    # Summary statistics
    summary_html = _generate_summary_cards(contact_counts, analysis)
    
    # Extra CSS (file-specific, not in common_css())
    extra_css = f"""
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 25px;
        }}

        /* Summary Cards */
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }}
        .summary-card.type-hbond {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        }}
        .summary-card.type-salt {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}
        .summary-card.type-disulfide {{
            background: linear-gradient(135deg, #f39c12 0%, #d68910 100%);
        }}
        .summary-card.type-hydrophobic {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        }}
        .summary-card.type-pi {{
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        }}
        .summary-card h3 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}

        /* Legend */
        .legend {{
            margin: 25px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 12px;
            color: #2c3e50;
        }}
        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .legend-box {{
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 2px solid rgba(0,0,0,0.1);
        }}

        /* Chain Section */
        .chain-section {{
            margin: 35px 0;
        }}
        .chain-header {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .chain-header h2 {{
            margin: 0;
            font-size: 20px;
            font-weight: 600;
        }}
        .chain-badges {{
            display: flex;
            gap: 10px;
        }}
        .badge {{
            background: rgba(255,255,255,0.2);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }}
        .badge.interface {{
            background: rgba(46, 204, 113, 0.3);
        }}

        /* Residue Grid */
        .residue-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax({box_size}px, 1fr));
            gap: 6px;
            padding: 20px;
            background: #fafbfc;
            border: 1px solid #e1e4e8;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}
        .residue-cell {{
            aspect-ratio: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 11px;
            padding: 6px;
            position: relative;
            border: 2px solid transparent;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .residue-cell:hover {{
            transform: scale(1.2) translateY(-4px);
            z-index: 100;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        .residue-cell.has-contacts {{
            border-color: rgba(0,0,0,0.15);
        }}
        .residue-cell.interface {{
            border: 3px solid #2ecc71;
            box-shadow: 0 2px 8px rgba(46, 204, 113, 0.3);
        }}
        .residue-cell .resname {{
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 2px;
        }}
        .residue-cell .resseq {{
            font-size: 10px;
            opacity: 0.7;
        }}
        .residue-cell .contact-badge {{
            position: absolute;
            top: 3px;
            right: 3px;
            background: rgba(0,0,0,0.75);
            color: white;
            font-size: 10px;
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .residue-cell .contact-indicators {{
            position: absolute;
            bottom: 3px;
            left: 3px;
            right: 3px;
            display: flex;
            gap: 2px;
            justify-content: center;
        }}
        .contact-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            border: 1.5px solid rgba(255,255,255,0.9);
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }}

        /* Multi-color gradient background for cells with multiple contact types */
        .residue-cell.multi-contact {{
            background: linear-gradient(135deg, var(--color1) 0%, var(--color2) 50%, var(--color3) 100%) !important;
        }}
        .residue-cell.multi-contact-2 {{
            background: linear-gradient(135deg, var(--color1) 0%, var(--color2) 100%) !important;
        }}
        .residue-cell.multi-contact-3 {{
            background: linear-gradient(120deg, var(--color1) 0%, var(--color2) 40%, var(--color3) 100%) !important;
        }}
        .residue-cell.multi-contact-4 {{
            background: linear-gradient(135deg, var(--color1) 0%, var(--color2) 33%, var(--color3) 66%, var(--color4) 100%) !important;
        }}
        .residue-cell.multi-contact-5 {{
            background: conic-gradient(var(--color1) 0deg 72deg, var(--color2) 72deg 144deg, var(--color3) 144deg 216deg, var(--color4) 216deg 288deg, var(--color5) 288deg 360deg) !important;
        }}

        /* Tooltip */
        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.92);
            color: white;
            padding: 14px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            max-width: 350px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        .tooltip.show {{
            display: block;
        }}
        .tooltip-header {{
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        .tooltip-section {{
            margin: 8px 0;
        }}
        .tooltip-section-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #aaa;
            margin-bottom: 5px;
            letter-spacing: 0.5px;
        }}
        .contact-item {{
            padding: 4px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .contact-type-badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
        }}
        .partner-list {{
            margin-top: 4px;
        }}
        .partner-item {{
            padding: 3px 0;
            font-size: 12px;
        }}

        /* 3D Viewer Section */
        .viewer-section {{
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e1e4e8;
        }}
        .viewer-section h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .viewer-controls {{
            display: flex;
            gap: 20px;
            align-items: flex-start;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .control-group > label {{
            font-size: 11px;
            font-weight: 700;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        #viewer-3d {{
            width: 100%;
            height: 600px;
            background: #1a1a2e;
            border-radius: 8px;
            position: relative;
        }}
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            transition: opacity 0.2s;
        }}
        button:hover {{ opacity: 0.85; }}

        /* Residue info panel */
        #contact-info-panel .cip-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }}

        /* Section toggle */
        .section-toggle {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 14px 20px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0 0 0;
            user-select: none;
        }}
        .section-toggle:hover {{ opacity: 0.9; }}
        .section-content {{
            max-height: 9999px;
            overflow: hidden;
            transition: max-height 0.35s ease-out;
        }}
        .section-content.collapsed {{
            max-height: 0;
        }}

        /* Contact Network View */
        .network-toggle {{
            margin: 20px 0;
            text-align: center;
        }}
        .network-toggle button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .network-toggle button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}

        /* Data Tables section */
        .dt-tabs {{
            display: flex;
            gap: 8px;
            margin: 18px 0 0 0;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 0;
        }}
        .dt-tab {{
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 8px 22px;
            font-size: 14px;
            font-weight: 600;
            color: #6c757d;
            cursor: pointer;
            margin-bottom: -2px;
            border-radius: 6px 6px 0 0;
            transition: color 0.15s, border-color 0.15s;
        }}
        .dt-tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: #f4f6ff;
        }}
        .dt-tab:hover:not(.active) {{ color: #495057; background: #f8f9fa; }}
        .dt-panel {{ display: none; }}
        .dt-panel.active {{ display: block; }}
        .dt-toolbar {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 0 10px 0;
            flex-wrap: wrap;
        }}
        .dt-toolbar input {{
            padding: 6px 12px;
            border: 1px solid #ced4da;
            border-radius: 6px;
            font-size: 13px;
            width: 220px;
            outline: none;
        }}
        .dt-toolbar input:focus {{ border-color: #667eea; box-shadow: 0 0 0 2px rgba(102,126,234,0.15); }}
        .dt-dl-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 6px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.15s;
        }}
        .dt-dl-btn:hover {{ opacity: 0.88; }}
        .dt-scroll {{
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            max-height: 480px;
            overflow-y: auto;
        }}
        .dt-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .dt-table thead th {{
            background: #f4f6ff;
            color: #2c3e50;
            font-weight: 700;
            padding: 10px 14px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 1;
            border-bottom: 2px solid #dee2e6;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }}
        .dt-table thead th:hover {{ background: #e8ecff; }}
        .dt-table thead th .sort-icon {{ color: #aaa; margin-left: 4px; font-size: 11px; }}
        .dt-table tbody tr:nth-child(even) {{ background: #f9f9fb; }}
        .dt-table tbody tr:hover {{ background: #eef0ff; }}
        .dt-table tbody td {{
            padding: 8px 14px;
            border-bottom: 1px solid #f0f0f0;
            white-space: nowrap;
        }}
        .dt-table .partners-cell {{ white-space: normal; max-width: 260px; font-size: 12px; color: #555; }}
        .dt-badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            color: white;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .dt-no-rows {{ padding: 24px; text-align: center; color: #888; font-style: italic; }}
    """

    pdb_escaped = pdb_content_3d.replace('`', '\\`') if show_3d else ""


    body = f"""
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Molecular contact analysis with {len(analysis.contacts)} total interactions</div>

        {summary_html}

        {'_3D_SECTION_PLACEHOLDER_' if show_3d else ''}

        <div class="section-toggle" onclick="toggleSection('2d-section')">
            <h2 style="margin:0;">2D Residue Grid View</h2>
            <span id="2d-toggle-icon">▼</span>
        </div>
        <div id="2d-section" class="section-content">
            {legend_html}
            {size_control_html}
            {''.join(html_chains)}
        </div>

        <div class="section-toggle" onclick="toggleSection('data-section')">
            <h2 style="margin:0;">📊 Data Tables</h2>
            <span id="data-toggle-icon">▼</span>
        </div>
        <div id="data-section" class="section-content">
            <div class="dt-tabs">
                <button class="dt-tab active" id="dt-tab-pairs"
                        onclick="showDtTab('pairs')">Contact Pairs</button>
                <button class="dt-tab" id="dt-tab-residues"
                        onclick="showDtTab('residues')">Per-Residue Summary</button>
            </div>

            <!-- Pairs panel -->
            <div id="dt-panel-pairs" class="dt-panel active">
                <div class="dt-toolbar">
                    <input type="text" id="dt-pairs-search" placeholder="🔍 Filter pairs…"
                           oninput="filterDt('pairs')">
                    <button class="dt-dl-btn" onclick="downloadDt('pairs')">⬇ Download CSV</button>
                    <span id="dt-pairs-count" style="color:#888; font-size:12px;"></span>
                </div>
                <div class="dt-scroll">
                    <table class="dt-table" id="dt-pairs-table">
                        <thead>
                            <tr>
                                <th onclick="sortDt('pairs',0)">Residue 1 <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('pairs',1)">Residue 2 <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('pairs',2)">Bond Type <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('pairs',3)">Distance (Å) <span class="sort-icon">⇅</span></th>
                            </tr>
                        </thead>
                        <tbody id="dt-pairs-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- Residues panel -->
            <div id="dt-panel-residues" class="dt-panel">
                <div class="dt-toolbar">
                    <input type="text" id="dt-residues-search" placeholder="🔍 Filter residues…"
                           oninput="filterDt('residues')">
                    <button class="dt-dl-btn" onclick="downloadDt('residues')">⬇ Download CSV</button>
                    <span id="dt-residues-count" style="color:#888; font-size:12px;"></span>
                </div>
                <div class="dt-scroll">
                    <table class="dt-table" id="dt-residues-table">
                        <thead>
                            <tr>
                                <th onclick="sortDt('residues',0)">Chain <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',1)">ResSeq <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',2)">Residue <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',3)">Total <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',4)">H-Bond <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',5)">Salt Bridge <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',6)">Disulfide <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',7)">Hydrophobic <span class="sort-icon">⇅</span></th>
                                <th onclick="sortDt('residues',8)">π-Stacking <span class="sort-icon">⇅</span></th>
                                <th>Partner Residues</th>
                            </tr>
                        </thead>
                        <tbody id="dt-residues-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="tooltip" id="tooltip"></div>
    </div>

    <script>
        const tooltip = document.getElementById('tooltip');
        const contactColors = {json.dumps({k.value: v for k, v in contact_colors.items()})};
        // Tooltip functionality
        document.querySelectorAll('.residue-cell').forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const info = JSON.parse(cell.dataset.info);

                let html = `<div class="tooltip-header">${{info.resname}} ${{info.chain}}:${{info.resseq}}</div>`;

                if (info.is_interface) {{
                    html += `<div class="tooltip-section">
                        <div class="tooltip-section-title">⭐ Interface Residue</div>
                    </div>`;
                }}

                if (info.contacts && info.contacts.length > 0) {{
                    html += `<div class="tooltip-section">
                        <div class="tooltip-section-title">Contacts (${{info.contacts.length}})</div>`;

                    const contactsByType = {{}};
                    info.contacts.forEach(c => {{
                        if (!contactsByType[c.type]) contactsByType[c.type] = [];
                        contactsByType[c.type].push(c);
                    }});

                    for (const [type, contacts] of Object.entries(contactsByType)) {{
                        const color = contactColors[type] || '#999';
                        html += `<div class="contact-item">
                            <span class="contact-type-badge" style="background: ${{color}}">
                                ${{type.replace('_', ' ').toUpperCase()}}
                            </span>
                            <span>${{contacts.length}}×</span>
                        </div>`;
                    }}

                    html += `</div>`;

                    // Show partners
                    if (info.partners && info.partners.length > 0) {{
                        html += `<div class="tooltip-section">
                            <div class="tooltip-section-title">Partner Residues</div>
                            <div class="partner-list">`;

                        info.partners.slice(0, 5).forEach(p => {{
                            html += `<div class="partner-item">→ ${{p}}</div>`;
                        }});

                        if (info.partners.length > 5) {{
                            html += `<div class="partner-item">... and ${{info.partners.length - 5}} more</div>`;
                        }}

                        html += `</div></div>`;
                    }}
                }} else {{
                    html += `<div class="tooltip-section">No contacts detected</div>`;
                }}

                tooltip.innerHTML = html;
                tooltip.classList.add('show');
            }});

            cell.addEventListener('mousemove', (e) => {{
                const offset = 15;
                let x = e.clientX + offset;
                let y = e.clientY + offset;

                const rect = tooltip.getBoundingClientRect();
                if (x + rect.width > window.innerWidth) x = e.clientX - rect.width - offset;
                if (y + rect.height > window.innerHeight) y = e.clientY - rect.height - offset;

                tooltip.style.left = x + 'px';
                tooltip.style.top = y + 'px';
            }});

            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});

        // Box size control
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
            boxSizeSlider.addEventListener('input', (e) => updateBoxSize(e.target.value));
        }}

        function resetBoxSize() {{
            const defaultSize = {box_size};
            if (boxSizeSlider) {{ boxSizeSlider.value = defaultSize; updateBoxSize(defaultSize); }}
        }}

        // Section toggle
        function toggleSection(id) {{
            const section = document.getElementById(id);
            const icon = document.getElementById(id.replace('-section', '') + '-toggle-icon');
            if (!section) return;
            if (section.classList.contains('collapsed')) {{
                section.classList.remove('collapsed');
                if (icon) icon.textContent = '▼';
            }} else {{
                section.classList.add('collapsed');
                if (icon) icon.textContent = '▶';
            }}
        }}

        {'_3D_JS_PLACEHOLDER_' if show_3d else ''}

        // ── Data Tables ──────────────────────────────────────────────
        const _dtResidueCsv = {contact_residue_csv_json};
        const _dtPairsCsv   = {contact_pairs_csv_json};

        // Colour map for bond-type badges
        const _dtTypeColors = {json.dumps({k.value: v for k, v in contact_colors.items()})};
        const _dtTypeLabels = {{
            'hydrogen_bond': 'H-Bond',
            'salt_bridge':   'Salt Bridge',
            'disulfide':     'Disulfide',
            'hydrophobic':   'Hydrophobic',
            'pi_stacking':   'π-Stacking',
        }};

        // State for each panel: filtered rows + sort state
        const _dtState = {{
            pairs:    {{ rows: _dtPairsCsv.slice(),    sortCol: -1, sortAsc: true }},
            residues: {{ rows: _dtResidueCsv.slice(),  sortCol: -1, sortAsc: true }},
        }};

        function _badge(type) {{
            const c = _dtTypeColors[type] || '#888';
            const l = _dtTypeLabels[type] || type;
            return `<span class="dt-badge" style="background:${{c}}">${{l}}</span>`;
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
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href = url; a.download = filename; a.click();
            URL.revokeObjectURL(url);
        }}

        // Render pairs tbody
        function _renderPairs(rows) {{
            const tb = document.getElementById('dt-pairs-body');
            if (!rows.length) {{
                tb.innerHTML = '<tr><td colspan="4" class="dt-no-rows">No contacts match the filter.</td></tr>';
            }} else {{
                tb.innerHTML = rows.map(r => `<tr>
                    <td>${{r.residue1}}</td>
                    <td>${{r.residue2}}</td>
                    <td>${{_badge(r.contact_type)}}</td>
                    <td>${{r.distance_angstrom.toFixed ? r.distance_angstrom.toFixed(3) : r.distance_angstrom}}</td>
                </tr>`).join('');
            }}
            const el = document.getElementById('dt-pairs-count');
            if (el) el.textContent = rows.length + ' row' + (rows.length !== 1 ? 's' : '');
        }}

        // Render residues tbody
        function _renderResidues(rows) {{
            const tb = document.getElementById('dt-residues-body');
            if (!rows.length) {{
                tb.innerHTML = '<tr><td colspan="10" class="dt-no-rows">No residues match the filter.</td></tr>';
            }} else {{
                tb.innerHTML = rows.map(r => `<tr>
                    <td>${{r.chain}}</td>
                    <td>${{r.resseq}}</td>
                    <td>${{r.resname}}</td>
                    <td><strong>${{r.total_contacts}}</strong></td>
                    <td>${{r.hydrogen_bond || 0}}</td>
                    <td>${{r.salt_bridge   || 0}}</td>
                    <td>${{r.disulfide     || 0}}</td>
                    <td>${{r.hydrophobic   || 0}}</td>
                    <td>${{r.pi_stacking   || 0}}</td>
                    <td class="partners-cell">${{r.partner_residues || ''}}</td>
                </tr>`).join('');
            }}
            const el = document.getElementById('dt-residues-count');
            if (el) el.textContent = rows.length + ' row' + (rows.length !== 1 ? 's' : '');
        }}

        function _render(panel) {{
            if (panel === 'pairs') _renderPairs(_dtState.pairs.rows);
            else _renderResidues(_dtState.residues.rows);
        }}

        // Filter
        function filterDt(panel) {{
            const q = (document.getElementById('dt-' + panel + '-search').value || '').toLowerCase();
            const src = panel === 'pairs' ? _dtPairsCsv : _dtResidueCsv;
            _dtState[panel].rows = q
                ? src.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)))
                : src.slice();
            _dtState[panel].sortCol = -1;
            _render(panel);
        }}

        // Sort
        function sortDt(panel, col) {{
            const st = _dtState[panel];
            st.sortAsc = (st.sortCol === col) ? !st.sortAsc : true;
            st.sortCol = col;
            const pairKeys = ['residue1','residue2','contact_type','distance_angstrom'];
            const resKeys  = ['chain','resseq','resname','total_contacts',
                              'hydrogen_bond','salt_bridge','disulfide','hydrophobic','pi_stacking'];
            const key = panel === 'pairs' ? pairKeys[col] : resKeys[col];
            if (!key) return;
            st.rows.sort((a, b) => {{
                const av = a[key], bv = b[key];
                const n = (typeof av === 'number' && typeof bv === 'number') ? av - bv
                        : String(av).localeCompare(String(bv), undefined, {{numeric: true}});
                return st.sortAsc ? n : -n;
            }});
            _render(panel);
        }}

        // Tab switch
        function showDtTab(panel) {{
            ['pairs','residues'].forEach(p => {{
                document.getElementById('dt-panel-' + p).classList.toggle('active', p === panel);
                document.getElementById('dt-tab-'   + p).classList.toggle('active', p === panel);
            }});
        }}

        // Download
        function downloadDt(panel) {{
            if (panel === 'pairs')
                _dlBlob(_toCsv(_dtPairsCsv),   'contacts_pairs.csv');
            else
                _dlBlob(_toCsv(_dtResidueCsv), 'contacts_per_residue.csv');
        }}

        // Initial render
        _renderPairs(_dtPairsCsv);
        _renderResidues(_dtResidueCsv);
    </script>
"""

    # Substitute 3D placeholders (avoids f-string escaping issues with complex JS)
    if show_3d:
        body = body.replace(
            "_3D_SECTION_PLACEHOLDER_",
            _render_contact_3d_section()
        )
        body = body.replace(
            "_3D_JS_PLACEHOLDER_",
            _render_contact_3d_javascript(pdb_escaped, contact_3d_json, contact_colors)
        )

    cdn_scripts = (
        ["https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"]
        if show_3d else []
    )
    html = html_page(title, body, extra_css=extra_css, cdn_scripts=cdn_scripts, max_width=1600)
    return html


def _generate_summary_cards(contact_counts: Dict, analysis: ContactAnalysis) -> str:
    """Generate summary statistic cards."""
    cards = []
    
    type_map = {
        ContactType.HYDROGEN_BOND: ("H-Bonds", "type-hbond"),
        ContactType.SALT_BRIDGE: ("Salt Bridges", "type-salt"),
        ContactType.DISULFIDE: ("Disulfides", "type-disulfide"),
        ContactType.HYDROPHOBIC: ("Hydrophobic", "type-hydrophobic"),
        ContactType.PI_STACKING: ("π-π Stacking", "type-pi"),
    }
    
    for ctype, count in contact_counts.items():
        label, css_class = type_map.get(ctype, (ctype.value, ""))
        cards.append(f"""
        <div class="summary-card {css_class}">
            <h3>{label}</h3>
            <div class="value">{count}</div>
        </div>
        """)
    
    total = sum(contact_counts.values())
    cards.insert(0, f"""
    <div class="summary-card">
        <h3>Total Contacts</h3>
        <div class="value">{total}</div>
    </div>
    """)
    
    return f'<div class="summary">{"".join(cards)}</div>'


def _generate_legend(contact_colors: Dict, color_scheme: str) -> str:
    """Generate legend HTML."""
    items = []
    
    type_labels = {
        ContactType.HYDROGEN_BOND: "Hydrogen Bond",
        ContactType.SALT_BRIDGE: "Salt Bridge",
        ContactType.DISULFIDE: "Disulfide Bond",
        ContactType.HYDROPHOBIC: "Hydrophobic Contact",
        ContactType.PI_STACKING: "π-π Stacking",
    }
    
    for ctype, color in contact_colors.items():
        label = type_labels.get(ctype, ctype.value)
        items.append(f"""
        <div class="legend-item">
            <div class="legend-box" style="background: {color}"></div>
            <span>{label}</span>
        </div>
        """)
    
    if color_scheme == "by_type":
        # Show example of multi-contact gradient
        items.append(f"""
        <div class="legend-item">
            <div class="legend-box" style="background: linear-gradient(135deg, {contact_colors[ContactType.HYDROGEN_BOND]} 0%, {contact_colors[ContactType.SALT_BRIDGE]} 50%, {contact_colors[ContactType.HYDROPHOBIC]} 100%)"></div>
            <span>Multiple Contact Types (gradient)</span>
        </div>
        """)
    
    items.append("""
    <div class="legend-item">
        <div class="legend-box" style="background: #f5f5f5; border: 2px solid #ddd"></div>
        <span>No Contacts</span>
    </div>
    """)
    
  
    
    items.append("""
    <div class="legend-item" style="margin-top: 10px; width: 100%; padding-top: 10px; border-top: 1px solid #ddd;">
        <span style="font-size: 12px; color: #666;">
            💡 Residues with multiple contact types show gradient backgrounds. 
            Colored dots at bottom indicate all contact types present.
        </span>
    </div>
    """)
    
    return f"""
    <div class="legend">
        <div class="legend-title">Legend - Color Scheme: {color_scheme.replace('_', ' ').title()}</div>
        <div class="legend-items">
            {"".join(items)}
        </div>
    </div>
    """


def _render_contact_chain(
    chain_id: str,
    residues: List,
    residue_summary: Dict,
    analysis: ContactAnalysis,
    interface_analysis: Optional[InterfaceAnalysis],
    color_scheme: str,
    contact_colors: Dict
) -> str:
    """Render a single chain with contact visualization."""
    
    # Get interface residue IDs if available
    interface_resids = set()
    if interface_analysis and chain_id in interface_analysis.chain_data:
        interface_resids = {
            rd.residue.id_str 
            for rd in interface_analysis.chain_data[chain_id] 
            if rd.is_interface
        }
    
    # Debug: Check what residue IDs look like
    # print(f"Chain {chain_id} residue summary keys: {list(residue_summary.keys())[:5]}")
    # if residues:
    #     print(f"First residue id_str: {residues[0].id_str}")
    
    # Count contacts in this chain
    total_contacts = 0
    interface_contact_count = 0
    
    cells = []
    for residue in residues:
        resid = residue.id_str
        is_interface = resid in interface_resids
        
        # Get contact information
        res_contacts = residue_summary.get(resid, {})
        contact_total = res_contacts.get('total', 0)
        total_contacts += contact_total
        
        if is_interface:
            interface_contact_count += contact_total
        
        # Determine cell color and styling
        contact_types_with_colors = []
        for ctype in ContactType:
            if res_contacts.get(ctype.value, 0) > 0:
                contact_types_with_colors.append((ctype, contact_colors.get(ctype, "#999")))
        
        num_contact_types = len(contact_types_with_colors)
        
        if color_scheme == "by_type" and num_contact_types > 0:
            if num_contact_types == 1:
                # Single contact type - solid color
                color = contact_types_with_colors[0][1]
                multi_class = ""
                style_vars = ""
            else:
                # Multiple contact types - gradient background
                color = "#ffffff"  # Base color (won't be used due to gradient)
                multi_class = f"multi-contact-{min(num_contact_types, 5)}"
                
                # Create CSS variables for gradient colors
                style_vars = " ".join([
                    f"--color{i+1}: {color};" 
                    for i, (_, color) in enumerate(contact_types_with_colors[:5])
                ])
        elif color_scheme == "by_count" and contact_total > 0:
            # Heatmap based on total contacts
            max_contacts = max((s.get('total', 0) for s in residue_summary.values()), default=1)
            color = _get_heatmap_color(contact_total, max_contacts)
            multi_class = ""
            style_vars = ""
        else:
            # No contacts - use light gray
            color = "#f5f5f5"
            multi_class = ""
            style_vars = ""
        
        # Text color for contrast
        if num_contact_types > 1:
            # For gradients, use dark text with white shadow for readability
            text_color = "#000"
            text_shadow = "text-shadow: 0 0 3px white, 0 0 3px white, 0 0 3px white;"
        else:
            text_color = "#000" if _is_light_color(color) else "#fff"
            text_shadow = ""
        
        # Get contact details for tooltip
        contact_details = []
        partners = set()
        for contact in analysis.contacts:
            if contact.residue1_id == resid:
                contact_details.append({
                    "type": contact.type.value,
                    "partner": contact.residue2_id,
                    "distance": contact.distance
                })
                partners.add(contact.residue2_id)
            elif contact.residue2_id == resid:
                contact_details.append({
                    "type": contact.type.value,
                    "partner": contact.residue1_id,
                    "distance": contact.distance
                })
                partners.add(contact.residue1_id)
        
        # Build contact type indicators (colored dots)
        contact_dots = []
        for ctype in ContactType:
            if res_contacts.get(ctype.value, 0) > 0:
                color_dot = contact_colors.get(ctype, "#999")
                contact_dots.append(f'<div class="contact-dot" style="background: {color_dot}"></div>')
        
        # Cell info for tooltip
        info = {
            "resname": residue.resname,
            "chain": chain_id,
            "resseq": f"{residue.resseq}{residue.icode}".strip(),
            "is_interface": is_interface,
            "total_contacts": contact_total,
            "contacts": contact_details,
            "partners": sorted(list(partners))
        }
        
        interface_class = "interface" if is_interface else ""
        has_contacts_class = "has-contacts" if contact_total > 0 else ""
        
        cells.append(f"""
        <div class="residue-cell {interface_class} {has_contacts_class} {multi_class}"
             style="background: {color}; color: {text_color}; {text_shadow} {style_vars}"
             data-info='{json.dumps(info)}'>
            <div class="resname">{residue.resname}</div>
            <div class="resseq">{info['resseq']}</div>
            {f'<div class="contact-badge">{contact_total}</div>' if contact_total > 0 else ''}
            {f'<div class="contact-indicators">{"".join(contact_dots[:5])}</div>' if contact_dots else ''}
        </div>
        """)
    
    # Chain statistics
    interface_badge = ""
    if interface_resids:
        interface_badge = f'<span class="badge interface">{len(interface_resids)} interface residues</span>'
    
    return f"""
    <div class="chain-section">
        <div class="chain-header">
            <h2>Chain {chain_id}</h2>
            <div class="chain-badges">
                {interface_badge}
                <span class="badge">{len(residues)} residues</span>
                <span class="badge">{total_contacts} contacts</span>
            </div>
        </div>
        <div class="residue-grid">
            {"".join(cells)}
        </div>
    </div>
    """


def _get_dominant_contact_color(res_contacts: Dict, contact_colors: Dict, mixed_threshold: int = 2) -> str:
    """
    Get color for the dominant contact type.
    If multiple contact types are present, blend colors or show as mixed.
    """
    if not res_contacts:
        return "#e8e8e8"  # Gray for no contacts
    
    # Count how many contact types are present
    contact_types_present = []
    for ctype in ContactType:
        count = res_contacts.get(ctype.value, 0)
        if count > 0:
            contact_types_present.append((ctype, count))
    
    if not contact_types_present:
        return "#e8e8e8"
    
    # If only one type, return its color
    if len(contact_types_present) == 1:
        return contact_colors.get(contact_types_present[0][0], "#999")
    
    # If multiple types and above threshold, show as "mixed" (gradient)
    if len(contact_types_present) >= mixed_threshold:
        # Return a purple-ish color to indicate mixed interactions
        return "#8e44ad"
    
    # Otherwise return the dominant (highest count) type
    dominant_type = max(contact_types_present, key=lambda x: x[1])[0]
    return contact_colors.get(dominant_type, "#999")


def _get_heatmap_color(count: int, max_count: int) -> str:
    """Get heatmap color based on contact count."""
    if count == 0:
        return "#f0f0f0"
    
    # Gradient from light yellow to dark red
    intensity = min(count / max_count, 1.0) if max_count > 0 else 0
    
    # Color interpolation: light -> dark
    r = int(255)
    g = int(255 * (1 - intensity * 0.8))
    b = int(255 * (1 - intensity))
    
    return f"#{r:02x}{g:02x}{b:02x}"


def _is_light_color(hex_color: str) -> bool:
    """Determine if a color is light (for contrast)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 128


# ==================== 3D Viewer Helpers ====================

def _prepare_contacts_for_3d(analysis: ContactAnalysis, cx: Complex) -> str:
    """Build JSON data for the 3D contact viewer."""
    # Cα lookup: residue.id_str -> {chain, resseq, resname, coord}
    ca_lookup: Dict = {}
    for chain_id, chain in cx.chains.items():
        for residue in chain.iter_residues():
            ca_coord = None
            for atom in residue.iter_atoms():
                if atom.name.strip() == "CA":
                    ca_coord = [round(float(c), 3) for c in atom.coord]
                    break
            if ca_coord is None:
                for atom in residue.iter_atoms():
                    if not atom.name.strip().upper().startswith("H"):
                        ca_coord = [round(float(c), 3) for c in atom.coord]
                        break
            if ca_coord:
                ca_lookup[residue.id_str] = {
                    "chain": chain_id,
                    "resseq": residue.resseq,
                    "resname": residue.resname,
                    "coord": ca_coord,
                }

    # contacts_by_type: for checkbox counts (type -> count of unique pairs)
    contacts_by_type: Dict[str, List] = defaultdict(list)
    # contact_pairs: one entry per unique residue pair, with ALL types present
    pair_map: Dict[tuple, Dict] = {}

    for contact in analysis.contacts:
        type_key = contact.type.value
        pair_key = tuple(sorted([contact.residue1_id, contact.residue2_id]))
        r1 = ca_lookup.get(contact.residue1_id)
        r2 = ca_lookup.get(contact.residue2_id)
        if not (r1 and r2):
            continue

        # contacts_by_type: track unique pairs per type
        if not any(
            e["res1_id"] == contact.residue1_id and e["res2_id"] == contact.residue2_id
            for e in contacts_by_type[type_key]
        ):
            contacts_by_type[type_key].append({
                "res1_id": contact.residue1_id,
                "res2_id": contact.residue2_id,
            })

        # pair_map: accumulate types per pair
        if pair_key not in pair_map:
            pair_map[pair_key] = {
                "res1_id": contact.residue1_id,
                "res2_id": contact.residue2_id,
                "chain1": r1["chain"],
                "resseq1": r1["resseq"],
                "chain2": r2["chain"],
                "resseq2": r2["resseq"],
                "coord1": r1["coord"],
                "coord2": r2["coord"],
                "distance": round(contact.distance, 2),
                "types": [],
            }
        if type_key not in pair_map[pair_key]["types"]:
            pair_map[pair_key]["types"].append(type_key)

    contact_pairs = list(pair_map.values())

    # Per-residue contact summary for info panel
    res_summary = analysis.get_residue_summary()
    contact_residues_by_chain: Dict[str, List] = defaultdict(list)
    seen_res: Set[str] = set()
    for contact in analysis.contacts:
        for res_id in (contact.residue1_id, contact.residue2_id):
            if res_id in seen_res or res_id not in ca_lookup:
                continue
            seen_res.add(res_id)
            info = ca_lookup[res_id]
            summary = res_summary.get(res_id, {})
            contact_residues_by_chain[info["chain"]].append({
                "resseq": info["resseq"],
                "resname": info["resname"],
                "contact_count": summary.get("total", 0),
                "contact_types": [
                    t.value for t in ContactType if summary.get(t.value, 0) > 0
                ],
            })

    return json.dumps({
        "contacts_by_type": {k: v for k, v in contacts_by_type.items()},
        "contact_pairs": contact_pairs,
        "contact_residues_by_chain": {k: v for k, v in contact_residues_by_chain.items()},
    })


def _render_contact_3d_section() -> str:
    """HTML for the 3D viewer section with bond-type checkboxes."""
    return """
        <div class="viewer-section">
            <h2>3D Structure Viewer</h2>

            <div class="viewer-controls">
                <div class="control-group">
                    <label>View Style</label>
                    <div style="display:flex; gap:8px;">
                        <button onclick="setContactStyle('cartoon')">Cartoon</button>
                        <button onclick="setContactStyle('stick')">Stick</button>
                        <button onclick="setContactStyle('sphere')">Sphere</button>
                    </div>
                </div>

                <div class="control-group" style="flex:1; min-width:300px;">
                    <label>Bond Types (click to toggle)</label>
                    <div id="bond-type-checkboxes"
                         style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
                    </div>
                </div>

                <div class="control-group">
                    <label>Actions</label>
                    <div style="display:flex; gap:8px;">
                        <button onclick="resetContactView()">Reset View</button>
                        <button onclick="toggleContactSpin()">Toggle Spin</button>
                    </div>
                </div>
            </div>

            <div style="position:relative;">
                <div id="viewer-3d"></div>
                <div id="contact-info-panel"
                     style="display:none; position:absolute; bottom:20px; right:20px;
                            width:290px; background:white; border-radius:10px;
                            box-shadow:0 4px 20px rgba(0,0,0,0.3); padding:16px;
                            z-index:1000; font-size:13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:10px; padding-bottom:8px;
                                border-bottom:2px solid #667eea;">
                        <span id="cip-title"
                              style="font-size:15px; font-weight:700; color:#2c3e50;">
                            Residue
                        </span>
                        <button onclick="closeContactPanel()"
                                style="background:#e74c3c; color:white; border:none;
                                       width:22px; height:22px; border-radius:50%;
                                       cursor:pointer; font-size:14px; padding:0;
                                       display:flex; align-items:center; justify-content:center;">
                            &#x2715;
                        </button>
                    </div>
                    <div id="cip-body"></div>
                </div>
            </div>
            <p style="font-size:12px; color:#888; margin-top:8px;">
                <i>Click a residue to see contact details.
                   Toggle bond types with the checkboxes above.</i>
            </p>
        </div>
    """


def _render_contact_3d_javascript(
    pdb_escaped: str,
    contact_3d_json: str,
    contact_colors: Dict,
) -> str:
    """JavaScript for the 3D contact viewer."""
    colors_js = json.dumps({k.value: v for k, v in contact_colors.items()})
    bond_radius = json.dumps({
        "hydrogen_bond": 0.12,
        "salt_bridge": 0.18,
        "disulfide": 0.25,
        "hydrophobic": 0.08,
        "pi_stacking": 0.15,
    })
    labels_js = json.dumps({
        "hydrogen_bond": "H-Bond",
        "salt_bridge": "Salt Bridge",
        "disulfide": "Disulfide",
        "hydrophobic": "Hydrophobic",
        "pi_stacking": "\u03c0-Stacking",
    })

    return f"""
        // ---- 3D Contact Viewer ----
        const pdbData3d = `{pdb_escaped}`;
        const contactData3d = {contact_3d_json};
        const contactTypeColors3d = {colors_js};
        const contactTypeBondRadius = {bond_radius};
        const contactTypeLabels3d = {labels_js};

        let viewer3dContact = null;
        let currentContactStyle3d = 'cartoon';
        let isSpinning3dContact = false;
        const bondTypeVisible = {{}};
        const bondShapes3d = [];   // flat array of all cylinder shape refs

        document.addEventListener('DOMContentLoaded', function () {{
            if (contactData3d) {{
                buildBondTypeCheckboxes();
                initContactViewer3d();
            }}
        }});

        function buildBondTypeCheckboxes() {{
            const container = document.getElementById('bond-type-checkboxes');
            if (!container || !contactData3d) return;
            const types = Object.keys(contactData3d.contacts_by_type);
            types.forEach(type => {{
                bondTypeVisible[type] = true;
                const color = contactTypeColors3d[type] || '#888';
                const label = contactTypeLabels3d[type] || type;
                const count = contactData3d.contacts_by_type[type].length;

                const lbl = document.createElement('label');
                lbl.style.cssText = 'display:flex; align-items:center; gap:5px; cursor:pointer; user-select:none; padding:5px 10px; background:white; border-radius:6px; border:2px solid ' + color + '; font-size:13px; font-weight:600; transition:opacity 0.2s;';
                lbl.innerHTML =
                    '<input type="checkbox" checked style="accent-color:' + color + '; width:14px; height:14px;" onchange="toggleBondType(\\'' + type + '\\', this.checked)">' +
                    '<span style="display:inline-block; width:12px; height:12px; background:' + color + '; border-radius:2px;"></span>' +
                    '<span>' + label + ' (' + count + ')</span>';
                container.appendChild(lbl);
            }});
        }}

        function initContactViewer3d() {{
            const el = document.getElementById('viewer-3d');
            if (!el) return;
            if (typeof $3Dmol === 'undefined') {{
                el.innerHTML = '<div style="color:#999; padding:30px; text-align:center; font-size:14px;">3Dmol.js failed to load — check internet connection.</div>';
                return;
            }}
            if (!pdbData3d || pdbData3d.trim().startsWith('HEADER    NO STRUCTURE')) {{
                el.innerHTML = '<div style="color:#999; padding:30px; text-align:center;">No PDB structure available.</div>';
                return;
            }}
            viewer3dContact = $3Dmol.createViewer(el, {{backgroundColor: 'white'}});
            viewer3dContact.addModel(pdbData3d, 'pdb');
            applyContactStyle3d('cartoon');
            drawContactBonds3d();
            registerContactClickable3d();
            viewer3dContact.zoomTo();
            viewer3dContact.render();
        }}

        function applyContactStyle3d(style) {{
            if (!viewer3dContact) return;
            currentContactStyle3d = style;
            viewer3dContact.setStyle({{}}, {{}});

            const grayOpacity = {{ cartoon: 0.55, stick: 0.45, sphere: 0.35 }}[style] || 0.55;
            if (style === 'cartoon')
                viewer3dContact.setStyle({{}}, {{cartoon: {{color: 'lightgray', opacity: grayOpacity}}}});
            else if (style === 'stick')
                viewer3dContact.setStyle({{}}, {{stick: {{color: 'lightgray', opacity: grayOpacity, radius: 0.12}}}});
            else if (style === 'sphere')
                viewer3dContact.setStyle({{}}, {{sphere: {{color: 'lightgray', opacity: grayOpacity, scale: 0.25}}}});

            const highlight = {{
                cartoon: c => ({{cartoon: {{color: c, opacity: 1.0}}}}),
                stick:   c => ({{stick: {{color: c, opacity: 1.0, radius: 0.22}}}}),
                sphere:  c => ({{sphere: {{color: c, opacity: 1.0, scale: 0.38}}}})
            }}[style] || (c => ({{cartoon: {{color: c, opacity: 1.0}}}}));

            Object.entries(contactData3d.contact_residues_by_chain).forEach(([chain, residues]) => {{
                residues.forEach(res => {{
                    const color = (res.contact_types && res.contact_types.length > 0)
                        ? (contactTypeColors3d[res.contact_types[0]] || '#aaa')
                        : '#aaa';
                    viewer3dContact.setStyle({{chain, resi: res.resseq}}, highlight(color));
                }});
            }});

            viewer3dContact.render();
        }}

        /**
         * Draw a dashed line between two 3D points using alternating cylinder segments.
         * colors: array of hex color strings — each dash cycles through them in order.
         * Returns array of shape references.
         */
        function addDashedLine(start, end, colors, radius, dashLen, gapLen) {{
            const dx = end.x - start.x, dy = end.y - start.y, dz = end.z - start.z;
            const totalLen = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (totalLen < 0.001 || !colors.length) return [];
            const ux = dx/totalLen, uy = dy/totalLen, uz = dz/totalLen;
            const shapes = [];
            let pos = 0, dashIdx = 0;
            while (pos < totalLen) {{
                const segEnd = Math.min(pos + dashLen, totalLen);
                const color = colors[dashIdx % colors.length];
                const s = viewer3dContact.addCylinder({{
                    start: {{x: start.x + ux*pos,    y: start.y + uy*pos,    z: start.z + uz*pos}},
                    end:   {{x: start.x + ux*segEnd,  y: start.y + uy*segEnd,  z: start.z + uz*segEnd}},
                    radius: radius,
                    color: color,
                    opacity: 0.9,
                    fromCap: true,
                    toCap: true,
                }});
                if (s) shapes.push(s);
                pos += dashLen + gapLen;
                dashIdx++;
            }}
            return shapes;
        }}

        function drawContactBonds3d() {{
            if (!viewer3dContact) return;

            // Remove old shapes
            bondShapes3d.forEach(s => {{ try {{ viewer3dContact.removeShape(s); }} catch(e) {{}} }});
            bondShapes3d.length = 0;

            (contactData3d.contact_pairs || []).forEach(pair => {{
                // Keep only types that are currently toggled on
                const visTypes = pair.types.filter(t => bondTypeVisible[t] !== false);
                if (!visTypes.length) return;

                const colors = visTypes.map(t => contactTypeColors3d[t] || '#888888');
                // Use the largest radius among visible types
                const radius = Math.max(...visTypes.map(t => contactTypeBondRadius[t] || 0.12));

                const start = {{x: pair.coord1[0], y: pair.coord1[1], z: pair.coord1[2]}};
                const end   = {{x: pair.coord2[0], y: pair.coord2[1], z: pair.coord2[2]}};
                const newShapes = addDashedLine(start, end, colors, radius, 0.5, 0.3);
                newShapes.forEach(s => bondShapes3d.push(s));
            }});

            viewer3dContact.render();
        }}

        function toggleBondType(type, visible) {{
            bondTypeVisible[type] = visible;
            drawContactBonds3d();
        }}

        function registerContactClickable3d() {{
            if (!viewer3dContact) return;
            const lookup = {{}};
            Object.entries(contactData3d.contact_residues_by_chain).forEach(([chain, residues]) => {{
                residues.forEach(res => {{ lookup[chain + ':' + res.resseq] = {{chain, ...res}}; }});
            }});
            viewer3dContact.setClickable({{}}, true, function(atom) {{
                showContactResiduePanel(atom, lookup[atom.chain + ':' + atom.resi] || null);
            }});
        }}

        function showContactResiduePanel(atom, resData) {{
            document.getElementById('cip-title').textContent =
                (atom.resn || '???') + ' ' + atom.chain + ':' + atom.resi;

            let rows = `
                <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f0f0f0;">
                    <span style="font-weight:600; color:#7f8c8d;">Chain</span>
                    <span style="color:#2c3e50;">${{atom.chain}}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f0f0f0;">
                    <span style="font-weight:600; color:#7f8c8d;">Position</span>
                    <span style="color:#2c3e50;">${{atom.resi}}</span>
                </div>`;

            if (resData) {{
                rows += `
                <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f0f0f0;">
                    <span style="font-weight:600; color:#7f8c8d;">Total Contacts</span>
                    <span style="color:#e74c3c; font-weight:700;">${{resData.contact_count}}</span>
                </div>`;
                if (resData.contact_types && resData.contact_types.length > 0) {{
                    rows += `<div style="padding:6px 0;">
                        <span style="font-weight:600; color:#7f8c8d; display:block; margin-bottom:5px;">Bond Types</span>
                        <div style="display:flex; gap:4px; flex-wrap:wrap;">`;
                    resData.contact_types.forEach(t => {{
                        const col = contactTypeColors3d[t] || '#888';
                        const lbl = contactTypeLabels3d[t] || t;
                        rows += `<span style="background:${{col}}; color:white; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:bold;">${{lbl}}</span>`;
                    }});
                    rows += `</div></div>`;
                }}
            }} else {{
                rows += `<div style="padding:6px 0; color:#aaa; font-style:italic;">No contacts at this residue</div>`;
            }}

            if (atom.b !== undefined && atom.b !== 0) {{
                rows += `
                <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f0f0f0;">
                    <span style="font-weight:600; color:#7f8c8d;">B-factor</span>
                    <span style="color:#2c3e50;">${{atom.b.toFixed(2)}}</span>
                </div>`;
            }}

            document.getElementById('cip-body').innerHTML = rows;
            document.getElementById('contact-info-panel').style.display = 'block';
        }}

        function closeContactPanel() {{
            document.getElementById('contact-info-panel').style.display = 'none';
        }}

        function setContactStyle(style) {{
            applyContactStyle3d(style);
            drawContactBonds3d();
        }}

        function resetContactView() {{
            if (viewer3dContact) {{ viewer3dContact.zoomTo(); viewer3dContact.render(); }}
        }}

        function toggleContactSpin() {{
            if (!viewer3dContact) return;
            isSpinning3dContact = !isSpinning3dContact;
            viewer3dContact.spin(isSpinning3dContact);
        }}
    """


# ==================== Example Usage ====================

def example_usage():
    """Demonstrate contact visualization."""
    from ..pdbparser import parse_pdb
    from ..contacts import analyze_contacts
    from ..interface import compute_interface
    
    # Load structure
    cx = parse_pdb("complex.pdb")
    
    # Analyze contacts
    contact_analysis = analyze_contacts(cx, "H,L", "A")
    
    # Optionally analyze interface too
    interface_analysis = compute_interface(cx, ["H", "L"], ["A"], cutoff=5.0)
    
    # Generate visualization
    html = generate_contact_html(
        contact_analysis,
        cx,
        interface_analysis=interface_analysis,
        title="Protein-Protein Contact Map",
        color_scheme="by_type"  # or "by_count"
    )
    
    # Save to file
    with open("contact_visualization.html", "w") as f:
        f.write(html)
    
    print("Visualization saved to contact_visualization.html")
    
    