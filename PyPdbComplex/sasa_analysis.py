from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import freesasa as freesasa
    freesasa.setVerbosity(freesasa.nowarnings)
    _HAS_FREESASA = True
except ImportError:
    _HAS_FREESASA = False

from .models import Complex, Residue, Atom
from .selection import select, selection_residues


# ---------------------------
# Van der Waals radii (Bondi, 1964)
# ---------------------------
VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "X": 1.75,  # Unknown/generic
}


# ---------------------------
# Helper Functions
# ---------------------------
def _is_h(atom: Atom) -> bool:
    """Check if atom is hydrogen based on element or name."""
    e = (atom.element or "").strip().upper()
    if e == "H":
        return True
    nm = (atom.name or "").strip().upper()
    return nm.startswith("H")


def _coerce_chain(chain) -> str:
    """Ensure a single-character chain label."""
    if chain is None:
        return "A"
    # If someone stored chain as an int (e.g., 0), map to letters A..Z cyclically
    if isinstance(chain, int):
        return chr(ord('A') + (chain % 26))
    s = str(chain)
    return s[0] if s else "A"


def _coerce_resseq(resseq) -> int:
    """Convert residue sequence number to int, stripping insertion codes."""
    if resseq is None:
        return 1
    if isinstance(resseq, int):
        return resseq
    m = re.match(r"\d+", str(resseq))
    return int(m.group(0)) if m else 1


def _fs_add_atom(fs_struct: "freesasa.Structure", a: Atom, radius: float) -> None:
    """
    Add an atom to a FreeSASA structure.

    Args:
        fs_struct: FreeSASA Structure object
        a: Atom object
        radius: VDW radius in Angstroms
    """
    name   = str(getattr(a, "name",  a.element if hasattr(a, "element") else "X"))
    resn   = str(getattr(a, "resname", "UNK"))
    chain  = _coerce_chain(getattr(a, "chain_id", "A"))
    resseq = _coerce_resseq(getattr(a, "resseq", 1))

    if hasattr(a, 'coord'):
        x, y, z = a.coord
    else:
        x = float(a.x)
        y = float(a.y)
        z = float(a.z)

    rad = float(radius) if radius and radius > 0 else 1.8

    fs_struct.addAtom(name, resn, resseq, chain, x, y, z)
    # Apply custom VdW radius (overrides FreeSASA's internal classifier)
    atom_idx = fs_struct.nAtoms() - 1
    fs_struct.setRadius(atom_idx, rad)


def _fs_structure_from_residues(
    residues: List[Residue],
    ignore_h: bool = True
) -> "freesasa.Structure":
    """
    Build FreeSASA structure from a list of Residue objects.

    Args:
        residues: List of Residue objects
        ignore_h: If True, ignore hydrogen atoms

    Returns:
        FreeSASA Structure object
    """
    if not _HAS_FREESASA:
        raise ImportError("freesasa is required for SASA calculations. Install it with: pip install freesasa")
    fs_struct = freesasa.Structure()
    for r in residues:
        for a in r.iter_atoms(ignore_h=False):
            if ignore_h and _is_h(a):
                continue
            # Get VDW radius from table
            elem = (a.element or "").strip().upper()
            rad = VDW_RADIUS.get(elem, 1.8)
            _fs_add_atom(fs_struct, a, rad)
    return fs_struct


# ---------------------------
# Core SASA Calculations
# ---------------------------
def per_residue_sasa(
    cx: Complex,
    exprs,
    probe: float = 1.4,
    n_points: int = 200,
    algorithm: str = "sr",
    ignore_h: bool = True,
) -> Dict[str, float]:
    """
        Calculate per-residue SASA (Ų) for residues in selection.
        
        IMPORTANT: This calculates SASA in the context of the selection.
        Residues within the selection occlude each other.
        For bound vs unbound analysis, use compare_bound_unbound_sasa().
        
        Args:
            cx: Complex object
            exprs: Selection expression(s)
            probe: Probe radius in Angstroms (default: 1.4, water)
            n_points: Number of test points/slices (higher = more accurate)
            algorithm: "sr" (Shrake-Rupley, faster) or "lr" (Lee-Richards, more accurate)
            ignore_h: If True, ignore hydrogen atoms
        
        Returns:
            Dictionary mapping residue_id_str to SASA in Ų
        
        Example:
            >>> # Calculate SASA for chain A in isolation
            >>> sasa_a = per_residue_sasa(complex, ["A"])
            >>> print(f"Residue A_52: {sasa_a['A_52']:.2f} Ų")
    """
    L_res = selection_residues(cx, select(cx, exprs))
    if not L_res:
        return {}

    fs_struct = _fs_structure_from_residues(L_res, ignore_h=ignore_h)

    # Configure FreeSASA parameters
    params = freesasa.Parameters()
    params.setProbeRadius(float(probe))
    
    if str(algorithm).lower().startswith("lr"):
        params.setAlgorithm(freesasa.LeeRichards)
        params.setNSlices(int(n_points))
    else:
        params.setAlgorithm(freesasa.ShrakeRupley)
        params.setNPoints(int(n_points))

    # Calculate SASA
    result = freesasa.calc(fs_struct, params)
    areas = result.residueAreas()

    # Build lookup from (chain, resseq) -> id_str
    by_chain_resnum: Dict[tuple, str] = {}
    for r in L_res:
        ch  = _coerce_chain(getattr(r, "chain_id", "A"))
        num = _coerce_resseq(getattr(r, "resseq", 1))
        by_chain_resnum[(ch, int(num))] = r.id_str

    out: Dict[str, float] = {}

    if not areas:
        for r in L_res:
            out[r.id_str] = 0.0
        return out

    # Parse FreeSASA results (can be nested dict or tuple-keyed)
    try:
        first_val = next(iter(areas.values()))
        if isinstance(first_val, dict):
            # Case 1: nested dict format
            for chain, resmap in areas.items():
                ch = _coerce_chain(chain)
                for resnum_str, area in resmap.items():
                    try:
                        num = int(resnum_str)
                    except (ValueError, TypeError):
                        num = _coerce_resseq(resnum_str)
                    rid = by_chain_resnum.get((ch, num))
                    if rid is not None:
                        out[rid] = float(area.total)
        else:
            # Case 2: tuple-keyed format
            for key, area in areas.items():
                try:
                    chain, resnum, _resname = key
                except (ValueError, TypeError):
                    continue
                ch  = _coerce_chain(chain)
                num = _coerce_resseq(resnum)
                rid = by_chain_resnum.get((ch, int(num)))
                if rid is not None:
                    out[rid] = float(area.total)
    except (StopIteration, AttributeError):
        pass

    # Ensure every residue appears (0.0 if missing)
    for r in L_res:
        out.setdefault(r.id_str, 0.0)

    return out


def per_atom_sasa(
    cx: Complex,
    exprs,
    probe: float = 1.4,
    n_points: int = 200,
    algorithm: str = "sr",
    ignore_h: bool = True,
) -> Dict[int, float]:
    """
    Calculate per-atom SASA (Ų) for atoms in selection.
    
    This function provides atom-level granularity, useful for:
    - Identifying specific exposed atoms
    - Fine-grained surface analysis
    - Atom-level property calculations
    
    Args:
        cx: Complex object
        exprs: Selection expression(s)
        probe: Probe radius in Angstroms
        n_points: Number of test points/slices
        algorithm: "sr" or "lr"
        ignore_h: If True, ignore hydrogen atoms
    
    Returns:
        Dictionary mapping atom_id (id(atom)) to SASA in Ų
    
    Example:
        >>> atom_sasa = per_atom_sasa(complex, "chain A and resid 50")
        >>> for atom_id, sasa in sorted(atom_sasa.items(), 
        ...                             key=lambda x: x[1], reverse=True)[:5]:
        ...     print(f"Atom {atom_id}: {sasa:.2f} Ų")
    """
    atoms = select(cx, exprs).atoms
    if ignore_h:
        atoms = [a for a in atoms if not _is_h(a)]
    
    if not atoms:
        return {}
    
    # Build FreeSASA structure from atoms
    fs_struct = freesasa.Structure()
    atom_indices: Dict[int, int] = {}
    
    for idx, a in enumerate(atoms):
        elem = (a.element or "").strip().upper()
        rad = VDW_RADIUS.get(elem, 1.8)
        _fs_add_atom(fs_struct, a, rad)
        atom_indices[id(a)] = idx
    
    # Configure FreeSASA
    params = freesasa.Parameters()
    params.setProbeRadius(float(probe))
    if str(algorithm).lower().startswith("lr"):
        params.setAlgorithm(freesasa.LeeRichards)
        params.setNSlices(int(n_points))
    else:
        params.setAlgorithm(freesasa.ShrakeRupley)
        params.setNPoints(int(n_points))
    
    result = freesasa.calc(fs_struct, params)
    
    # Extract per-atom SASA values
    out: Dict[int, float] = {}
    for a in atoms:
        idx = atom_indices[id(a)]
        try:
            sasa = result.atomArea(idx)
            out[id(a)] = float(sasa)
        except Exception:
            out[id(a)] = 0.0
    
    return out


# ---------------------------
# Bound vs Unbound SASA Comparison
# ---------------------------
def compare_bound_unbound_sasa(
    cx: Complex,
    chain_exprs: List[str],
    probe: float = 1.4,
    n_points: int = 200,
    algorithm: str = "lr",
    ignore_h: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate SASA for each chain in isolation (unbound) and in complex (bound).
    
    This is the CORRECT way to analyze binding interfaces.
    
    Args:
        cx: Complex object
        chain_exprs: List of selection expressions for each chain
                    e.g., ["A", "B"]
        probe: Probe radius (Å)
        n_points: Number of test points/slices
        algorithm: "sr" (Shrake-Rupley) or "lr" (Lee-Richards)
        ignore_h: Ignore hydrogen atoms
    
    Returns:
        Dictionary mapping residue_id_str to:
        {
            'unbound': SASA when chain is alone (Ų),
            'bound': SASA when in complex (Ų),
            'delta': SASA_unbound - SASA_bound (Ų),
            'buried_fraction': delta / unbound (0-1)
        }
    
    Example:
        >>> # Compare chain A and chain B
        >>> results = compare_bound_unbound_sasa(cx, ["chain A", "chain B"])
        >>> 
        >>> # Find interface residues (buried >50% upon binding)
        >>> interface = {r: v for r, v in results.items() 
        ...              if v['buried_fraction'] > 0.5}
    """
    # Step 1: Calculate unbound SASA for each chain separately
    unbound_sasa: Dict[str, float] = {}
    
    for chain_expr in chain_exprs:
        chain_sasa = per_residue_sasa(cx, chain_expr, probe, n_points, algorithm, ignore_h)
        unbound_sasa.update(chain_sasa)
    
    # Step 2: Calculate bound SASA for the entire complex
    all_chains_expr = chain_exprs 
    bound_sasa = per_residue_sasa(cx, all_chains_expr, probe, n_points, algorithm, ignore_h)
    
    # Step 3: Calculate delta SASA and buried fraction
    results: Dict[str, Dict[str, float]] = {}
    
    for res_id in unbound_sasa:
        unbound = unbound_sasa.get(res_id, 0.0)
        bound = bound_sasa.get(res_id, 0.0)
        delta = unbound - bound
        buried_frac = delta / unbound if unbound > 0 else 0.0
        
        results[res_id] = {
            'unbound': unbound,
            'bound': bound,
            'delta': delta,
            'buried_fraction': buried_frac,
        }
    
    return results


# ---------------------------
# Interface Analysis
# ---------------------------
def identify_interface_residues(
    sasa_results: Dict[str, Dict[str, float]],
    burial_threshold: float = 0.5,
    min_delta_sasa: float = 10.0,
) -> List[Tuple[str, float, float]]:
    """
    Identify interface residues based on SASA burial.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        burial_threshold: Minimum burial fraction (0-1) to be considered interface
        min_delta_sasa: Minimum absolute ΔSASA (Ų) to be considered interface
    
    Returns:
        List of (residue_id, buried_fraction, delta_sasa) sorted by burial
    
    Example:
        >>> results = compare_bound_unbound_sasa(cx, ["A", "B"])
        >>> interface = identify_interface_residues(
        ...     results,
        ...     burial_threshold=0.5,  # >50% buried
        ...     min_delta_sasa=10.0    # >10 Ų buried
        ... )
    """
    interface = []
    
    for res_id, values in sasa_results.items():
        if (values['buried_fraction'] >= burial_threshold and 
            values['delta'] >= min_delta_sasa):
            interface.append((
                res_id,
                values['buried_fraction'],
                values['delta']
            ))
    
    # Sort by burial fraction (most buried first)
    interface.sort(key=lambda x: x[1], reverse=True)
    
    return interface


def summarize_sasa_by_chain(
    sasa_results: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Summarize SASA results by chain.
    
    If chain_ids is not provided, automatically detects all unique chain IDs
    from the residue IDs in sasa_results.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        chain_ids: Optional list of chain IDs to summarize. If None, auto-detects
                  all chains from the residue IDs.
    
    Returns:
        Dictionary with chain-level statistics:
        {
            chain_id: {
                'total_unbound': sum of unbound SASA (Ų),
                'total_bound': sum of bound SASA (Ų),
                'total_buried': sum of buried SASA (Ų),
                'interface_residues': count of residues with >50% burial,
                'avg_burial_fraction': average burial fraction,
                'num_residues': total number of residues
            }
        }
    
    Example:
        >>> results = compare_bound_unbound_sasa(cx, ["chain A", "chain B"])
        >>> # Auto-detect chains
        >>> summary = summarize_sasa_by_chain(results)
        >>> print(f"Chain A buried surface: {summary['A']['total_buried']:.1f} Ų")
        >>> 
        >>> # Or specify chains explicitly
        >>> summary = summarize_sasa_by_chain(results, chain_ids=["A", "B"])
    """
    from collections import defaultdict
    
    chain_data: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    
    # Group by chain - extract chain ID from residue ID
    # Actual format is "TYR H:32" (resname space chain:resnum)
    for res_id, values in sasa_results.items():
        try:
            if ':' in res_id:
                # "TYR H:32" -> chain = "H"
                before_colon = res_id.split(':')[0]
                parts = before_colon.strip().split()
                chain = parts[-1] if parts else "?"
            elif '_' in res_id:
                # Legacy "H_32" or "TYR H_32" format
                before_sep = res_id.split('_')[0]
                parts = before_sep.strip().split()
                chain = parts[-1] if parts else "?"
            else:
                chain = res_id[0] if res_id else "?"
        except Exception:
            chain = "?"
        chain_data[chain].append(values)
    
    # If chain_ids is None, use all detected chains (auto-detect)
    chain_ids = set(sorted(chain_data.keys())) 
    
    
    # Summarize each chain
    summary: Dict[str, Dict[str, float]] = {}
    
    for chain in chain_ids:
        residues = chain_data.get(chain, [])
        
        if not residues:
            # Chain requested but no data found - return zeros
            summary[chain] = {
                'total_unbound': 0.0,
                'total_bound': 0.0,
                'total_buried': 0.0,
                'interface_residues': 0,
                'avg_burial_fraction': 0.0,
                'num_residues': 0,
            }
            continue
        
        total_unbound = sum(r['unbound'] for r in residues)
        total_bound = sum(r['bound'] for r in residues)
        total_buried = sum(r['delta'] for r in residues)
        interface_count = sum(1 for r in residues if r['buried_fraction'] > 0.5)
        avg_burial = sum(r['buried_fraction'] for r in residues) / len(residues)
        
        summary[chain] = {
            'total_unbound': total_unbound,
            'total_bound': total_bound,
            'total_buried': total_buried,
            'interface_residues': interface_count,
            'avg_burial_fraction': avg_burial,
            'num_residues': len(residues),
        }
    
    return summary


# ---------------------------
# Reporting and Export
# ---------------------------
def print_sasa_report(
    sasa_results: Dict[str, Dict[str, float]],
    top_n: int = 20,
    burial_threshold: float = 0.5,
):
    """
    Print a formatted report of SASA analysis.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        top_n: Number of top interface residues to show
        burial_threshold: Minimum burial fraction for interface residues
    
    Example:
        >>> results = compare_bound_unbound_sasa(cx, ["chain A", "chain B"])
        >>> print_sasa_report(results, top_n=20)
    """
    print("=" * 80)
    print("SASA ANALYSIS REPORT: Bound vs Unbound")
    print("=" * 80)
    
    # Summary by chain
    summary = summarize_sasa_by_chain(sasa_results)
    
    print("\n--- CHAIN SUMMARY ---")
    print(f"{'Chain':<8} {'#Res':<8} {'Unbound':<12} {'Bound':<12} {'Buried':<12} {'Interface':<12}")
    print("-" * 80)
    for chain, stats in summary.items():
        print(f"{chain:<8} {int(stats['num_residues']):<8} "
              f"{stats['total_unbound']:>10.1f} Ų "
              f"{stats['total_bound']:>10.1f} Ų "
              f"{stats['total_buried']:>10.1f} Ų "
              f"{int(stats['interface_residues']):<12}")
    
    # Interface residues
    interface = identify_interface_residues(sasa_results, burial_threshold)
    
    print(f"\n--- TOP {top_n} INTERFACE RESIDUES (Burial > {burial_threshold*100:.0f}%) ---")
    print(f"{'Residue':<12} {'Unbound':<12} {'Bound':<12} {'ΔSASA':<12} {'Buried %':<12}")
    print("-" * 80)
    
    for res_id, burial_frac, delta in interface[:top_n]:
        values = sasa_results[res_id]
        print(f"{res_id:<12} "
              f"{values['unbound']:>10.2f} Ų "
              f"{values['bound']:>10.2f} Ų "
              f"{delta:>10.2f} Ų "
              f"{burial_frac*100:>10.1f}%")
    
    print(f"\nTotal interface residues: {len(interface)}")
    print("=" * 80)


def export_sasa_results(
    sasa_results: Dict[str, Dict[str, float]],
    output_file: str = "sasa_results.csv",
):
    """
    Export SASA results to CSV file.
    
    Args:
        sasa_results: Output from compare_bound_unbound_sasa()
        output_file: Output CSV filename
    
    Example:
        >>> results = compare_bound_unbound_sasa(cx, ["chain A", "chain B"])
        >>> export_sasa_results(results, "interface_sasa.csv")
    """
    with open(output_file, 'w') as f:
        f.write("residue_id,chain,resnum,unbound,bound,delta,buried_fraction,is_interface\n")
        
        for res_id, values in sorted(sasa_results.items()):
            # Parse residue ID: format is "TYR H:32" (resname chain:resnum)
            if ':' in res_id:
                before_colon = res_id.split(':')[0]
                parts = before_colon.strip().split()
                chain = parts[-1] if parts else "?"
                resnum = res_id.split(':')[1].strip() if len(res_id.split(':')) > 1 else "?"
            elif '_' in res_id:
                before_sep = res_id.split('_')[0]
                parts = before_sep.strip().split()
                chain = parts[-1] if parts else "?"
                resnum = res_id.split('_')[1].strip() if len(res_id.split('_')) > 1 else "?"
            else:
                chain = res_id[0] if res_id else "?"
                resnum = "?"
            
            is_interface = "YES" if values['buried_fraction'] > 0.5 else "NO"
            
            f.write(f"{res_id},{chain},{resnum},"
                   f"{values['unbound']:.2f},{values['bound']:.2f},"
                   f"{values['delta']:.2f},{values['buried_fraction']:.4f},"
                   f"{is_interface}\n")
    
    print(f"SASA results exported to {output_file}")

