from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math

from .models import Complex, Residue, Atom
from .selection import select, selection_residues

# # interpreted as AMBER Rmin/2 for LJ 
VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "X": 1.75,
}
def _vdw(elem: str) -> float:
    return VDW_RADIUS.get((elem or "X").upper(), VDW_RADIUS["X"])


# ---------------------------
# Per-element LJ well depths (kcal/mol)
# ---------------------------
PER_EPS: Dict[str, float] = {
    "C": 0.12, 
    "N": 0.17, 
    "O": 0.20, 
    "S": 0.25, 
    "H": 0.02,
    "P": 0.20,
    "F": 0.06,
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


# ---------------------------
# Per-residue Lennard-Jones (AMBER form)
# ---------------------------
def per_residue_LJ_decomposition(
    cx: Complex,
    left_exprs,
    right_exprs,
    ignore_h: bool = True,
    cutoff: float = 5.0,
    eps_table: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Per-residue van der Waals energy on LEFT vs RIGHT using AMBER 12-6 form:

      E_ij = eps_ij * [ (Rmin_ij / r)^12 - 2*(Rmin_ij / r)^6 ]
      
    Where:
      Rmin_ij = (Rmin_i/2 + Rmin_j/2) = ri + rj
      ri = _vdw(elem_i) interpreted as AMBER Rmin/2
      eps_ij = sqrt(eps_i * eps_j)
      eps_i from eps_table or PER_EPS

    Args:
        cx: Complex object
        left_exprs: Selection expression for left side (e.g., "chain A")
        right_exprs: Selection expression for right side (e.g., "chain B")
        ignore_h: If True, ignore hydrogen atoms (default: True)
        cutoff: Distance cutoff in Angstroms (default: 5.0)
        eps_table: Optional custom epsilon values by element
    
    Returns:
        Dictionary mapping residue_id_str to VDW energy in kcal/mol
        Negative values indicate favorable interactions
    
    Example:
        >>> energies = per_residue_LJ_decomposition(
        ...     complex,
        ...     ["A"],
        ...     ["B"],
        ...     cutoff=5.0
        ... )
        >>> print(f"Residue A_52: {energies['A_52']:.2f} kcal/mol")
    """
    
    eps_tab = eps_table or PER_EPS

    L_res = selection_residues(cx, select(cx, left_exprs))
    R_atoms = select(cx, right_exprs).atoms
    if ignore_h:
        R_atoms = [a for a in R_atoms if not _is_h(a)]
    if not L_res or not R_atoms:
        return {}

    cutoff2 = cutoff * cutoff

    out: Dict[str, float] = {}
    for res in L_res:
        e = 0.0
        for ai in res.iter_atoms(ignore_h=ignore_h):
            # AMBER Rmin/2 from your internal _vdw
            ri = _vdw(ai.element)
            if ri is None or ri <= 0:
                continue
            ei = eps_tab.get((ai.element or "").strip().upper(), None)
            if ei is None or ei <= 0:
                ei = 0.1  # conservative fallback

            # Handle coord attribute properly
            if hasattr(ai, 'coord'):
                xi, yi, zi = ai.coord
            else:
                xi, yi, zi = ai.x, ai.y, ai.z

            for aj in R_atoms:
                rj = _vdw(aj.element)
                if rj is None or rj <= 0:
                    continue
                ej = eps_tab.get((aj.element or "").strip().upper(), None)
                if ej is None or ej <= 0:
                    ej = 0.1

                # Handle coord attribute properly for aj too
                if hasattr(aj, 'coord'):
                    xj, yj, zj = aj.coord
                else:
                    xj, yj, zj = aj.x, aj.y, aj.z

                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r2 = dx*dx + dy*dy + dz*dz
                
                if r2 <= 1e-12 or r2 > cutoff2:
                    continue

                Rmin_ij = ri + rj
                eps_ij  = math.sqrt(ei * ej)
                inv_r = 1.0 / math.sqrt(r2)
                sr = Rmin_ij * inv_r
                sr6 = sr**6
                sr12 = sr6 * sr6

                # AMBER 12-6 form
                e += eps_ij * (sr12 - 2.0 * sr6)
        out[res.id_str] = e

    return out



def per_residue_pair_LJ(
    cx: Complex,
    chain_exprs: List[str],
    ignore_h: bool = True,
    cutoff: float = 5.0,
    eps_table: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate pairwise VDW energies between all chain pairs.
    
    Args:
        cx: Complex object
        chain_exprs: List of chain selection expressions
        ignore_h: If True, ignore hydrogen atoms
        cutoff: Distance cutoff in Angstroms
        eps_table: Optional custom epsilon values
    
    Returns:
        Nested dictionary: {(chain1, chain2): {residue_id: energy}}
    
    Example:
        >>> energies = per_residue_pair_LJ(
        ...     complex,
        ...     ["A", "B", "C"]
        ... )
        >>> ab_energies = energies[("A", "B")]
    """
    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    
    for i, left_expr in enumerate(chain_exprs):
        for right_expr in chain_exprs[i+1:]:
            # Calculate in both directions
            left_to_right = per_residue_LJ_decomposition(
                cx, left_expr, right_expr, ignore_h, cutoff, eps_table
            )
            right_to_left = per_residue_LJ_decomposition(
                cx, right_expr, left_expr, ignore_h, cutoff, eps_table
            )
            
            results[(left_expr, right_expr)] = left_to_right
            results[(right_expr, left_expr)] = right_to_left
    
    return results


# ---------------------------
# Hotspot Analysis
# ---------------------------
def rank_hotspots(
    per_res_energy: Dict[str, float], 
    topk: int = 10,
    sort_ascending: bool = True,
) -> List[Tuple[str, float]]:
    """
    Rank and display residues by VDW energy.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        topk: Number of top residues to display
        sort_ascending: If True, most negative (favorable) first
    
    Returns:
        Sorted list of (residue_id, energy) tuples
    
    Example:
        >>> energies = per_residue_LJ_decomposition(cx, ["A"], ["B"])
        >>> hotspots = rank_hotspots(energies, topk=20)
    """
    # Sort ascending (most negative energies first for favorable interactions)
    top_residues = sorted(
        per_res_energy.items(), 
        key=lambda x: x[1], 
        reverse=not sort_ascending
    ) 
    
    print(f"{'Residue':<12s} {'LJ Energy (kcal/mol)':<20s}")
    print('-' * 40)
    for res_id, energy in top_residues[:topk]:
        print(f"{res_id:<12s} {energy:>18.5f}")
    
    return top_residues


def identify_energetic_hotspots(
    per_res_energy: Dict[str, float],
    energy_threshold: float = -2.0,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Identify energetic hotspots (residues with strong VDW interactions).
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        energy_threshold: Maximum energy to be considered a hotspot (kcal/mol)
                         Negative values indicate favorable interactions
        top_n: Optional limit on number of hotspots to return
    
    Returns:
        List of (residue_id, energy) tuples sorted by energy

    """
    hotspots = [
        (res_id, energy) 
        for res_id, energy in per_res_energy.items()
        if energy <= energy_threshold
    ]
    
    # Sort by energy (most negative first)
    hotspots.sort(key=lambda x: x[1])
    
    if top_n is not None:
        hotspots = hotspots[:top_n]
    
    return hotspots


def compare_energetic_contributions(
    per_res_energy: Dict[str, float],
    energy_threshold: float = -2.0,
) -> Dict[str, Dict[str, float]]:
    """
    Summarize energetic contributions by chain.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
    
    Returns:
        Dictionary with chain-level statistics:
        {
            chain_id: {
                'total_energy': sum of all residue energies,
                'favorable_energy': sum of negative energies,
                'unfavorable_energy': sum of positive energies,
                'num_hotspots': count of residues with energy < -2.0,
                'avg_energy': average energy per residue,
                'num_residues': total number of residues
            }
        }

    """
    from collections import defaultdict
    
    chain_data: Dict[str, List[float]] = defaultdict(list)
    
    # Group by chain - extract chain ID from residue ID
    for res_id, energy in per_res_energy.items():
        # Extract chain from residue ID (assumes format like "ASP A_123" or "ASP A:123")
        
        if '_' in res_id:
            chain = res_id.split('_')[0].split(' ')[1]
        elif ':' in res_id:
            chain = res_id.split(':')[0].split(' ')[1]
        else:
            # Fallback: assume first character is chain ID
            chain = res_id[0] if res_id else "?"
        
        chain_data[chain].append(energy)
    
    chain_ids = set(sorted(chain_data.keys()))
    
    # Compute statistics
    summary: Dict[str, Dict[str, float]] = {}
    
    for chain in chain_ids:
        energies = chain_data.get(chain, [])
        
        if not energies:
            # Chain requested but no data found - return zeros
            summary[chain] = {
                'total_energy': 0.0,
                'favorable_energy': 0.0,
                'unfavorable_energy': 0.0,
                'num_hotspots': 0,
                'avg_energy': 0.0,
                'num_residues': 0,
            }
            continue
        
        total = sum(energies)
        favorable = sum(e for e in energies if e < 0)
        unfavorable = sum(e for e in energies if e >= 0)
        hotspots = sum(1 for e in energies if e <= energy_threshold)
        avg = total / len(energies)
        
        summary[chain] = {
            'total_energy': total,
            'favorable_energy': favorable,
            'unfavorable_energy': unfavorable,
            'num_hotspots': hotspots,
            'avg_energy': avg,
            'num_residues': len(energies),
        }
    
    return summary



def print_vdw_report(
    per_res_energy: Dict[str, float],
    top_n: int = 20,
    energy_threshold: float = -2.0,
):
    """
    Print a formatted report of VDW energy analysis.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        top_n: Number of top residues to display
        energy_threshold: Energy cutoff for hotspots (kcal/mol)
    
    Example:
        >>> energies = per_residue_LJ_decomposition(cx, "chain A", "chain B")
        >>> print_vdw_report(energies, top_n=20, energy_threshold=-2.0)
    """
    print("=" * 80)
    print("VDW ENERGY ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary statistics
    all_energies = list(per_res_energy.values())
    total = sum(all_energies)
    favorable = sum(e for e in all_energies if e < 0)
    unfavorable = sum(e for e in all_energies if e >= 0)
    avg = total / len(all_energies) if all_energies else 0.0
    
    print("\n--- OVERALL SUMMARY ---")
    print(f"Total residues:        {len(all_energies)}")
    print(f"Total VDW energy:      {total:>10.2f} kcal/mol")
    print(f"Favorable energy:      {favorable:>10.2f} kcal/mol")
    print(f"Unfavorable energy:    {unfavorable:>10.2f} kcal/mol")
    print(f"Average per residue:   {avg:>10.2f} kcal/mol")
    
    # Chain summary
    summary = compare_energetic_contributions(per_res_energy, energy_threshold)
    
    print("\n--- CHAIN SUMMARY ---")
    print(f"{'Chain':<8} {'#Res':<8} {'Total':<15} {'Favorable':<15} {'Hotspots':<12}")
    print("-" * 80)
    for chain, stats in summary.items():
        print(f"{chain:<8} {int(stats['num_residues']):<8} "
              f"{stats['total_energy']:>12.2f} "
              f"{stats['favorable_energy']:>12.2f} "
              f"{int(stats['num_hotspots']):<12}")
    
    # Top hotspots
    hotspots = identify_energetic_hotspots(
        per_res_energy, 
        energy_threshold=energy_threshold,
        top_n=top_n
    )
    
    print(f"\n--- TOP {len(hotspots)} ENERGETIC HOTSPOTS "
          f"(Energy < {energy_threshold:.1f} kcal/mol) ---")
    print(f"{'Residue':<12} {'Energy (kcal/mol)':<20}")
    print("-" * 40)
    
    for res_id, energy in hotspots:
        print(f"{res_id:<12} {energy:>18.5f}")
    
    print(f"\nTotal hotspots: {len(hotspots)}")
    print("=" * 80)


def export_vdw_results(
    per_res_energy: Dict[str, float],
    output_file: str = "vdw_energies.csv",
):
    """
    Export VDW energy results to CSV file.
    
    Args:
        per_res_energy: Dictionary of residue_id -> energy
        output_file: Output CSV filename
    
    Example:
        >>> energies = per_residue_LJ_decomposition(cx, "chain A", "chain B")
        >>> export_vdw_results(energies, "interface_energies.csv")
    """
    with open(output_file, 'w') as f:
        f.write("residue_id,chain,resnum,vdw_energy,is_hotspot\n")
        
        for res_id, energy in sorted(per_res_energy.items()):
            # Parse residue ID
            parts = res_id.split('_')
            chain = parts[0] if len(parts) > 0 else "?"
            resnum = parts[1] if len(parts) > 1 else "?"
            
            is_hotspot = "YES" if energy <= -2.0 else "NO"
            
            f.write(f"{res_id},{chain},{resnum},{energy:.5f},{is_hotspot}\n")
    
    print(f"VDW energies exported to {output_file}")
