from __future__ import annotations
from typing import List, Tuple, Iterable, Optional, Union
import math

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

from .models import Complex, Atom, Residue
from .selection import select, selection_residues 


# ===================================================================
# Selection Normalization (for compatibility with other scripts)
# ===================================================================

def _normalize_selection(sel):
    """
    Normalize selection to handle both string and list formats.
    
    Examples:
        "H,L" -> ["H", "L"]
        "H" -> ["H"]
        ["H", "L"] -> ["H", "L"]
        "H:32" -> ["H:32"]
        None -> None
    """
    if sel is None:
        return None
    if isinstance(sel, str):
        # Handle comma-separated chains (e.g., "H,L" -> ["H", "L"])
        if ',' in sel:
            return [s.strip() for s in sel.split(',')]
        return sel  # Single selection like "H" or "H:32"
    return sel  # Already list/tuple or Selection object


# ===================================================================
# Minimum Distance Between Point Sets
# ===================================================================

def pairwise_min_distance(
    coordsA: List[Tuple[float, float, float]],
    coordsB: List[Tuple[float, float, float]]
) -> float:
    """
    Calculate minimum distance between two sets of 3D coordinates.
    
    Args:
        coordsA: List of (x, y, z) tuples for first set
        coordsB: List of (x, y, z) tuples for second set
    
    Returns:
        Minimum distance in Angstroms, or inf if either set is empty
    """
    if not coordsA or not coordsB:
        return float("inf")
    
    if _HAS_NUMPY:
        A = np.array(coordsA, dtype=float)
        B = np.array(coordsB, dtype=float)
        diff = A[:, None, :] - B[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        return float(np.sqrt(d2.min()))
    
    best = float("inf")
    for (x1, y1, z1) in coordsA:
        for (x2, y2, z2) in coordsB:
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if d < best:
                best = d
    return best


# ===================================================================
# Helper Functions for Selection to Coordinates
# ===================================================================

def _atoms_from_expr(cx: Complex, expr_or_list) -> List[Atom]:
    """
    Extract atoms from selection expression.
    
    Args:
        cx: Complex structure
        expr_or_list: Selection expression (str, list, or Selection object)
            Examples: "H", "H,L", ["H", "L"], "H:32"
    
    Returns:
        List of Atom objects
    """
    expr_or_list = _normalize_selection(expr_or_list)
    return select(cx, expr_or_list).atoms


def _residues_from_expr(cx: Complex, expr_or_list) -> List[Residue]:
    """
    Extract residues from selection expression.
    
    Args:
        cx: Complex structure
        expr_or_list: Selection expression
    
    Returns:
        List of Residue objects in PDB order
    """
    expr_or_list = _normalize_selection(expr_or_list)
    return selection_residues(cx, select(cx, expr_or_list))


def _coords_of_atoms(atoms: List[Atom]) -> List[Tuple[float, float, float]]:
    """Extract (x, y, z) coordinates from atom list."""
    return [(a.x, a.y, a.z) for a in atoms]


def _coords_of_residue(
    res: Residue,
    mode: str = "CA",
    ignore_h: bool = True
) -> Tuple[Optional[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Get representative coordinate(s) for a residue.
    
    Args:
        res: Residue object
        mode: "CA" for alpha carbon, "centroid" for geometric center
        ignore_h: Exclude hydrogen atoms
    
    Returns:
        (representative_coord, all_coords) tuple where:
            - representative_coord: Single (x,y,z) or None if no atoms
            - all_coords: List of all heavy atom coordinates
    """
    # Get all heavy atom coordinates
    coords_full = [(a.x, a.y, a.z) for a in res.iter_atoms(ignore_h=ignore_h)]
    
    if not coords_full:
        return (None, [])
    
    if mode.upper() == "CA":
        # Try to find CA atom
        ca = next((a for a in res.atoms if a.name.strip().upper() == "CA"), None)
        if ca is not None:
            return ((ca.x, ca.y, ca.z), coords_full)
        
        # Fallback: centroid of heavy atoms
        sx = sum(x for x, _, _ in coords_full)
        sy = sum(y for _, y, _ in coords_full)
        sz = sum(z for _, _, z in coords_full)
        n = len(coords_full)
        return ((sx/n, sy/n, sz/n), coords_full)
    
    # For MIN mode, we only need all_coords
    return (None, coords_full)


# ===================================================================
# Atom-Level Distance Matrix
# ===================================================================

def atom_distance_matrix(
    cx: Complex,
    selA,
    selB=None,
    ignore_h: bool = True,
) -> Tuple[Union["np.ndarray", List[List[float]]], List[str], List[str]]:
    """
    Calculate atom-wise pairwise distance matrix.
    
    Args:
        cx: Complex structure
        selA: First selection (str like "H,L" or list like ["H", "L"])
        selB: Second selection (None = compare selA with itself)
        ignore_h: Exclude hydrogen atoms
    
    Returns:
        (distance_matrix, row_labels, col_labels) where:
            - distance_matrix: 2D array/list of distances
            - row_labels: Atom IDs for rows (e.g., ["H:32.CA", "H:33.N"])
            - col_labels: Atom IDs for columns
    
    Example:
        >>> D, rows, cols = atom_distance_matrix(cx, "H,L", "A")
        >>> # D[i,j] = distance between atoms rows[i] and cols[j]
    """
    # Get atoms for selection A
    atomsA = _atoms_from_expr(cx, selA)
    if ignore_h:
        atomsA = [a for a in atomsA 
                 if a.element.upper() != "H" and not a.name.strip().startswith("H")]
    coordsA = _coords_of_atoms(atomsA)
    labelsA = [a.atom_id for a in atomsA]
    
    # Get atoms for selection B (or use A if not specified)
    if selB is None:
        atomsB = atomsA
        coordsB = coordsA
        labelsB = labelsA
    else:
        atomsB = _atoms_from_expr(cx, selB)
        if ignore_h:
            atomsB = [a for a in atomsB 
                     if a.element.upper() != "H" and not a.name.strip().startswith("H")]
        coordsB = _coords_of_atoms(atomsB)
        labelsB = [a.atom_id for a in atomsB]
    
    # Calculate distance matrix
    if _HAS_NUMPY:
        A = np.array(coordsA, dtype=float) if coordsA else np.zeros((0, 3))
        B = np.array(coordsB, dtype=float) if coordsB else np.zeros((0, 3))
        
        if A.size == 0 or B.size == 0:
            D = np.zeros((len(coordsA), len(coordsB)), dtype=float)
        else:
            diff = A[:, None, :] - B[None, :, :]
            D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        
        return D, labelsA, labelsB
    else:
        # Pure Python implementation
        n, m = len(coordsA), len(coordsB)
        M = [[0.0] * m for _ in range(n)]
        for i, (x1, y1, z1) in enumerate(coordsA):
            for j, (x2, y2, z2) in enumerate(coordsB):
                dx, dy, dz = x1-x2, y1-y2, z1-z2
                M[i][j] = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        return M, labelsA, labelsB


# ===================================================================
# Residue-Level Distance Matrix
# ===================================================================

def residue_distance_matrix(
    cx: Complex,
    selA,
    selB=None,
    mode: str = "min",
    ignore_h: bool = True,
) -> Tuple[Union["np.ndarray", List[List[float]]], List[str], List[str]]:
    """
    Calculate residue-wise pairwise distance matrix.
    
    Args:
        cx: Complex structure
        selA: First selection (str like "H,L" or list like ["H", "L"])
        selB: Second selection (None = compare selA with itself)
        mode: Distance calculation method:
            - "min": Minimum atom-atom distance between residues
            - "CA": CA-CA distance (or centroid if CA missing)
        ignore_h: Exclude hydrogen atoms
    
    Returns:
        (distance_matrix, row_labels, col_labels) where:
            - distance_matrix: 2D array/list of distances
            - row_labels: Residue IDs (e.g., ["TYR H:32", "SER H:33"])
            - col_labels: Residue IDs for columns
    
    Example:
        >>> D, rows, cols = residue_distance_matrix(cx, "H,L", "A", mode="min")
        >>> # D[i,j] = min distance between residues rows[i] and cols[j]
    """
    mode = mode.upper()
    if mode not in ("MIN", "CA"):
        raise ValueError("residue_distance_matrix: mode must be 'min' or 'CA'")
    
    # Get residues
    resA = _residues_from_expr(cx, selA)
    labelsA = [r.id_str for r in resA]
    resB = resA if selB is None else _residues_from_expr(cx, selB)
    labelsB = [r.id_str for r in resB]
    
    if mode == "CA":
        # CA-CA distance mode
        coordsA = []
        coordsB = []
        
        for r in resA:
            c, _ = _coords_of_residue(r, mode="CA", ignore_h=ignore_h)
            coordsA.append(c)
        
        for r in resB:
            c, _ = _coords_of_residue(r, mode="CA", ignore_h=ignore_h)
            coordsB.append(c)
        
        def _dist_ca(c1, c2):
            """Calculate distance, return NaN if either coord is None."""
            if c1 is None or c2 is None:
                return float("nan")
            dx, dy, dz = c1[0]-c2[0], c1[1]-c2[1], c1[2]-c2[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        
        n, m = len(coordsA), len(coordsB)
        
        if _HAS_NUMPY:
            M = np.empty((n, m), dtype=float)
            for i in range(n):
                for j in range(m):
                    M[i, j] = _dist_ca(coordsA[i], coordsB[j])
            return M, labelsA, labelsB
        else:
            M = [[_dist_ca(coordsA[i], coordsB[j]) for j in range(m)] for i in range(n)]
            return M, labelsA, labelsB
    
    else:  # mode == "MIN"
        # Minimum atom-atom distance mode
        A_sets = [
            [(a.x, a.y, a.z) for a in r.iter_atoms(ignore_h=ignore_h)] 
            for r in resA
        ]
        B_sets = A_sets if selB is None else [
            [(a.x, a.y, a.z) for a in r.iter_atoms(ignore_h=ignore_h)] 
            for r in resB
        ]
        
        n, m = len(A_sets), len(B_sets)
        
        if _HAS_NUMPY:
            M = np.empty((n, m), dtype=float)
            for i in range(n):
                for j in range(m):
                    M[i, j] = pairwise_min_distance(A_sets[i], B_sets[j])
            return M, labelsA, labelsB
        else:
            M = [
                [pairwise_min_distance(A_sets[i], B_sets[j]) for j in range(m)] 
                for i in range(n)
            ]
            return M, labelsA, labelsB


# ===================================================================
# Convenience Functions
# ===================================================================

def all_distance_matrix(
    cx: Complex,
    level: str = "residue",
    residue_mode: str = "min",
    ignore_h: bool = True,
) -> Tuple[Union["np.ndarray", List[List[float]]], List[str]]:
    """
    Calculate distance matrix for entire complex.
    
    Args:
        cx: Complex structure
        level: "atom" or "residue"
        residue_mode: When level="residue", use "min" or "CA"
        ignore_h: Exclude hydrogen atoms
    
    Returns:
        (distance_matrix, labels) where matrix is square and symmetric
    
    Example:
        >>> D, labels = all_distance_matrix(cx, level="residue", residue_mode="CA")
        >>> # D is NxN matrix for all N residues in complex
    """
    # Use all chain IDs as selection
    expr = list(cx.chains.keys())
    
    if level.lower() == "atom":
        D, rows, _ = atom_distance_matrix(cx, expr, selB=None, ignore_h=ignore_h)
        return D, rows
    elif level.lower() == "residue":
        D, rows, _ = residue_distance_matrix(
            cx, expr, selB=None, mode=residue_mode, ignore_h=ignore_h
        )
        return D, rows
    else:
        raise ValueError("all_distance_matrix: level must be 'atom' or 'residue'")


def selection_distance_matrix(
    cx: Complex,
    selA,
    selB=None,
    level: str = "residue",
    residue_mode: str = "min",
    ignore_h: bool = True,
) -> Tuple[Union["np.ndarray", List[List[float]]], List[str], List[str]]:
    """
    Unified entry point for distance matrix calculation.
    
    Args:
        cx: Complex structure
        selA: First selection (e.g., "H,L" or ["H", "L"])
        selB: Second selection (None = within selA)
        level: "atom" or "residue"
        residue_mode: When level="residue", use "min" or "CA"
        ignore_h: Exclude hydrogen atoms
    
    Returns:
        (distance_matrix, row_labels, col_labels)
    
    Example:
        >>> # Atom-wise distance between antibody and antigen
        >>> D, ab, ag = selection_distance_matrix(cx, "H,L", "A", level="atom")
        
        >>> # Residue-wise CA-CA distances
        >>> D, ab, ag = selection_distance_matrix(
        ...     cx, "H,L", "A", level="residue", residue_mode="CA"
        ... )
    """
    if level.lower() == "atom":
        return atom_distance_matrix(cx, selA, selB, ignore_h=ignore_h)
    elif level.lower() == "residue":
        return residue_distance_matrix(cx, selA, selB, mode=residue_mode, ignore_h=ignore_h)
    else:
        raise ValueError("selection_distance_matrix: level must be 'atom' or 'residue'")


# ===================================================================
# Example Usage
# ===================================================================

def example_usage():
    """Demonstrate distance matrix calculations."""
    from .pdbparser import parse_pdb
    
    cx = parse_pdb("complex.pdb")
    
    # 1. Atom-wise distances between antibody and antigen
    D_atom, ab_atoms, ag_atoms = atom_distance_matrix(cx, "H,L", "A")
    print(f"Atom distance matrix: {D_atom.shape if _HAS_NUMPY else (len(D_atom), len(D_atom[0]))}")
    
    # 2. Residue-wise minimum distances
    D_min, ab_res, ag_res = residue_distance_matrix(cx, "H,L", "A", mode="min")
    print(f"Residue min distance matrix: {D_min.shape if _HAS_NUMPY else (len(D_min), len(D_min[0]))}")
    
    # 3. CA-CA distances
    D_ca, ab_res, ag_res = residue_distance_matrix(cx, "H,L", "A", mode="CA")
    print(f"CA-CA distance matrix: {D_ca.shape if _HAS_NUMPY else (len(D_ca), len(D_ca[0]))}")
    
    # 4. All-vs-all for entire complex
    D_all, all_labels = all_distance_matrix(cx, level="residue", residue_mode="CA")
    print(f"All-vs-all matrix: {D_all.shape if _HAS_NUMPY else (len(D_all), len(D_all[0]))}")
    
    # 5. Find closest contacts
    if _HAS_NUMPY:
        import numpy as np
        min_dist = np.min(D_min)
        i, j = np.unravel_index(np.argmin(D_min), D_min.shape)
        print(f"Closest contact: {ab_res[i]} - {ag_res[j]} = {min_dist:.2f} Å")
    
    return D_atom, D_min, D_ca

