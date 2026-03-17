from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Tuple, Union 
import re 
import math 
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def _coerce_chain(chain) -> str:
    # Ensure a single-character chain label
    if chain is None:
        return "A"
    # If someone stored chain as an int (e.g., 0), map to letters A..Z cyclically
    if isinstance(chain, int):
        return chr(ord('A') + (chain % 26))
    s = str(chain)
    return s[0] if s else "A"

def _coerce_resseq(resseq) -> int:
    # residueNumber must be an int; strip insertion codes like "101A" -> 101
    if resseq is None:
        return 1
    if isinstance(resseq, int):
        return resseq
    m = re.match(r"\d+", str(resseq))
    return int(m.group(0)) if m else 1

def _sasa_fs(atoms) -> float:
    import freesasa
    freesasa.setVerbosity(freesasa.nowarnings)
    struc = freesasa.Structure()
    for a in atoms:
        name   = str(getattr(a, "name",  a.element if hasattr(a, "element") else "X"))
        resn   = str(getattr(a, "resname", "UNK"))
        chain  = _coerce_chain(getattr(a, "chain_id", "A"))
        resseq = _coerce_resseq(getattr(a, "resseq", 1))
        x = float(a.x); y = float(a.y); z = float(a.z)
        struc.addAtom(name, resn, resseq, chain,  x, y, z)
        # OPTIONAL: set a custom vdW radius (else FreeSASA will use its classifier)
        # struc.setRadius(struc.nAtoms() - 1, _vdw(getattr(a, "element", "X")))

    return float(freesasa.calc(struc).totalArea())


# ----------------- Shrake–Rupley SASA (workaround) ----------------- #
# Minimal element -> VdW radius for fallback SASA (used to build freesasa struct)
VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "X": 1.75,
}
def _vdw(elem: str) -> float:
    return VDW_RADIUS.get((elem or "X").upper(), VDW_RADIUS["X"])

def _fibonacci_sphere(n: int) -> List[Tuple[float, float, float]]:
    """
    Even-ish distribution of points on a unit sphere (deterministic).
    """
    pts: List[Tuple[float, float, float]] = []
    # golden angle
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = i * ga
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        pts.append((x, y, z))
    return pts

_SPHERE_CACHE: Dict[int, List[Tuple[float, float, float]]] = {}

def _sphere_points(n: int) -> List[Tuple[float, float, float]]:
    if n not in _SPHERE_CACHE:
        _SPHERE_CACHE[n] = _fibonacci_sphere(n)
    return _SPHERE_CACHE[n]

def _approx_sasa(atoms, probe: float = 1.4, n_points: int = 960, ignore_h: bool = True) -> float:
    """
    Approximate SASA via Shrake–Rupley point sampling.
    Returns total SASA (Å^2).
    Complexity ~ O(N^2 * n_points). For large N, consider a grid/kd-tree.
    """
    # Prepare atom arrays
    cent: List[Tuple[float, float, float]] = []
    radii: List[float] = []
    for a in atoms:
        if ignore_h and (a.element.upper() == "H" or a.name.strip().startswith("H")):
            continue
        cent.append((a.x, a.y, a.z))
        radii.append(_vdw(a.element) + probe)

    if not cent:
        return 0.0

    pts = _sphere_points(n_points)  # unit vectors
    if _HAS_NUMPY:
        C = np.asarray(cent, dtype=float)            # (N,3)
        R = np.asarray(radii, dtype=float)           # (N,)
        exposed_counts = np.zeros(len(cent), dtype=int)
        # naive neighbor checks
        for i in range(len(cent)):
            ci = C[i]                                # (3,)
            ri = R[i]
            # candidate surface sample points in world coords
            S = ci[None, :] + ri * np.asarray(pts)   # (P,3)
            # For each sample, check if any other atom eclipses it
            # We can vectorize against all other atoms j
            for s in S:
                d2 = np.sum((C - s) ** 2, axis=1)    # (N,)
                # ignore atom i itself by forcing its radius small
                cover = d2 < (R ** 2) - 1e-8
                cover[i] = False
                if not np.any(cover):
                    exposed_counts[i] += 1
        # area per exposed sample on sphere of radius ri ~ 4πr^2 / P
        areas = (4.0 * math.pi * (R ** 2)) * (exposed_counts / float(n_points))
        return float(np.sum(areas))
    else:
        exposed_counts = [0] * len(cent)
        for i, (xi, yi, zi) in enumerate(cent):
            ri = radii[i]
            area_weight = 4.0 * math.pi * ri * ri / float(n_points)
            for (ux, uy, uz) in pts:
                sx = xi + ri * ux; sy = yi + ri * uy; sz = zi + ri * uz
                occluded = False
                for j, (xj, yj, zj) in enumerate(cent):
                    if j == i:
                        continue
                    rj = radii[j]
                    dx = sx - xj; dy = sy - yj; dz = sz - zj
                    if dx*dx + dy*dy + dz*dz < rj * rj - 1e-8:
                        occluded = True
                        break
                if not occluded:
                    exposed_counts[i] += 1
        total = 0.0
        for i, ri in enumerate(radii):
            total += (4.0 * math.pi * ri * ri) * (exposed_counts[i] / float(n_points))
        return total

def _compute_bsa_fallback(A_atoms, B_atoms, probe: float = 1.4, n_points: int = 960) -> Tuple[float, str]:
    """
    Compute BSA via approximate SASA when freesasa is unavailable.
    Returns (bsa, note).
    """
    sasaA = _approx_sasa(A_atoms, probe=probe, n_points=n_points, ignore_h=False)
    sasaB = _approx_sasa(B_atoms, probe=probe, n_points=n_points, ignore_h=False)
    sasaAB = _approx_sasa(A_atoms + B_atoms, probe=probe, n_points=n_points, ignore_h=False)
    # Common convention divides by 2 to average the two sides' buried areas
    bsa = (sasaA + sasaB - sasaAB) / 2.0
    note = f"BSA via approximate Shrake-Rupley (probe={probe}, points={n_points}); freesasa not used."
    return bsa, note


 