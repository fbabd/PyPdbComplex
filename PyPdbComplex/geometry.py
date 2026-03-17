from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict, Optional, Union
import math
import csv

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False 

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
    
from .models import Complex, Residue, Atom, Selection 
from .selection import select, selection_residues 


# ===================================================================
# Vector Math Utilities
# ===================================================================

def _vec(a: Atom, b: Atom) -> Tuple[float, float, float]:
    """Vector from atom a to atom b."""
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _dot(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
    """Dot product of two vectors."""
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def _cross(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Cross product of two vectors."""
    return (
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0],
    )


def _norm(u: Tuple[float, float, float]) -> float:
    """Euclidean norm of vector."""
    return math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)  


def _unit(u: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Unit vector in direction of u."""
    n = _norm(u)
    if n < 1e-10:  # Better threshold than 0.0
        return (0.0, 0.0, 0.0)
    return (u[0]/n, u[1]/n, u[2]/n)


def angle_deg(a: Atom, b: Atom, c: Atom) -> float:
    """
    Calculate angle ∠ABC in degrees.
    
    Args:
        a: First atom
        b: Vertex atom (center)
        c: Third atom
    
    Returns:
        Angle in degrees [0, 180]
    """
    u = _unit(_vec(b, a))
    v = _unit(_vec(b, c))
    cos_theta = max(-1.0, min(1.0, _dot(u, v)))
    return math.degrees(math.acos(cos_theta))



def dihedral_deg(p0: Atom, p1: Atom, p2: Atom, p3: Atom) -> float:
    """
    Calculate dihedral (torsion) angle defined by four atoms.
    link: https://leimao.github.io/blog/Dihedral-Angles/ 
    Returns:
        Dihedral angle in degrees in (-180, 180]
    """
    # Define bond vectors: p0->p1, p1->p2, p2->p3
    b0 = _vec(p0, p1)
    b1 = _vec(p1, p2)
    b2 = _vec(p2, p3)


    v = _cross(b0, b1)
    w = _cross(b1, b2)
    y = _norm(v)*_norm(w)
    
    cosTheta = _dot(v, w) / y
    
    sinTheta = _dot(_cross(v, w), _unit(b1)) / y
    

    return math.degrees(math.atan2(sinTheta,cosTheta))


# ===================================================================
# Data Classes for Geometry Results
# ===================================================================

@dataclass(frozen=True)
class BackboneAngles:
    """Backbone dihedral angles for a single residue."""
    chain_id: str
    resseq: int
    icode: str
    resname: str
    phi: Optional[float]    # N-terminal residues have None
    psi: Optional[float]    # C-terminal residues have None
    omega: Optional[float]  # Peptide bond planarity (cis ~0°, trans ~180°)
    
    @property
    def residue_id(self) -> str:
        """Residue identifier string."""
        ic = self.icode.strip()
        return f"{self.resname} {self.chain_id}:{self.resseq}{ic}"
    
    @property
    def is_valid(self) -> bool:
        """Check if at least one angle is computed."""
        return self.phi is not None or self.psi is not None or self.omega is not None


@dataclass(frozen=True)
class SidechainChi:
    """Sidechain chi angles for a single residue."""
    chain_id: str
    resseq: int
    icode: str
    resname: str
    chis: Dict[str, Optional[float]]  # {"chi1": deg, "chi2": ..., may be None}
    
    @property
    def residue_id(self) -> str:
        """Residue identifier string."""
        ic = self.icode.strip()
        return f"{self.resname} {self.chain_id}:{self.resseq}{ic}"
    
    @property
    def num_chis(self) -> int:
        """Number of defined chi angles for this residue type."""
        return len([v for v in self.chis.values() if v is not None])


@dataclass(frozen=True)
class BendAngle:
    """CA trace bend angle at a residue."""
    chain_id: str
    resseq_center: int
    icode_center: str
    resname_center: str
    angle_deg: Optional[float]  # None when neighbors missing
    
    @property
    def residue_id(self) -> str:
        """Residue identifier string."""
        ic = self.icode_center.strip()
        return f"{self.resname_center} {self.chain_id}:{self.resseq_center}{ic}"
    
    @property
    def is_straight(self) -> bool:
        """Check if angle is close to 180° (straight, extended)."""
        return self.angle_deg is not None and self.angle_deg > 160.0
    
    @property
    def is_bent(self) -> bool:
        """Check if angle is significantly bent (<140°)."""
        return self.angle_deg is not None and self.angle_deg < 140.0


@dataclass
class GeometryAnalysis:
    """Container for all geometry analysis results."""
    backbone: Optional[List[BackboneAngles]] = None
    sidechains: Optional[List[SidechainChi]] = None
    bends: Optional[List[BendAngle]] = None
    
    def to_dataframe(self, analysis_type: str = "backbone"):
        """
        Export to pandas DataFrame.
        
        Args:
            analysis_type: "backbone", "sidechains", or "bends"
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")
        
        if analysis_type == "backbone" and self.backbone:
            rows = [asdict(r) for r in self.backbone]
            return pd.DataFrame(rows)
        elif analysis_type == "sidechains" and self.sidechains:
            rows = _as_rows_chi(self.sidechains)
            return pd.DataFrame(rows)
        elif analysis_type == "bends" and self.bends:
            rows = [asdict(r) for r in self.bends]
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()
    
    def save_csv(self, prefix: str = "geometry"):
        """Save all analyses to CSV files with given prefix."""
        if self.backbone:
            save_csv(f"{prefix}_backbone.csv", [asdict(r) for r in self.backbone])
        if self.sidechains:
            save_csv(f"{prefix}_sidechains.csv", _as_rows_chi(self.sidechains))
        if self.bends:
            save_csv(f"{prefix}_bends.csv", [asdict(r) for r in self.bends])
    
    def get_outliers(self, analysis_type: str = "backbone") -> List:
        """
        Identify outlier conformations.
        
        Args:
            analysis_type: "backbone" or "bends"
        
        Returns:
            List of outlier records
        """
        if analysis_type == "backbone" and self.backbone:
            # Ramachandran outliers (simplified)
            outliers = []
            for rec in self.backbone:
                if rec.phi is not None and rec.psi is not None:
                    # Very simplified: flag unusual combinations
                    # Real implementation should use proper Ramachandran regions
                    if abs(rec.phi) > 175 and abs(rec.psi) > 175:
                        outliers.append(rec)
            return outliers
        elif analysis_type == "bends" and self.bends:
            # Unusually sharp bends
            return [b for b in self.bends if b.angle_deg and b.angle_deg < 100.0]
        return []


# ===================================================================
# Sidechain Chi Angle Definitions
# ===================================================================

# Standard χ definitions (IUPAC) for protein sidechains
# Each entry: list of (atom1, atom2, atom3, atom4) tuples defining chi1..chiN
CHI_DEFS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "ARG": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ")
    ],
    "ASN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1")
    ],
    "ASP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1")
    ],
    "CYS": [
        ("N", "CA", "CB", "SG")
    ],
    "GLN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1")
    ],
    "GLU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1")
    ],
    "HIS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1")
    ],
    "ILE": [
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1")
    ],
    "LEU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1")
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ")
    ],
    "MET": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE")
    ],
    "PHE": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1")
    ],
    "PRO": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD")
    ],
    "SER": [
        ("N", "CA", "CB", "OG")
    ],
    "THR": [
        ("N", "CA", "CB", "OG1")
    ],
    "TRP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1")
    ],
    "TYR": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1")
    ],
    "VAL": [
        ("N", "CA", "CB", "CG1")
    ],
    # GLY and ALA have no chi angles
}


# ===================================================================
# Helper Functions
# ===================================================================

def _get_atom(res: Residue, name: str) -> Optional[Atom]:
    """
    Get atom by name from residue, handling altlocs.
    Chooses highest occupancy if multiple altlocs exist.
    """
    # Try direct access first (if Residue has get_atom method)
    if hasattr(res, 'get_atom'):
        try:
            return res.get_atom(name)
        except:
            pass
    
    # Fallback: scan atoms
    best = None
    best_occ = -1.0
    name_upper = name.strip().upper()
    
    for a in res.iter_atoms(ignore_h=False):
        if a.name.strip().upper() == name_upper:
            occ = getattr(a, "occupancy", 1.0) or 1.0
            if occ > best_occ:
                best, best_occ = a, occ
    
    return best


def _sorted_residues(residues: List[Residue]) -> List[Residue]:
    """Sort residues by chain, resseq, icode."""
    return sorted(residues, key=lambda r: (
        r.chain_id,
        r.resseq,
        getattr(r, "icode", "") or ""
    ))


# ===================================================================
# Export Utilities
# ===================================================================

def _as_rows_chi(recs: List[SidechainChi]) -> List[Dict[str, object]]:
    """Convert SidechainChi records to flat dict rows with chi1..chi4 columns."""
    rows = []
    for r in recs:
        row = {
            "chain_id": r.chain_id,
            "resseq": r.resseq,
            "icode": r.icode,
            "resname": r.resname
        }
        # Ensure chi1..chi4 columns exist (even if None)
        for i in range(1, 5):
            row[f"chi{i}"] = r.chis.get(f"chi{i}")
        rows.append(row)
    return rows


def save_csv(path: str, rows: List[Dict[str, object]]):
    """Save data to CSV file."""
    if not rows:
        # Create empty file
        with open(path, "w", newline="") as f:
            pass
        return
    
    if _HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    else:
        # CSV fallback
        header = set()
        for r in rows:
            header.update(r.keys())
        header = sorted(header)  # Consistent column order
        
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)


# ===================================================================
# Core Geometry Calculations
# ===================================================================

def compute_backbone_angles(cx: Complex, selection: Union[str, Selection, Iterable[Residue]]) -> List[BackboneAngles]:
    """
        Compute backbone dihedral angles (φ, ψ, ω) for selected residues.
    
        Args:
            cx: Complex structure
            selection: Residue selection  
        
        Returns:
            List of BackboneAngles records
        
        Notes:
            - φ (phi): C(i-1) - N(i) - CA(i) - C(i)
            - ψ (psi): N(i) - CA(i) - C(i) - N(i+1)
            - ω (omega): CA(i) - C(i) - N(i+1) - CA(i+1) (peptide bond)
            - N-terminal residues have phi=None
            - C-terminal residues have psi=None
    """
    sel = select(cx, selection)
    residues = _sorted_residues(selection_residues(cx, sel))
    
    out: List[BackboneAngles] = []
    
    # Group by chain
    by_chain: Dict[str, List[Residue]] = {}
    for r in residues:
        by_chain.setdefault(r.chain_id, []).append(r)
    
    for chain_id, rlist in by_chain.items():
        n = len(rlist)

        for i, r in enumerate(rlist):
            N_i  = _get_atom(r, "N")
            CA_i = _get_atom(r, "CA")
            C_i  = _get_atom(r, "C")

            phi = psi = omega = None

            # Previous residue contiguous? (simple integer check, adjust if you care about gaps/missing residues)  
            if i > 0:
                r_prev = rlist[i - 1]
                if r_prev.resseq == r.resseq - 1:  # or more sophisticated gap/icode logic
                    C_prev = _get_atom(r_prev, "C")
                    if C_prev and N_i and CA_i and C_i:
                        phi = dihedral_deg(C_prev, N_i, CA_i, C_i)

            # Next residue contiguous?
            if i < n - 1:
                r_next = rlist[i + 1]
                if r_next.resseq == r.resseq + 1:
                    N_next  = _get_atom(r_next, "N")
                    CA_next = _get_atom(r_next, "CA")

                    if N_i and CA_i and C_i and N_next:
                        psi = dihedral_deg(N_i, CA_i, C_i, N_next)

                    if CA_i and C_i and N_next and CA_next:
                        omega = dihedral_deg(CA_i, C_i, N_next, CA_next)
            # if i != 0:
            #     print(C_prev.serial, N_i.serial, CA_i.serial, C_i.serial, phi, psi )
            
            out.append(BackboneAngles(
                chain_id=r.chain_id,
                resseq=r.resseq,
                icode=getattr(r, "icode", "") or "",
                resname=r.resname,
                phi=phi,
                psi=psi,
                omega=omega
            ))
    
    return out


def compute_sidechain_chis(cx: Complex, selection: Union[str, Selection, Iterable[Residue]]) -> List[SidechainChi]:
    """
        Compute sidechain chi angles for selected residues.
        
        Args:
            cx: Complex structure
            selection: Residue selection
        
        Returns:
            List of SidechainChi records
        
        Notes:
            - Only residues with defined chi angles are included
            - GLY and ALA have no chi angles
            - Missing atoms result in chi=None for that angle 
    """
    sel = select(cx, selection)
    residues = _sorted_residues(selection_residues(cx, sel))
    
    out: List[SidechainChi] = []
    
    for r in residues:
        resname_upper = r.resname.upper()
        
        # Skip residues without chi definitions
        if resname_upper not in CHI_DEFS:
            continue
        
        chis: Dict[str, Optional[float]] = {}
        defs = CHI_DEFS[resname_upper]
        
        for idx, (a, b, c, d) in enumerate(defs, start=1):
            chi_name = f"chi{idx}"
            
            A = _get_atom(r, a)
            B = _get_atom(r, b)
            C = _get_atom(r, c)
            D = _get_atom(r, d)
            
            if A and B and C and D:
                chis[chi_name] = dihedral_deg(A, B, C, D)
            else:
                chis[chi_name] = None
        
        if chis:  # Only add if we have chi definitions
            out.append(SidechainChi(
                chain_id=r.chain_id,
                resseq=r.resseq,
                icode=getattr(r, "icode", "") or "",
                resname=r.resname,
                chis=chis
            ))
    
    return out


def compute_ca_bend_angles(cx: Complex, selection: Union[str, Selection, Iterable[Residue]]) -> List[BendAngle]:
    """
        Compute CA trace bend angles for selected residues.
        
        Args:
            cx: Complex structure
            selection: Residue selection
        
        Returns:
            List of BendAngle records
        
        Notes:
            - Angle at residue i is: CA(i-1) - CA(i) - CA(i+1)
            - Terminal residues have angle=None
            - 180° = straight/extended
            - <140° = significantly bent 
    """
    sel = select(cx, selection)
    residues = _sorted_residues(selection_residues(cx, sel)) 
    
    out: List[BendAngle] = []
    
    # Group by chain
    by_chain: Dict[str, List[Residue]] = {}
    for r in residues:
        by_chain.setdefault(r.chain_id, []).append(r)
    
    for chain_id, rlist in by_chain.items():
        for i, r in enumerate(rlist):
            # Get CAs from neighbors
            CA_prev = _get_atom(rlist[i-1], "CA") if i > 0 else None
            CA_i = _get_atom(r, "CA")
            CA_next = _get_atom(rlist[i+1], "CA") if i < len(rlist) - 1 else None
            
            # Calculate bend angle
            ang = None
            if CA_prev and CA_i and CA_next:
                ang = angle_deg(CA_prev, CA_i, CA_next)
            
            out.append(BendAngle(
                chain_id=chain_id,
                resseq_center=r.resseq,
                icode_center=getattr(r, "icode", "") or "",
                resname_center=r.resname,
                angle_deg=ang
            ))
    
    return out


def analyze_geometry(
    cx: Complex,
    selection: Union[str, Selection, Iterable[Residue]],
    compute_backbone: bool = True,
    compute_sidechains: bool = True,
    compute_bends: bool = True
) -> GeometryAnalysis:
    """
        Comprehensive geometry analysis in one call.
        
        Args:
            cx: Complex structure
            selection: Residue selection
            compute_backbone: Calculate phi/psi/omega angles
            compute_sidechains: Calculate chi angles
            compute_bends: Calculate CA bend angles
        
        Returns:
            GeometryAnalysis object with all results
    """
    analysis = GeometryAnalysis()
    
    if compute_backbone:
        analysis.backbone = compute_backbone_angles(cx, selection)
    
    if compute_sidechains:
        analysis.sidechains = compute_sidechain_chis(cx, selection)
    
    if compute_bends:
        analysis.bends = compute_ca_bend_angles(cx, selection)
    
    return analysis

