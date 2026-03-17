from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

from .models import Complex, Atom, Residue
from .selection import select, selection_residues

# ==================== Constants ====================

# Residue classifications
POSITIVE_RES = {"LYS", "ARG"}
NEGATIVE_RES = {"ASP", "GLU"}
HYDROPHOBIC_RES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"}

# Aromatic ring atom definitions
AROMATIC_RINGS = {
    "PHE": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
    "TYR": [["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]],
    "TRP": [["CD2", "CE2", "CE3", "CZ3", "CZ2", "CH2"], 
            ["CG", "CD1", "NE1", "CE2", "CD2"]],
    "HIS": [["CG", "ND1", "CE1", "NE2", "CD2"]],
}

# Interaction cutoffs (Ã…)
HBOND_CUTOFF = 3.5
SALT_BRIDGE_CUTOFF = 4.0
DISULFIDE_CUTOFF = 2.2
HYDROPHOBIC_CUTOFF = 4.5
PI_STACK_DIST_RANGE = (4.5, 7.0)
PI_PARALLEL_ANGLE_TOL = 30.0
PI_T_ANGLE_TOL = 30.0

# Charged atom definitions for salt bridges (at pH 7.4)
POSITIVE_ATOMS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE")}
NEGATIVE_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}

# pKa values for titratable residues (standard values in water)
PKA_VALUES = {
    "ASP": 3.9,   # Aspartic acid (acidic)
    "GLU": 4.2,   # Glutamic acid (acidic)
    "HIS": 6.0,   # Histidine (can be positive or neutral)
    "CYS": 8.3,   # Cysteine (usually protonated/neutral)
    "TYR": 10.1,  # Tyrosine (usually protonated/neutral)
    "LYS": 10.5,  # Lysine (basic)
    "ARG": 12.5,  # Arginine (basic)
    "N_TERM": 8.0,  # N-terminus
    "C_TERM": 3.1,  # C-terminus
}

# Atoms for pH-dependent charged groups
PH_DEPENDENT_ATOMS = {
    "ASP": {
        "negative": [("ASP", "OD1"), ("ASP", "OD2")],  # Deprotonated form (COO-)
        "neutral": [("ASP", "OD1"), ("ASP", "OD2")],   # Protonated form (COOH)
    },
    "GLU": {
        "negative": [("GLU", "OE1"), ("GLU", "OE2")],  # Deprotonated form (COO-)
        "neutral": [("GLU", "OE1"), ("GLU", "OE2")],   # Protonated form (COOH)
    },
    "HIS": {
        "positive": [("HIS", "ND1"), ("HIS", "NE2")],  # Protonated form (HIS+)
        "neutral": [("HIS", "ND1"), ("HIS", "NE2")],   # Neutral form
    },
    "LYS": {
        "positive": [("LYS", "NZ")],                   # Protonated form (NH3+)
        "neutral": [("LYS", "NZ")],                    # Deprotonated form (NH2)
    },
    "ARG": {
        "positive": [("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE")],  # Protonated (always at normal pH)
        "neutral": [("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE")],   # Deprotonated (rare)
    },
}

# Hydrogen bond atoms (simplified heavy atom approach)
HBOND_ELEMENTS = {"N", "O", "S"}


# ==================== Parameter Configuration ====================

@dataclass
class ContactParameters:
    """
    Configuration parameters for contact detection.
    
    All distance values are in Ångströms (Å).
    All angle values are in degrees.
    """
    # Distance cutoffs
    hbond_cutoff: float = HBOND_CUTOFF
    salt_bridge_cutoff: float = SALT_BRIDGE_CUTOFF
    disulfide_cutoff: float = DISULFIDE_CUTOFF
    hydrophobic_cutoff: float = HYDROPHOBIC_CUTOFF
    
    # Pi-stacking parameters
    pi_stack_dist_min: float = PI_STACK_DIST_RANGE[0]
    pi_stack_dist_max: float = PI_STACK_DIST_RANGE[1]
    pi_parallel_angle_tol: float = PI_PARALLEL_ANGLE_TOL
    pi_t_angle_tol: float = PI_T_ANGLE_TOL
    
    # pH for protonation state calculations
    ph: float = 7.4  # Physiological pH
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate parameter values."""
        # Check distance cutoffs are positive
        if self.hbond_cutoff <= 0:
            raise ValueError(f"hbond_cutoff must be positive, got {self.hbond_cutoff}")
        if self.salt_bridge_cutoff <= 0:
            raise ValueError(f"salt_bridge_cutoff must be positive, got {self.salt_bridge_cutoff}")
        if self.disulfide_cutoff <= 0:
            raise ValueError(f"disulfide_cutoff must be positive, got {self.disulfide_cutoff}")
        if self.hydrophobic_cutoff <= 0:
            raise ValueError(f"hydrophobic_cutoff must be positive, got {self.hydrophobic_cutoff}")
        
        # Check pi-stacking distance range
        if self.pi_stack_dist_min <= 0:
            raise ValueError(f"pi_stack_dist_min must be positive, got {self.pi_stack_dist_min}")
        if self.pi_stack_dist_max <= self.pi_stack_dist_min:
            raise ValueError(
                f"pi_stack_dist_max ({self.pi_stack_dist_max}) must be greater than "
                f"pi_stack_dist_min ({self.pi_stack_dist_min})"
            )
        
        # Check angle tolerances are between 0 and 90
        if not (0 <= self.pi_parallel_angle_tol <= 90):
            raise ValueError(f"pi_parallel_angle_tol must be between 0 and 90, got {self.pi_parallel_angle_tol}")
        if not (0 <= self.pi_t_angle_tol <= 90):
            raise ValueError(f"pi_t_angle_tol must be between 0 and 90, got {self.pi_t_angle_tol}")
        
        # Check pH is reasonable
        if not (0 <= self.ph <= 14):
            raise ValueError(f"pH must be between 0 and 14, got {self.ph}")
    
    @property
    def pi_stack_dist_range(self) -> Tuple[float, float]:
        """Get pi-stacking distance range as tuple."""
        return (self.pi_stack_dist_min, self.pi_stack_dist_max)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            'hbond_cutoff': self.hbond_cutoff,
            'salt_bridge_cutoff': self.salt_bridge_cutoff,
            'disulfide_cutoff': self.disulfide_cutoff,
            'hydrophobic_cutoff': self.hydrophobic_cutoff,
            'pi_stack_dist_min': self.pi_stack_dist_min,
            'pi_stack_dist_max': self.pi_stack_dist_max,
            'pi_parallel_angle_tol': self.pi_parallel_angle_tol,
            'pi_t_angle_tol': self.pi_t_angle_tol,
            'ph': self.ph,
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'ContactParameters':
        """Create ContactParameters from dictionary."""
        return cls(**params)
    
    def copy(self) -> 'ContactParameters':
        """Create a copy of the parameters."""
        return ContactParameters(**self.to_dict())
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContactParameters(\n"
            f"  H-bond: {self.hbond_cutoff} Å\n"
            f"  Salt bridge: {self.salt_bridge_cutoff} Å\n"
            f"  Disulfide: {self.disulfide_cutoff} Å\n"
            f"  Hydrophobic: {self.hydrophobic_cutoff} Å\n"
            f"  π-stacking dist: {self.pi_stack_dist_min}-{self.pi_stack_dist_max} Å\n"
            f"  π-stacking angles: parallel={self.pi_parallel_angle_tol}°, T={self.pi_t_angle_tol}°\n"
            f"  pH: {self.ph}\n"
            f")"
        )


def setup_contact_parameters(
    hbond_cutoff: Optional[float] = None,
    salt_bridge_cutoff: Optional[float] = None,
    disulfide_cutoff: Optional[float] = None,
    hydrophobic_cutoff: Optional[float] = None,
    pi_stack_dist_min: Optional[float] = None,
    pi_stack_dist_max: Optional[float] = None,
    pi_parallel_angle_tol: Optional[float] = None,
    pi_t_angle_tol: Optional[float] = None,
    ph: Optional[float] = None,
) -> ContactParameters:
    """
    Set up contact detection parameters.
    
    Args:
        hbond_cutoff: Maximum distance for hydrogen bonds (Å). Default: 3.5
        salt_bridge_cutoff: Maximum distance for salt bridges (Å). Default: 4.0
        disulfide_cutoff: Maximum distance for disulfide bonds (Å). Default: 2.2
        hydrophobic_cutoff: Maximum distance for hydrophobic contacts (Å). Default: 4.5
        pi_stack_dist_min: Minimum centroid distance for π-stacking (Å). Default: 4.5
        pi_stack_dist_max: Maximum centroid distance for π-stacking (Å). Default: 7.0
        pi_parallel_angle_tol: Angle tolerance for parallel π-stacking (degrees). Default: 30.0
        pi_t_angle_tol: Angle tolerance for T-shaped π-stacking (degrees). Default: 30.0
        ph: pH value for protonation state calculations. Default: 7.4
    
    Returns:
        ContactParameters object with validated values
    
    Examples:
        >>> # Use default parameters (pH 7.4)
        >>> params = setup_contact_parameters()
        
        >>> # Customize specific parameters including pH
        >>> params = setup_contact_parameters(hbond_cutoff=3.0, salt_bridge_cutoff=4.5, ph=6.0)
        
        >>> # Use in analysis
        >>> analysis = analyze_contacts(cx, ["H","L"], ["A"], params=params)
    """
    # Build kwargs from non-None values
    kwargs = {}
    if hbond_cutoff is not None:
        kwargs['hbond_cutoff'] = hbond_cutoff
    if salt_bridge_cutoff is not None:
        kwargs['salt_bridge_cutoff'] = salt_bridge_cutoff
    if disulfide_cutoff is not None:
        kwargs['disulfide_cutoff'] = disulfide_cutoff
    if hydrophobic_cutoff is not None:
        kwargs['hydrophobic_cutoff'] = hydrophobic_cutoff
    if pi_stack_dist_min is not None:
        kwargs['pi_stack_dist_min'] = pi_stack_dist_min
    if pi_stack_dist_max is not None:
        kwargs['pi_stack_dist_max'] = pi_stack_dist_max
    if pi_parallel_angle_tol is not None:
        kwargs['pi_parallel_angle_tol'] = pi_parallel_angle_tol
    if pi_t_angle_tol is not None:
        kwargs['pi_t_angle_tol'] = pi_t_angle_tol
    if ph is not None:
        kwargs['ph'] = ph
    
    return ContactParameters(**kwargs)


# ==================== Data Classes ====================

class ContactType(Enum):
    """Types of molecular contacts."""
    HYDROGEN_BOND = "hydrogen_bond"
    SALT_BRIDGE = "salt_bridge"
    DISULFIDE = "disulfide"
    HYDROPHOBIC = "hydrophobic"
    PI_STACKING = "pi_stacking"


@dataclass
class Contact:
    """Represents a single molecular contact."""
    type: ContactType
    atom1: Atom
    atom2: Atom
    distance: float
    metadata: Dict = field(default_factory=dict)
    
    @property
    def residue1_id(self) -> str:
        """Return full residue ID with resname (e.g., 'TYR H:32')"""
        ic = self.atom1.icode.strip()
        return f"{self.atom1.resname} {self.atom1.chain_id}:{self.atom1.resseq}{ic}"
    
    @property
    def residue2_id(self) -> str:
        """Return full residue ID with resname (e.g., 'SER A:45')"""
        ic = self.atom2.icode.strip()
        return f"{self.atom2.resname} {self.atom2.chain_id}:{self.atom2.resseq}{ic}"
    
    @property
    def residue_pair(self) -> Tuple[str, str]:
        return (self.residue1_id, self.residue2_id)


@dataclass
class ContactAnalysis:
    """Complete contact analysis with summaries."""
    contacts: List[Contact] = field(default_factory=list)
    complex: Optional[Complex] = None
    
    def filter_by_type(self, contact_type: ContactType) -> List[Contact]:
        """Get all contacts of a specific type."""
        return [c for c in self.contacts if c.type == contact_type]
    
    def get_contact_counts(self) -> Dict[ContactType, int]:
        """Get total count for each contact type."""
        counts = defaultdict(int)
        for contact in self.contacts:
            counts[contact.type] += 1
        return dict(counts)
    
    def get_residue_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get contact counts per residue.
        Returns: {resid_str: {contact_type: count}}
        """
        summary = defaultdict(lambda: defaultdict(int))
        
        for contact in self.contacts:
            summary[contact.residue1_id][contact.type.value] += 1
            summary[contact.residue2_id][contact.type.value] += 1
        
        # Convert to regular dict and add total
        result = {}
        for resid, type_counts in summary.items():
            result[resid] = dict(type_counts)
            result[resid]['total'] = sum(type_counts.values())
        
        return result
    
    def get_atom_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get contact counts per atom.
        Returns: {atom_id: {contact_type: count}}
        """
        summary = defaultdict(lambda: defaultdict(int))
        
        for contact in self.contacts:
            atom1_id = contact.atom1.atom_id
            atom2_id = contact.atom2.atom_id
            summary[atom1_id][contact.type.value] += 1
            summary[atom2_id][contact.type.value] += 1
        
        # Convert to regular dict and add total
        result = {}
        for atom_id, type_counts in summary.items():
            result[atom_id] = dict(type_counts)
            result[atom_id]['total'] = sum(type_counts.values())
        
        return result
    
    def get_residue_partners(self, residue_id: str) -> Dict[str, List[Tuple[str, ContactType, float]]]:
        """
        Get all partner residues for a given residue.
        Returns: {partner_resid: [(atom_pair, ContactType, distance), ...]}
        """
        partners = defaultdict(list)
        
        for contact in self.contacts:
            if contact.residue1_id == residue_id:
                partners[contact.residue2_id].append(
                    (f"{contact.atom1.name.strip()}-{contact.atom2.name.strip()}", 
                     contact.type, contact.distance)
                )
            elif contact.residue2_id == residue_id:
                partners[contact.residue1_id].append(
                    (f"{contact.atom2.name.strip()}-{contact.atom1.name.strip()}", 
                     contact.type, contact.distance)
                )
        
        return dict(partners)
    
    def to_dataframe(self):
        """Export to pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        rows = []
        for contact in self.contacts:
            row = {
                "contact_type": contact.type.value,
                "atom1_id": contact.atom1.atom_id,
                "atom2_id": contact.atom2.atom_id,
                "atom1_name": contact.atom1.name.strip(),
                "atom2_name": contact.atom2.name.strip(),
                "residue1": contact.residue1_id,
                "residue2": contact.residue2_id,
                "resname1": contact.atom1.resname,
                "resname2": contact.atom2.resname,
                "chain1": contact.atom1.chain_id,
                "chain2": contact.atom2.chain_id,
                "distance": contact.distance,
            }
            row.update(contact.metadata)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def annotate_complex(self, 
                         attr_prefix: str = "contact_",
                         include_types: bool = True) -> Complex:
        """
        Add contact annotations to residues/atoms in the Complex.
        
        Args:
            attr_prefix: Prefix for attribute names
            include_types: If True, add per-type counts; if False, only total
        """
        if not self.complex:
            raise ValueError("Complex not set in ContactAnalysis")
        
        residue_summary = self.get_residue_summary()
        
        # Annotate residues
        for chain in self.complex.chains.values():
            for residue in chain.iter_residues():
                resid = residue.id_str
                if resid in residue_summary:
                    counts = residue_summary[resid]
                    setattr(residue, f"{attr_prefix}total", counts.get('total', 0))
                    
                    if include_types:
                        for ctype in ContactType:
                            setattr(residue, f"{attr_prefix}{ctype.value}", 
                                   counts.get(ctype.value, 0))
                else:
                    setattr(residue, f"{attr_prefix}total", 0)
                    if include_types:
                        for ctype in ContactType:
                            setattr(residue, f"{attr_prefix}{ctype.value}", 0)
        
        return self.complex


# ==================== Helper Functions ====================

def _distance(a: Atom, b: Atom) -> float:
    """Calculate Euclidean distance between two atoms."""
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _get_atoms(cx: Complex, selection_expr: str, ignore_h: bool = False) -> List[Atom]:
    """Get atoms from selection expression."""
    atoms = select(cx, selection_expr).atoms
    if ignore_h:
        atoms = [a for a in atoms 
                if a.element.upper() != "H" and not a.name.strip().startswith("H")]
    return atoms


def _ring_geometry(residue_atoms: List[Atom], ring_atoms: List[str]) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Calculate ring centroid and normal vector using PCA.
    Returns: ((cx, cy, cz), (nx, ny, nz)) or None if insufficient atoms
    """
    ring_coords = [
        (a.x, a.y, a.z) for a in residue_atoms 
        if a.name.strip().upper() in ring_atoms
    ]
    
    if len(ring_coords) < 3:
        return None
    
    # Calculate centroid
    cx = sum(p[0] for p in ring_coords) / len(ring_coords)
    cy = sum(p[1] for p in ring_coords) / len(ring_coords)
    cz = sum(p[2] for p in ring_coords) / len(ring_coords)
    
    # Calculate normal using SVD (requires numpy)
    try:
        import numpy as np
        P = np.array(ring_coords) - np.array([[cx, cy, cz]])
        _, _, VT = np.linalg.svd(P, full_matrices=False)
        normal = VT[-1]  # Last principal component is the normal
        return (cx, cy, cz), (float(normal[0]), float(normal[1]), float(normal[2]))
    except (ImportError, Exception):
        # Fallback: use simple cross product for planar approximation
        if len(ring_coords) >= 3:
            p1, p2, p3 = ring_coords[0], ring_coords[1], ring_coords[2]
            v1 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
            v2 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
            # Cross product
            nx = v1[1]*v2[2] - v1[2]*v2[1]
            ny = v1[2]*v2[0] - v1[0]*v2[2]
            nz = v1[0]*v2[1] - v1[1]*v2[0]
            norm = math.sqrt(nx*nx + ny*ny + nz*nz)
            if norm > 1e-6:
                return (cx, cy, cz), (nx/norm, ny/norm, nz/norm)
        return None


# ==================== pH-Dependent Protonation Functions ====================

def calculate_fraction_protonated(ph: float, pka: float) -> float:
    """
    Calculate the fraction of a group that is protonated at a given pH.
    Uses the Henderson-Hasselbalch equation.
    
    Args:
        ph: pH value
        pka: pKa of the titratable group
    
    Returns:
        Fraction protonated (0.0 to 1.0)
    
    Notes:
        For acids (e.g., ASP, GLU): fraction_protonated = 1 / (1 + 10^(pH - pKa))
        For bases (e.g., LYS, ARG): fraction_protonated = 1 / (1 + 10^(pKa - pH))
    """ 
    if ph < pka:
        # Below pKa: mostly protonated
        return 1.0 / (1.0 + 10**(ph - pka))
    else:
        # Above pKa: mostly deprotonated
        return 1.0 / (1.0 + 10**(ph - pka))


def get_charge_state(residue_name: str, ph: float, threshold: float = 0.5) -> str:
    """
    Determine the predominant charge state of a residue at a given pH.
    
    Args:
        residue_name: Three-letter residue code (e.g., "ASP", "LYS", "HIS")
        ph: pH value
        threshold: Fraction threshold for considering a group charged (default: 0.5)
    
    Returns:
        "positive", "negative", "neutral", or "both" (for HIS in transition range)
    
    Examples:
        >>> get_charge_state("ASP", 7.4)  # Returns "negative"
        >>> get_charge_state("ASP", 3.0)  # Returns "neutral"
        >>> get_charge_state("LYS", 7.4)  # Returns "positive"
        >>> get_charge_state("HIS", 6.0)  # Returns "both" (near pKa)
    """
    if residue_name not in PKA_VALUES:
        return "neutral"
    
    pka = PKA_VALUES[residue_name]
    
    # Calculate fraction in protonated state
    if residue_name in ["ASP", "GLU"]:
        # Acidic residues: protonated = neutral, deprotonated = negative
        fraction_protonated = calculate_fraction_protonated(ph, pka)
        if fraction_protonated > threshold:
            return "neutral"  # COOH form
        else:
            return "negative"  # COO- form
    
    elif residue_name in ["LYS", "ARG"]:
        # Basic residues: protonated = positive, deprotonated = neutral
        fraction_protonated = calculate_fraction_protonated(ph, pka)
        if fraction_protonated > threshold:
            return "positive"  # NH3+ form
        else:
            return "neutral"  # NH2 form
    
    elif residue_name == "HIS":
        # Histidine is special - can be positive or neutral
        fraction_protonated = calculate_fraction_protonated(ph, pka)
        
        # Near pKa (within 1 pH unit), consider both states
        if abs(ph - pka) < 1.0:
            return "both"  # Significant population of both states
        elif fraction_protonated > threshold:
            return "positive"  # Protonated HIS+
        else:
            return "neutral"  # Neutral HIS
    
    else:
        # CYS, TYR - usually neutral at physiological pH
        return "neutral"


def get_charged_atoms_at_ph(atoms: List[Atom], ph: float) -> Tuple[List[Atom], List[Atom]]:
    """
    Get lists of positively and negatively charged atoms at a given pH.
    
    Args:
        atoms: List of Atom objects to classify
        ph: pH value for protonation state calculation
    
    Returns:
        Tuple of (positive_atoms, negative_atoms)
    
    Notes:
        - Uses Henderson-Hasselbalch equation to determine predominant charge state
        - HIS can contribute to both lists if pH is near its pKa (~6.0)
    """
    positive = []
    negative = []
    
    for atom in atoms:
        resname = atom.resname
        atom_name = atom.name.strip().upper()
        
        # Get charge state at this pH
        charge_state = get_charge_state(resname, ph)
        
        # Check if this atom is a charged atom for this residue
        if charge_state == "positive":
            if resname == "LYS" and atom_name == "NZ":
                positive.append(atom)
            elif resname == "ARG" and atom_name in ["NH1", "NH2", "NE"]:
                positive.append(atom)
            elif resname == "HIS" and atom_name in ["ND1", "NE2"]:
                positive.append(atom)
        
        elif charge_state == "negative":
            if resname == "ASP" and atom_name in ["OD1", "OD2"]:
                negative.append(atom)
            elif resname == "GLU" and atom_name in ["OE1", "OE2"]:
                negative.append(atom)
        
        elif charge_state == "both":
            # HIS near its pKa - include in both lists with reduced weight
            # (we'll detect salt bridges but note the pH-dependent nature)
            if resname == "HIS" and atom_name in ["ND1", "NE2"]:
                positive.append(atom)
    
    return positive, negative


# ==================== Contact Detection Functions ====================

def find_hydrogen_bonds(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    cutoff: float = HBOND_CUTOFF,
    ignore_h: bool = True,
    params: Optional[ContactParameters] = None
) -> List[Contact]:
    """
    Find hydrogen bonds between two selections.
    Uses distance-only heuristic (heavy atom approach).
    
    Args:
        cx: Complex structure
        selection_A: First selection expression
        selection_B: Second selection expression
        cutoff: Maximum H-bond distance (Å). Overridden by params if provided.
        ignore_h: Ignore hydrogen atoms in calculation
        params: ContactParameters object. If provided, uses params.hbond_cutoff
    
    Returns:
        List of Contact objects
    """
    # Use params if provided
    if params is not None:
        cutoff = params.hbond_cutoff
    
    atoms_A = _get_atoms(cx, selection_A, ignore_h=ignore_h)
    atoms_B = _get_atoms(cx, selection_B, ignore_h=ignore_h)
    
    # Filter for potential H-bond atoms
    hbond_A = [a for a in atoms_A if a.element.upper() in HBOND_ELEMENTS]
    hbond_B = [a for a in atoms_B if a.element.upper() in HBOND_ELEMENTS]
    
    contacts = []
    for a in hbond_A:
        for b in hbond_B:
            dist = _distance(a, b)
            if dist <= cutoff:
                contacts.append(Contact(
                    type=ContactType.HYDROGEN_BOND,
                    atom1=a,
                    atom2=b,
                    distance=dist
                ))
    
    return contacts


def find_salt_bridges(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    cutoff: float = SALT_BRIDGE_CUTOFF,
    cross_chain_only: bool = True,
    params: Optional[ContactParameters] = None,
    ph: Optional[float] = None
) -> List[Contact]:
    """
    Find salt bridges between charged residues with pH-dependent charge states.
    
    Args:
        cx: Complex structure
        selection_A: First selection expression
        selection_B: Second selection expression
        cutoff: Maximum salt bridge distance (Å). Overridden by params if provided.
        cross_chain_only: Only consider inter-chain salt bridges
        params: ContactParameters object. If provided, uses params.salt_bridge_cutoff and params.ph
        ph: pH value for protonation state. Overridden by params if provided. Default: 7.4
    
    Returns:
        List of Contact objects with pH-dependent charge states
    
    Notes:
        - At pH 7.4: LYS, ARG are positive; ASP, GLU are negative
        - At pH < 4: ASP, GLU become neutral (protonated)
        - At pH > 10: LYS becomes less charged
        - HIS (pKa ~6.0) can be positive or neutral depending on pH
    """
    # Use params if provided
    if params is not None:
        cutoff = params.salt_bridge_cutoff
        ph = params.ph
    elif ph is None:
        ph = 7.4  # Default physiological pH
    
    atoms_A = _get_atoms(cx, selection_A, ignore_h=True)
    atoms_B = _get_atoms(cx, selection_B, ignore_h=True)
    
    # Get pH-dependent charged atoms
    all_atoms = atoms_A + atoms_B
    positive, negative = get_charged_atoms_at_ph(all_atoms, ph)
    
    contacts = []
    for pos in positive:
        for neg in negative:
            # Skip if same chain and cross_chain_only is True
            if cross_chain_only and pos.chain_id == neg.chain_id:
                continue
            
            dist = _distance(pos, neg)
            if dist <= cutoff:
                # Determine if this is a pH-sensitive interaction
                pos_state = get_charge_state(pos.resname, ph)
                neg_state = get_charge_state(neg.resname, ph)
                ph_sensitive = (pos_state == "both" or neg_state == "both")
                
                contacts.append(Contact(
                    type=ContactType.SALT_BRIDGE,
                    atom1=pos,
                    atom2=neg,
                    distance=dist,
                    metadata={
                        "charge_pair": f"{pos.resname}-{neg.resname}",
                        "ph": ph,
                        "ph_sensitive": ph_sensitive,
                        "pos_charge_state": pos_state,
                        "neg_charge_state": neg_state,
                    }
                ))
    
    return contacts


def find_disulfide_bonds(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    cutoff: float = DISULFIDE_CUTOFF,
    params: Optional[ContactParameters] = None
) -> List[Contact]:
    """
    Find disulfide bonds between cysteine residues.
    
    Args:
        cx: Complex structure
        selection_A: First selection expression
        selection_B: Second selection expression
        cutoff: Maximum disulfide bond distance (Å). Overridden by params if provided.
        params: ContactParameters object. If provided, uses params.disulfide_cutoff
    
    Returns:
        List of Contact objects
    """
    # Use params if provided
    if params is not None:
        cutoff = params.disulfide_cutoff
    
    atoms_A = _get_atoms(cx, selection_A, ignore_h=True)
    atoms_B = _get_atoms(cx, selection_B, ignore_h=True)
    
    # Get sulfur atoms from cysteines
    sg_A = [a for a in atoms_A if a.resname == "CYS" and a.name.strip().upper() == "SG"]
    sg_B = [a for a in atoms_B if a.resname == "CYS" and a.name.strip().upper() == "SG"]
    
    contacts = []
    for a in sg_A:
        for b in sg_B:
            dist = _distance(a, b)
            if dist <= cutoff:
                contacts.append(Contact(
                    type=ContactType.DISULFIDE,
                    atom1=a,
                    atom2=b,
                    distance=dist
                ))
    
    return contacts


def find_hydrophobic_contacts(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    cutoff: float = HYDROPHOBIC_CUTOFF,
    params: Optional[ContactParameters] = None
) -> List[Contact]:
    """
    Find hydrophobic contacts between nonpolar residues.
    Considers carbon-carbon contacts within cutoff.
    
    Args:
        cx: Complex structure
        selection_A: First selection expression
        selection_B: Second selection expression
        cutoff: Maximum hydrophobic contact distance (Å). Overridden by params if provided.
        params: ContactParameters object. If provided, uses params.hydrophobic_cutoff
    
    Returns:
        List of Contact objects
    """
    # Use params if provided
    if params is not None:
        cutoff = params.hydrophobic_cutoff
    
    atoms_A = _get_atoms(cx, selection_A, ignore_h=True)
    atoms_B = _get_atoms(cx, selection_B, ignore_h=True)
    
    # Filter for carbon atoms in hydrophobic residues
    hydro_A = [
        a for a in atoms_A 
        if a.resname in HYDROPHOBIC_RES and a.element.upper() == "C"
    ]
    hydro_B = [
        a for a in atoms_B 
        if a.resname in HYDROPHOBIC_RES and a.element.upper() == "C"
    ]
    
    contacts = []
    for a in hydro_A:
        for b in hydro_B:
            dist = _distance(a, b)
            if dist <= cutoff:
                contacts.append(Contact(
                    type=ContactType.HYDROPHOBIC,
                    atom1=a,
                    atom2=b,
                    distance=dist
                ))
    
    return contacts


def find_pi_stacking(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    dist_range: Tuple[float, float] = PI_STACK_DIST_RANGE,
    angle_parallel_tol: float = PI_PARALLEL_ANGLE_TOL,
    angle_T_tol: float = PI_T_ANGLE_TOL,
    params: Optional[ContactParameters] = None
) -> List[Contact]:
    """
    Find π-π stacking interactions between aromatic residues.
    
    Args:
        cx: Complex structure
        selection_A: First selection expression
        selection_B: Second selection expression
        dist_range: (min, max) centroid distance range (Å). Overridden by params if provided.
        angle_parallel_tol: Angle tolerance for parallel stacking (degrees). Overridden by params if provided.
        angle_T_tol: Angle tolerance for T-shaped stacking (degrees). Overridden by params if provided.
        params: ContactParameters object. If provided, uses params.pi_stack_* values
    
    Returns:
        List of Contact objects (using ring centroid atoms)
    """
    # Use params if provided
    if params is not None:
        dist_range = params.pi_stack_dist_range
        angle_parallel_tol = params.pi_parallel_angle_tol
        angle_T_tol = params.pi_t_angle_tol
    
    res_A = selection_residues(cx, select(cx, selection_A))
    res_B = selection_residues(cx, select(cx, selection_B))
    
    # Filter for aromatic residues
    arom_A = [r for r in res_A if r.resname in AROMATIC_RINGS]
    arom_B = [r for r in res_B if r.resname in AROMATIC_RINGS]
    
    contacts = []
    dmin, dmax = dist_range
    
    for res_a in arom_A:
        for res_b in arom_B:
            # Check all ring combinations
            for ring_def_A in AROMATIC_RINGS[res_a.resname]:
                for ring_def_B in AROMATIC_RINGS[res_b.resname]:
                    geom_A = _ring_geometry(res_a.atoms, ring_def_A)
                    geom_B = _ring_geometry(res_b.atoms, ring_def_B)
                    
                    if not geom_A or not geom_B:
                        continue
                    
                    cent_A, norm_A = geom_A
                    cent_B, norm_B = geom_B
                    
                    # Calculate centroid distance
                    dx = cent_A[0] - cent_B[0]
                    dy = cent_A[1] - cent_B[1]
                    dz = cent_A[2] - cent_B[2]
                    cent_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if cent_dist < dmin or cent_dist > dmax:
                        continue
                    
                    # Calculate angle between normals
                    dot = norm_A[0]*norm_B[0] + norm_A[1]*norm_B[1] + norm_A[2]*norm_B[2]
                    mag_A = math.sqrt(norm_A[0]**2 + norm_A[1]**2 + norm_A[2]**2)
                    mag_B = math.sqrt(norm_B[0]**2 + norm_B[1]**2 + norm_B[2]**2)
                    
                    if mag_A < 1e-6 or mag_B < 1e-6:
                        continue
                    
                    cos_angle = max(-1.0, min(1.0, dot / (mag_A * mag_B)))
                    angle = math.degrees(math.acos(cos_angle))
                    
                    # Classify stacking type
                    stack_type = None
                    if angle <= angle_parallel_tol or abs(180.0 - angle) <= angle_parallel_tol:
                        stack_type = "parallel"
                    elif abs(90.0 - angle) <= angle_T_tol:
                        stack_type = "T-shaped"
                    
                    if stack_type:
                        # Use first atom from each ring as representative
                        atom_a = next((a for a in res_a.atoms if a.name.strip().upper() in ring_def_A), res_a.atoms[0])
                        atom_b = next((a for a in res_b.atoms if a.name.strip().upper() in ring_def_B), res_b.atoms[0])
                        
                        contacts.append(Contact(
                            type=ContactType.PI_STACKING,
                            atom1=atom_a,
                            atom2=atom_b,
                            distance=cent_dist,
                            metadata={
                                "stack_type": stack_type,
                                "angle": angle,
                                "centroid_A": cent_A,
                                "centroid_B": cent_B
                            }
                        ))
    
    return contacts


# ==================== Main Analysis Function ====================

def analyze_contacts(
    cx: Complex,
    selection_A: str,
    selection_B: str,
    include_hbonds: bool = True,
    include_salt_bridges: bool = True,
    include_disulfides: bool = True,
    include_hydrophobic: bool = True,
    include_pi_stacking: bool = True,
    params: Optional[ContactParameters] = None,
    **kwargs
) -> ContactAnalysis:
    """
    Comprehensive contact analysis between two selections.
    
    Args:
        cx: Complex structure
        selection_A: First selection expression (e.g., "H,L" for antibody)
        selection_B: Second selection expression (e.g., "A" for antigen)
        include_hbonds: Find hydrogen bonds
        include_salt_bridges: Find salt bridges
        include_disulfides: Find disulfide bonds
        include_hydrophobic: Find hydrophobic contacts
        include_pi_stacking: Find π-π stacking
        params: ContactParameters object with cutoff values. If provided, overrides individual cutoffs in kwargs.
        **kwargs: Additional parameters for specific contact types (e.g., cutoff, ignore_h, cross_chain_only)
    
    Returns:
        ContactAnalysis object with all detected contacts
    
    Examples:
        >>> # Use default parameters
        >>> analysis = analyze_contacts(cx, "H,L", "A")
        
        >>> # Use custom parameters
        >>> params = setup_contact_parameters(hbond_cutoff=3.0, salt_bridge_cutoff=4.5)
        >>> analysis = analyze_contacts(cx, "H,L", "A", params=params)
        
        >>> # Pass params to kwargs for backward compatibility
        >>> analysis = analyze_contacts(cx, "H,L", "A", cutoff=3.0)  # Still works
    """
    all_contacts = []
    
    # If params is provided, add it to kwargs for each function call
    if params is not None:
        kwargs['params'] = params
    
    if include_hbonds:
        hbonds = find_hydrogen_bonds(cx, selection_A, selection_B, **kwargs)
        all_contacts.extend(hbonds)
    
    if include_salt_bridges:
        salt_bridges_list = find_salt_bridges(cx, selection_A, selection_B, **kwargs)
        all_contacts.extend(salt_bridges_list)
    
    if include_disulfides:
        disulfides = find_disulfide_bonds(cx, selection_A, selection_B, **kwargs)
        all_contacts.extend(disulfides)
    
    if include_hydrophobic:
        hydrophobic = find_hydrophobic_contacts(cx, selection_A, selection_B, **kwargs)
        all_contacts.extend(hydrophobic)
    
    if include_pi_stacking:
        pi_stacks = find_pi_stacking(cx, selection_A, selection_B, **kwargs)
        all_contacts.extend(pi_stacks)
    
    return ContactAnalysis(contacts=all_contacts, complex=cx)


# ==================== Example Usage ====================

def example_usage():
    from .pbdparser import parse_pdb
    
    # Load structure
    cx = parse_pdb("complex.pdb")
    
    # Example 1: Use default parameters
    print("=== Example 1: Default Parameters ===")
    analysis = analyze_contacts(
        cx,
        selection_A="H,L",  # Antibody chains
        selection_B="A",     # Antigen chain
    )
    
    # Example 2: Use custom parameters
    print("\n=== Example 2: Custom Parameters ===")
    params = setup_contact_parameters(
        hbond_cutoff=3.0,      # Stricter H-bond cutoff
        salt_bridge_cutoff=4.5,  # Relaxed salt bridge cutoff
        pi_stack_dist_max=8.0   # Allow wider π-stacking range
    )
    print(params)
    
    analysis_custom = analyze_contacts(
        cx,
        selection_A="H,L",
        selection_B="A",
        params=params
    )
    
    # Example 3: Create parameters from dictionary (for webapp storage)
    print("\n=== Example 3: Parameters from Dictionary ===")
    param_dict = {
        'hbond_cutoff': 3.2,
        'salt_bridge_cutoff': 4.2,
        'disulfide_cutoff': 2.5,
        'hydrophobic_cutoff': 5.0,
        'pi_stack_dist_min': 4.0,
        'pi_stack_dist_max': 7.5,
        'pi_parallel_angle_tol': 25.0,
        'pi_t_angle_tol': 25.0,
    }
    params_from_dict = ContactParameters.from_dict(param_dict)
    analysis_dict = analyze_contacts(cx, "H,L", "A", params=params_from_dict)
    
    # Get summary statistics
    counts = analysis.get_contact_counts()
    print(f"Total contacts: {sum(counts.values())}")
    for ctype, count in counts.items():
        print(f"  {ctype.value}: {count}")
    
    # Get contacts by type
    hbonds = analysis.filter_by_type(ContactType.HYDROGEN_BOND)
    print(f"\nFound {len(hbonds)} hydrogen bonds")
    
    # Per-residue summary
    residue_summary = analysis.get_residue_summary()
    for resid, counts in sorted(residue_summary.items(), 
                                 key=lambda x: x[1]['total'], 
                                 reverse=True)[:10]:
        print(f"{resid}: {counts['total']} total contacts")
    
    # Get partners for a specific residue
    partners = analysis.get_residue_partners("H:32")
    for partner, interactions in partners.items():
        print(f"  {partner}: {len(interactions)} interactions")
    
    # Export to DataFrame
    df = analysis.to_dataframe()
    print(df[df['contact_type'] == 'hydrogen_bond'])
    
    # Annotate the complex
    analysis.annotate_complex(attr_prefix="contact_")
    # Now residues have: residue.contact_total, residue.contact_hydrogen_bond, etc.
    
    return analysis

