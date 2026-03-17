"""
Comprehensive Residue-Level Feature Extraction

This module combines all available analyses from the package to create
a unified residue-level feature representation. Each residue is characterized by:
- Basic properties (name, position, chain)
- Structural geometry (angles, accessibility)
- Distance metrics (to other selections)
- Interface participation
- Molecular interactions (H-bonds, salt bridges, etc.)
- Surface area and burial
- Van der Waals contacts

This creates a complete "fingerprint" for each residue suitable for:
- Comparative analysis across structures
- Machine learning feature matrices
- Hotspot identification
- Mutation effect prediction
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import csv
import json
import math 

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .models import Complex, Residue, Atom
from .pdbparser import parse_pdb
from .selection import select, selection_residues
from .distances import residue_distance_matrix, pairwise_min_distance
from .geometry import analyze_geometry, GeometryAnalysis
from .interface import compute_interface, InterfaceAnalysis
from .contacts import analyze_contacts, ContactParameters, ContactType, ContactAnalysis
from .sasa_analysis import compare_bound_unbound_sasa

# Optional SASA import
try:
    from .sasa import _sasa_fs
    _HAS_FREESASA_WRAPPER = True
except (ImportError, AttributeError):
    _HAS_FREESASA_WRAPPER = False

# Optional VdW import
try:
    from .vdw import per_residue_LJ_decomposition
    _HAS_VDW = True
except (ImportError, AttributeError):
    _HAS_VDW = False


# ==================== Feature Data Structures ====================

@dataclass
class ResidueFeatures:
    """
    Comprehensive feature representation for a single residue.
    All features are optional to allow partial analysis.
    """
    # ===== Basic Identification =====
    chain_id: str
    resname: str
    resseq: int
    icode: str = ""
    residue_id: str = ""  # Formatted string like "TYR H:32"
    
    # ===== Geometry Features =====
    phi: Optional[float] = None  # Backbone dihedral
    psi: Optional[float] = None  # Backbone dihedral
    omega: Optional[float] = None  # Backbone dihedral
    bend_angle: Optional[float] = None # Bend angle at this center
    chi1: Optional[float] = None  # Side-chain rotamer
    chi2: Optional[float] = None
    chi3: Optional[float] = None
    chi4: Optional[float] = None
    
    # ===== Surface Accessibility =====
    unbound_sasa: Optional[float] = None  # Solvent accessible surface area (Ų) unbound state
    bound_sasa: Optional[float] = None  # SASA in complex 
    delta_sasa: Optional[float] = None  # change in SASA after binding  
    buried_fraction: Optional[float] = None  # fraction of area buried in complex
    
    # ===== Distance Features =====
    # Distances to different selections (customizable)
    min_dist_to_partner: Optional[float] = None  # Minimum distance to partner selection
    ca_dist_to_partner: Optional[float] = None  # CA-CA distance to nearest partner residue
    # Distance statistics within same chain
    avg_dist_to_neighbors: Optional[float] = None  # Average to ±5 sequence neighbors
    
    # ===== Interface Features =====
    is_interface: bool = False  # Within cutoff of partner
    interface_contact_count: int = 0  # Number of atom-atom contacts
    num_partner_residues: int = 0  # Number of contacting partner residues
    partner_residue_ids: List[str] = field(default_factory=list)  # IDs of partners
    
    # ===== Molecular Interactions =====
    # Count by interaction type
    num_hbonds: int = 0
    num_salt_bridges: int = 0
    num_hydrophobic: int = 0
    num_pi_stacking: int = 0
    num_disulfides: int = 0
    
    # Total interaction counts
    total_interactions: int = 0
    interaction_types: List[str] = field(default_factory=list)
    
    # Detailed interaction partners (residue IDs)
    hbond_partners: List[str] = field(default_factory=list)
    salt_bridge_partners: List[str] = field(default_factory=list)
    hydrophobic_partners: List[str] = field(default_factory=list)
    pi_stacking_partners: List[str] = field(default_factory=list)
    
    # ===== VDW LJ engergies =====
    vdw_energy: Optional[float] = None
    
    # ===== Chemical Properties =====
    is_charged: bool = False  # LYS, ARG, ASP, GLU
    is_polar: bool = False  # SER, THR, ASN, GLN, TYR, CYS
    is_hydrophobic: bool = False  # ALA, VAL, ILE, LEU, MET, PHE, TRP, PRO
    is_aromatic: bool = False  # PHE, TYR, TRP, HIS
    is_positive: bool = False  # LYS, ARG
    is_negative: bool = False  # ASP, GLU
    
    # ===== Structural Context =====
    secondary_structure: Optional[str] = None  # "helix", "sheet", "loop" (if computed)
    
    # ===== Atom Counts =====
    num_heavy_atoms: int = 0
    num_all_atoms: int = 0
    
    # ===== B-factors (dynamics/uncertainty) =====
    avg_bfactor: Optional[float] = None
    max_bfactor: Optional[float] = None
    min_bfactor: Optional[float] = None
    
    # ===== Custom/Additional Features =====
    custom_features: Dict[str, Union[float, int, str, bool]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding empty lists and None values by default."""
        data = asdict(self)
        # Convert lists to comma-separated strings for easier export
        for key in ['partner_residue_ids', 'interaction_types', 'hbond_partners', 
                    'salt_bridge_partners', 'hydrophobic_partners', 
                    'pi_stacking_partners']:
            if key in data and data[key]:
                data[key] = ','.join(data[key])
            elif key in data:
                data[key] = ''
        return data
    
    @property
    def is_hotspot_candidate(self) -> bool:
        """
        Heuristic for identifying potential hotspot residues.
        Hotspots typically have: interface participation + multiple interactions.
        """
        return (self.is_interface and 
                self.total_interactions >= 3 and
                (self.num_hbonds >= 1 or self.num_salt_bridges >= 1))


# ==================== Residue Classification ====================

# Standard residue type classifications
CHARGED_RES = {"LYS", "ARG", "ASP", "GLU", "HIS"}
POSITIVE_RES = {"LYS", "ARG"}
NEGATIVE_RES = {"ASP", "GLU"}
POLAR_RES = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
HYDROPHOBIC_RES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}

# Maximum SASA values by residue type (Ų) - from Miller et al. 1987
MAX_SASA = {
    "ALA": 129.0, "CYS": 167.0, "ASP": 193.0, "GLU": 223.0, "PHE": 240.0,
    "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LYS": 236.0, "LEU": 201.0,
    "MET": 224.0, "ASN": 195.0, "PRO": 159.0, "GLN": 225.0, "ARG": 274.0,
    "SER": 155.0, "THR": 172.0, "VAL": 174.0, "TRP": 285.0, "TYR": 263.0,
}


def classify_residue_chemistry(resname: str) -> Dict[str, bool]:
    """Return chemical classification flags for a residue."""
    resname = resname.upper()
    return {
        'is_charged': resname in CHARGED_RES,
        'is_positive': resname in POSITIVE_RES,
        'is_negative': resname in NEGATIVE_RES,
        'is_polar': resname in POLAR_RES,
        'is_hydrophobic': resname in HYDROPHOBIC_RES,
        'is_aromatic': resname in AROMATIC_RES,
    }


# ==================== Main Feature Extraction ====================

def extract_residue_features(
    cx: Complex,
    selection_A: Union[str, List[str]],
    selection_B: Optional[Union[str, List[str]]] = None,
    interface_cutoff: float = 5.0,
    contact_params: Optional[ContactParameters] = None,
    compute_sasa: bool = True,
    compute_geometry: bool = True,
    compute_vdw: bool = True,
    verbose: bool = True,
) -> Dict[str, ResidueFeatures]:
    """
    Extract comprehensive residue-level features from a protein complex.
    
    This is the main entry point that orchestrates all analyses and combines
    them into a unified feature representation for each residue.
    
    Args:
        cx: Complex structure
        selection_A: First selection (e.g., ["H", "L"] for antibody)
        selection_B: Second selection (e.g., ["A"] for antigen). 
                     If None, analyzes selection_A in isolation.
        interface_cutoff: Distance cutoff for interface definition (Å)
        contact_params: Contact detection parameters (uses defaults if None)
        compute_sasa: Calculate solvent accessible surface area
        use_freesasa: Use FreeSASA if available (faster, more accurate)
        compute_geometry: Calculate backbone/sidechain angles
        compute_vdw: Include Van der Waals contact analysis
        verbose: Print progress messages
    
    Returns:
        Dictionary mapping residue_id -> ResidueFeatures
        residue_id format: "TYR H:32" (resname chain:resseq+icode)
    
    """
    
    if verbose:
        print("Starting comprehensive residue feature extraction...")
    
    # Initialize feature storage
    features: Dict[str, ResidueFeatures] = {}
    
    # Normalize selections
    if isinstance(selection_A, str):
        selection_A = [selection_A] if ',' not in selection_A else selection_A.split(',')
    chains_A = [s.strip() for s in selection_A]
    
    if selection_B is not None:
        if isinstance(selection_B, str):
            selection_B = [selection_B] if ',' not in selection_B else selection_B.split(',')
        chains_B = [s.strip() for s in selection_B]
    else:
        chains_B = None
    
    all_chains = [c for c in cx.chains.keys()] 
    
    # Get all residues in the complex
    all_residues: List[Residue] = []
    for chain_id, chain in cx.chains.items():
        all_residues.extend(chain.iter_residues())
    
    if verbose:
        print(f"  Total residues in complex: {len(all_residues)}")
    
    # ===== STEP 1: Initialize basic features =====
    if verbose:
        print("  [1/7] Initializing basic features...")
    
    for residue in all_residues:
        res_id = residue.id_str
        
        # Basic identification
        feat = ResidueFeatures(
            chain_id=residue.chain_id,
            resname=residue.resname,
            resseq=residue.resseq,
            icode=residue.icode,
            residue_id=res_id
        )
        
        # Chemical classification
        chem_props = classify_residue_chemistry(residue.resname)
        feat.is_charged = chem_props['is_charged']
        feat.is_positive = chem_props['is_positive']
        feat.is_negative = chem_props['is_negative']
        feat.is_polar = chem_props['is_polar']
        feat.is_hydrophobic = chem_props['is_hydrophobic']
        feat.is_aromatic = chem_props['is_aromatic']
        
        # Atom counts
        atoms = list(residue.iter_atoms(ignore_h=True))
        feat.num_heavy_atoms = len(atoms)
        feat.num_all_atoms = len(residue.atoms)
        
        # B-factors
        if atoms:
            bfactors = [a.bfactor for a in atoms]
            feat.avg_bfactor = sum(bfactors) / len(bfactors)
            feat.max_bfactor = max(bfactors)
            feat.min_bfactor = min(bfactors)
        
        features[res_id] = feat
    
    # ===== STEP 2: Geometry analysis =====
    if compute_geometry and verbose:
        print("  [2/7] Computing backbone and sidechain angles...")
    
    if compute_geometry:
        try:
            # Use analyze_geometry which returns GeometryAnalysis object
            geom_analysis = analyze_geometry(
                cx, 
                all_chains,
                compute_backbone=True,
                compute_sidechains=True,
                compute_bends=True
            )
            
            # Extract backbone angles (phi/psi/omega)
            if geom_analysis.backbone:
                for backbone_angles in geom_analysis.backbone:
                    res_id = backbone_angles.residue_id
                    if res_id in features:
                        features[res_id].phi = backbone_angles.phi
                        features[res_id].psi = backbone_angles.psi
                        features[res_id].omega = backbone_angles.omega 
                        if backbone_angles.phi and backbone_angles.psi:
                            # Alpha helix region: φ ∈ [-90, -30], ψ ∈ [-70, 10]
                            if -90 <= backbone_angles.phi <= -30 and -70 <= backbone_angles.psi <= 10:
                                features[res_id].secondary_structure = "alpha_helix"
                            # Beta sheet region: φ ∈ [-180, -90], ψ ∈ [90, 180]
                            elif -180 <= backbone_angles.phi <= -90 and 90 <= backbone_angles.psi <= 180:
                                features[res_id].secondary_structure = "beta_sheet"
                            # Left-handed helix: φ ∈ [30, 100], ψ ∈ [0, 80]
                            elif 30 <= backbone_angles.phi <= 100 and 0 <= backbone_angles.psi <= 80:
                                features[res_id].secondary_structure = "left_helix"
                            else:
                                features[res_id].secondary_structure = "loop"
                        
                            
            
            # Extract sidechain chi angles
            if geom_analysis.sidechains:
                for sidechain_chi in geom_analysis.sidechains:
                    res_id = sidechain_chi.residue_id
                    if res_id in features:
                        chis = sidechain_chi.chis
                        features[res_id].chi1 = chis.get('chi1')
                        features[res_id].chi2 = chis.get('chi2')
                        features[res_id].chi3 = chis.get('chi3')
                        features[res_id].chi4 = chis.get('chi4')
                        
            # Extract bend angles
            if geom_analysis.bends:
                for bent in geom_analysis.bends:
                    res_id = bent.residue_id  
                    if res_id in features:
                        features[res_id].bend_angle = bent.angle_deg
                        
        except Exception as e:
            if verbose:
                print(f"    Warning: Geometry calculation encountered error: {e}")
    
    # ===== STEP 3: SASA calculation =====
    if compute_sasa and verbose:
        print("  [3/7] Computing solvent accessible surface area...")
    
    if compute_sasa:
        try:
            sasa_results = compare_bound_unbound_sasa(cx, all_chains) 
            
            for res_id, values in sasa_results.items():
                features[res_id].unbound_sasa = values['unbound'] 
                features[res_id].bound_sasa = values['bound'] 
                features[res_id].delta_sasa = values['delta'] 
                features[res_id].buried_fraction = values['buried_fraction'] 
              
        except Exception as e:
            if verbose:
                print(f"    Warning: SASA calculation failed: {e}")
    
    # ===== STEP 4: Distance features =====
    if chains_B is not None and verbose:
        print("  [4/7] Computing distance features...")
    
    if chains_B is not None:
        def atom_distance(a, b):
            dx = a.x - b.x
            dy = a.y - b.y
            dz = a.z - b.z
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        try:
            # Get residues for each selection
            resA = selection_residues(cx, select(cx, chains_A))
            resB = selection_residues(cx, select(cx, chains_B))
            
            # Compute minimum distances for res in A -> B
            for i, res_a in enumerate(resA):
                res_a_id = res_a.id_str
                coords_a = [(a.x, a.y, a.z) for a in res_a.iter_atoms(ignore_h=True)]
                
                if not coords_a:
                    continue
                
                min_dist = float('inf')
                nearest_res_b = None
                ca_a = next((a for a in res_a.atoms if a.name.strip().upper() == "CA"), None) 
                
                for res_b in resB:
                    coords_b = [(a.x, a.y, a.z) for a in res_b.iter_atoms(ignore_h=True)]
                    if coords_b:
                        d = pairwise_min_distance(coords_a, coords_b)
                        if d < min_dist:
                            min_dist = d
                            nearest_res_b = res_b 
                
                if min_dist != float('inf'):
                    features[res_a_id].min_dist_to_partner = min_dist
                    if nearest_res_b:
                        ca_b = next((a for a in nearest_res_b.atoms if a.name.strip().upper() == "CA"), None)
                        if ca_a and ca_b:
                            features[res_a_id].ca_dist_to_partner = atom_distance(ca_a, ca_b)  
                            
                if ca_a is None:
                    features[res_a_id].avg_dist_to_neighbors = None
                    continue
                else:
                    neighbor_offsets = (-2, -1, 1, 2)
                    dists = []
                    for off in neighbor_offsets:
                        j = i + off 
                        if j < 0 or j >= len(resA):     # Edge case: out of range
                            continue
                        res_a1 = resA[j]
                        # Optional: avoid crossing chains if resA includes multiple chains
                        if hasattr(res_a, "chain_id") and hasattr(res_a1, "chain_id"):
                            if res_a1.chain_id != res_a.chain_id:
                                continue
                        ca_a1 = next((a for a in res_a1.atoms if a.name.strip().upper() == "CA"), None)
                        if ca_a1 is None:
                            continue
                        dists.append(atom_distance(ca_a, ca_a1))
                    avg_dist = sum(dists) / len(dists) if dists else None
                    features[res_a_id].avg_dist_to_neighbors = avg_dist 
            
            # Symmetric: compute for B -> A
            for i,res_b in enumerate(resB):
                res_b_id = res_b.id_str
                coords_b = [(a.x, a.y, a.z) for a in res_b.iter_atoms(ignore_h=True)]
                
                if not coords_b:
                    continue
                
                min_dist = float('inf')
                nearest_res_a = None
                ca_b = next((a for a in res_b.atoms if a.name.strip().upper() == "CA"), None)
                
                for res_a in resA:
                    coords_a = [(a.x, a.y, a.z) for a in res_a.iter_atoms(ignore_h=True)]
                    if coords_a:
                        d = pairwise_min_distance(coords_b, coords_a)
                        if d < min_dist:
                            min_dist = d
                            nearest_res_a = res_a 
                
                if min_dist != float('inf'):
                    features[res_b_id].min_dist_to_partner = min_dist
                    if nearest_res_a:
                        ca_a = next((a for a in nearest_res_a.atoms if a.name.strip().upper() == "CA"), None)
                        if ca_a and ca_b:
                            features[res_b_id].ca_dist_to_partner = atom_distance(ca_a, ca_b) 
                            
                if ca_b is None:
                    features[res_b_id].avg_dist_to_neighbors = None
                    continue
                else:
                    neighbor_offsets = (-2, -1, 1, 2)
                    dists = []
                    for off in neighbor_offsets:
                        j = i + off 
                        if j < 0 or j >= len(resB):     # Edge case: out of range
                            continue
                        res_b1 = resB[j]
                        # Optional: avoid crossing chains if resB includes multiple chains
                        if hasattr(res_b, "chain_id") and hasattr(res_b1, "chain_id"):
                            if res_b.chain_id != res_b1.chain_id:
                                continue
                        ca_b1 = next((a for a in res_b1.atoms if a.name.strip().upper() == "CA"), None)
                        if ca_b1 is None:
                            continue
                        dists.append(atom_distance(ca_b, ca_b1))
                    avg_dist = sum(dists) / len(dists) if dists else None
                    features[res_b_id].avg_dist_to_neighbors = avg_dist 
        
        except Exception as e:
            if verbose:
                print(f"    Warning: Distance calculation failed: {e}")
    
    
    # ===== STEP 5: Interface analysis =====
    if chains_B is not None and verbose:
        print("  [5/7] Analyzing interface residues...")
    
    if chains_B is not None:
        try:
            interface_result = compute_interface(
                cx, 
                selection_A=chains_A,
                selection_B=chains_B,
                cutoff=interface_cutoff,
                use_freesasa=False,  # Already computed if needed
                ignore_h=True
            )
            
            # Extract interface features
            for chain_id, res_data_list in interface_result.chain_data.items():
                for rd in res_data_list:
                    res_id = rd.residue.id_str
                    if res_id in features:
                        features[res_id].is_interface = rd.is_interface
                        features[res_id].interface_contact_count = rd.contact_count
                        features[res_id].num_partner_residues = len(rd.partner_residues)
                        features[res_id].partner_residue_ids = rd.partner_residues.copy()

        
        except Exception as e:
            if verbose:
                print(f"    Warning: Interface analysis failed: {e}")
    
    # ===== STEP 6: Contact analysis =====
    if chains_B is not None and verbose:
        print("  [6/7] Detecting molecular interactions...")
    
    if chains_B is not None:
        try:
            if contact_params is None:
                contact_params = ContactParameters()
            
            # analyze_contacts returns ContactAnalysis object
            contact_analysis = analyze_contacts(
                cx,
                chains_A,
                chains_B,
                params=contact_params
            )
            
            # Build residue-level contact summaries from ContactAnalysis.contacts
            residue_contacts: Dict[str, Dict[str, List[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            
            # Iterate through Contact objects in contact_analysis.contacts
            for contact in contact_analysis.contacts:
                res1_id = contact.residue1_id
                res2_id = contact.residue2_id
                contact_type = contact.type.value  # Get the string value from enum
                
                residue_contacts[res1_id][contact_type].append(res2_id)
                residue_contacts[res2_id][contact_type].append(res1_id)
            
            # Update features with contact information
            for res_id, contact_dict in residue_contacts.items():
                if res_id not in features:
                    continue
                
                feat = features[res_id]
                
                # Count by type (using ContactType enum values)
                feat.num_hbonds = len(contact_dict.get('hydrogen_bond', []))
                feat.num_salt_bridges = len(contact_dict.get('salt_bridge', []))
                feat.num_hydrophobic = len(contact_dict.get('hydrophobic', []))
                feat.num_pi_stacking = len(contact_dict.get('pi_stacking', []))
                feat.num_disulfides = len(contact_dict.get('disulfide', []))
                
                # Store partners
                feat.hbond_partners = contact_dict.get('hydrogen_bond', [])
                feat.salt_bridge_partners = contact_dict.get('salt_bridge', [])
                feat.hydrophobic_partners = contact_dict.get('hydrophobic', [])
                feat.pi_stacking_partners = contact_dict.get('pi_stacking', [])
                
                # Total counts
                feat.total_interactions = (
                    feat.num_hbonds + 
                    feat.num_salt_bridges + 
                    feat.num_hydrophobic + 
                    feat.num_pi_stacking + 
                    feat.num_disulfides
                )
                
                # List of interaction types present
                if feat.num_hbonds > 0:
                    feat.interaction_types.append('H-bond')
                if feat.num_salt_bridges > 0:
                    feat.interaction_types.append('Salt-bridge')
                if feat.num_hydrophobic > 0:
                    feat.interaction_types.append('Hydrophobic')
                if feat.num_pi_stacking > 0:
                    feat.interaction_types.append('Pi-stacking')
                if feat.num_disulfides > 0:
                    feat.interaction_types.append('Disulfide')
        
        except Exception as e:
            if verbose:
                print(f"    Warning: Contact analysis failed: {e}")
    
    # ===== STEP 7: VdW contact analysis (optional) =====
    if compute_vdw and _HAS_VDW and chains_B is not None and verbose:
        print("  [7/7] Computing Van der Waals contacts...")
    
    if compute_vdw and _HAS_VDW and chains_B is not None:
        try:
            # per_residue_LJ_decomposition returns Dict with residue pair energies
            vdw_result_chainsA = per_residue_LJ_decomposition(
                cx, 
                chains_A, 
                chains_B,
                ignore_h=True
            )
            for res_id, energy in vdw_result_chainsA.items():
                features[res_id].vdw_energy = energy 
            
            
            vdw_result_chainsB = per_residue_LJ_decomposition(
                cx, 
                chains_B, 
                chains_A,
                ignore_h=True
            )
            for res_id, energy in vdw_result_chainsB.items():
                features[res_id].vdw_energy = energy
        
        except Exception as e:
            if verbose:
                print(f"    Warning: VdW analysis failed: {e}")
    
    if verbose:
        print(f"  Extraction complete! Features generated for {len(features)} residues.")
        
        # Summary statistics
        interface_count = sum(1 for f in features.values() if f.is_interface)
        hotspot_count = sum(1 for f in features.values() if f.is_hotspot_candidate)
        print(f"  - Interface residues: {interface_count}")
        print(f"  - Potential hotspots: {hotspot_count}")
    
    return features


# ==================== Export Functions ====================

def features_to_dataframe(features: Dict[str, ResidueFeatures]) -> 'pd.DataFrame':
    """
    Convert feature dictionary to pandas DataFrame for analysis.
    
    Args:
        features: Dictionary from extract_comprehensive_features()
    
    Returns:
        DataFrame with one row per residue, columns for all features
    
    Raises:
        ImportError: If pandas is not installed
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas required for features_to_dataframe()")
    
    rows = []
    for res_id, feat in features.items():
        rows.append(feat.to_dict())
    
    df = pd.DataFrame(rows)
    
    # Sort by chain and position
    df = df.sort_values(['chain_id', 'resseq', 'icode'], ignore_index=True)
    
    return df


def export_features_csv(
    features: Dict[str, ResidueFeatures],
    output_path: str,
    include_empty_columns: bool = False
) -> None:
    """
    Export features to CSV file.
    
    Args:
        features: Dictionary from extract_comprehensive_features()
        output_path: Path to output CSV file
        include_empty_columns: If False, omit columns that are all None/empty
    """
    if not features:
        raise ValueError("No features to export")
    
    # Get all possible columns from first feature
    sample_feat = next(iter(features.values()))
    all_columns = list(sample_feat.to_dict().keys())
    
    # Filter columns if requested
    if not include_empty_columns:
        # Check which columns have at least one non-empty value
        active_columns = []
        for col in all_columns:
            has_value = False
            for feat in features.values():
                val = getattr(feat, col, None)
                if val is not None and val != '' and val != [] and val != {}:
                    has_value = True
                    break
            if has_value:
                active_columns.append(col)
        columns = active_columns
    else:
        columns = all_columns
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for res_id in sorted(features.keys()):
            feat = features[res_id]
            row = feat.to_dict()
            # Filter to active columns
            row = {k: v for k, v in row.items() if k in columns}
            writer.writerow(row)


def export_features_json(
    features: Dict[str, ResidueFeatures],
    output_path: str,
    pretty: bool = True
) -> None:
    """
    Export features to JSON file.
    
    Args:
        features: Dictionary from extract_comprehensive_features()
        output_path: Path to output JSON file
        pretty: If True, format JSON with indentation
    """
    data = {res_id: feat.to_dict() for res_id, feat in features.items()}
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


# ==================== Analysis Utilities ====================

def filter_interface_residues(
    features: Dict[str, ResidueFeatures],
    min_contacts: int = 1
) -> Dict[str, ResidueFeatures]:
    """Filter to only interface residues with minimum contact count."""
    return {
        res_id: feat for res_id, feat in features.items()
        if feat.is_interface and feat.interface_contact_count >= min_contacts
    }


def identify_hotspots(
    features: Dict[str, ResidueFeatures],
    min_interactions: int = 3,
    require_hbond_or_salt: bool = True
) -> Dict[str, ResidueFeatures]:
    """
    Identify potential hotspot residues based on interaction criteria.
    
    Hotspots are interface residues with high interaction counts,
    particularly those forming specific interactions like H-bonds or salt bridges.
    """
    hotspots = {}
    
    for res_id, feat in features.items():
        if not feat.is_interface:
            continue
        
        if feat.total_interactions < min_interactions:
            continue
        
        if require_hbond_or_salt:
            if feat.num_hbonds == 0 and feat.num_salt_bridges == 0:
                continue
        
        hotspots[res_id] = feat
    
    return hotspots


def get_feature_summary(features: Dict[str, ResidueFeatures]) -> Dict:
    """
    Generate summary statistics across all residues.

    Returns dictionary with counts, averages, and distributions.
    """
    summary = {
        'total_residues': len(features),
        'interface_residues': sum(1 for f in features.values() if f.is_interface),
        'buried_residues': sum(1 for f in features.values() if f.buried_fraction is not None and f.buried_fraction > 0.5),
        'charged_residues': sum(1 for f in features.values() if f.is_charged),
        'hydrophobic_residues': sum(1 for f in features.values() if f.is_hydrophobic),
        'aromatic_residues': sum(1 for f in features.values() if f.is_aromatic),
    }

    # Interaction statistics
    all_interactions = [f.total_interactions for f in features.values()]
    if all_interactions:
        summary['avg_interactions_per_residue'] = sum(all_interactions) / len(all_interactions)
        summary['max_interactions'] = max(all_interactions)

    # SASA statistics
    sasa_values = [f.delta_sasa for f in features.values() if f.delta_sasa is not None]
    if sasa_values:
        summary['avg_delta_sasa'] = sum(sasa_values) / len(sasa_values)

    # Contact type distribution
    summary['total_hbonds'] = sum(f.num_hbonds for f in features.values())
    summary['total_salt_bridges'] = sum(f.num_salt_bridges for f in features.values())
    summary['total_hydrophobic'] = sum(f.num_hydrophobic for f in features.values())
    summary['total_pi_stacking'] = sum(f.num_pi_stacking for f in features.values())

    # Hotspot candidates
    summary['hotspot_candidates'] = sum(1 for f in features.values() if f.is_hotspot_candidate)

    return summary


# ==================== Tabular Data Export ====================

_ALL_ANALYSES = ['contacts', 'interface', 'sasa', 'geometry', 'vdw', 'distances']

# Column mapping for combined table: which fields from ResidueFeatures per analysis module
_COMBINED_COLS: Dict[str, List[str]] = {
    'base':      ['residue_id', 'chain_id', 'resname', 'resseq',
                  'is_charged', 'is_polar', 'is_hydrophobic', 'is_aromatic',
                  'num_heavy_atoms', 'avg_bfactor'],
    'contacts':  ['num_hbonds', 'num_salt_bridges', 'num_hydrophobic',
                  'num_pi_stacking', 'num_disulfides', 'total_interactions',
                  'interaction_types', 'hbond_partners', 'salt_bridge_partners',
                  'hydrophobic_partners', 'pi_stacking_partners'],
    'interface': ['is_interface', 'interface_contact_count',
                  'num_partner_residues', 'partner_residue_ids'],
    'sasa':      ['unbound_sasa', 'bound_sasa', 'delta_sasa', 'buried_fraction'],
    'geometry':  ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4',
                  'bend_angle', 'secondary_structure'],
    'vdw':       ['vdw_energy'],
    'distances': ['min_dist_to_partner', 'ca_dist_to_partner',
                  'avg_dist_to_neighbors'],
}

_TABLE_LABELS = {
    'contacts':  'Contacts',
    'interface': 'Interface',
    'sasa':      'SASA',
    'geometry':  'Geometry',
    'vdw':       'VDW Energy',
    'distances': 'Distances',
    'combined':  'Combined',
}


def _fmt(v, decimals: int = 3):
    """Format a single value for table output."""
    if v is None:
        return ''
    if isinstance(v, float):
        return round(v, decimals)
    if isinstance(v, list):
        return ';'.join(str(x) for x in v)
    if isinstance(v, bool):
        return 'Yes' if v else 'No'
    return v


def _parse_res_id(res_id: str):
    """Parse 'TYR H:32' → (resname, chain_id, resseq)."""
    parts = res_id.split()
    resname = parts[0] if parts else ''
    chain_part = parts[1] if len(parts) > 1 else ''
    if ':' in chain_part:
        chain_id, resseq_str = chain_part.split(':', 1)
    else:
        chain_id = chain_part
        resseq_str = ''
    try:
        resseq = int(''.join(c for c in resseq_str if c.isdigit() or c == '-'))
    except (ValueError, TypeError):
        resseq = 0
    return resname, chain_id, resseq


def extract_separate_tables(
    cx: Complex,
    group_a: Union[str, List[str]],
    group_b: Optional[Union[str, List[str]]] = None,
    analyses: Optional[List[str]] = None,
    interface_cutoff: float = 5.0,
    contact_params: Optional[ContactParameters] = None,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Extract each analysis type as its own flat table.

    Args:
        cx: Parsed protein Complex.
        group_a: Chain IDs for group A (e.g. ['H', 'L']).
        group_b: Chain IDs for group B (e.g. ['A']).  Required for contacts,
                 interface, vdw, distances.  SASA and geometry always use
                 all chains from both groups combined.
        analyses: Subset of ['contacts','interface','sasa','geometry','vdw',
                  'distances'].  Defaults to all.
        interface_cutoff: Heavy-atom distance cutoff for interface (Å).
        contact_params: Override default ContactParameters.
        verbose: Print progress messages.

    Returns:
        Dict mapping analysis name → list of row dicts.
        contacts rows are pair-level; all others are per-residue.
    """
    if analyses is None:
        analyses = list(_ALL_ANALYSES)

    if isinstance(group_a, str):
        group_a = [group_a] if ',' not in group_a else group_a.split(',')
    chains_a = [s.strip() for s in group_a]

    if group_b is not None:
        if isinstance(group_b, str):
            group_b = [group_b] if ',' not in group_b else group_b.split(',')
        chains_b = [s.strip() for s in group_b]
    else:
        chains_b = None

    # SASA / geometry use all unique chains from both groups
    combined_chains = list(dict.fromkeys(chains_a + (chains_b or []))) or list(cx.chains.keys())

    tables: Dict[str, List[Dict]] = {}

    # ── CONTACTS ──────────────────────────────────────────────────────────────
    if 'contacts' in analyses and chains_b is not None:
        if verbose:
            print("  [tabular] Contacts...")
        try:
            cp = contact_params if contact_params is not None else ContactParameters()
            ca = analyze_contacts(cx, chains_a, chains_b, params=cp)
            rows = []
            for c in ca.contacts:
                rn1, ch1, rs1 = _parse_res_id(c.residue1_id)
                rn2, ch2, rs2 = _parse_res_id(c.residue2_id)
                rows.append({
                    'residue1_id':   c.residue1_id,
                    'chain1':        ch1,
                    'resname1':      rn1,
                    'resseq1':       rs1,
                    'residue2_id':   c.residue2_id,
                    'chain2':        ch2,
                    'resname2':      rn2,
                    'resseq2':       rs2,
                    'contact_type':  c.type.value,
                    'distance_A':    _fmt(c.distance),
                })
            rows.sort(key=lambda r: (r['chain1'], r['resseq1'], r['contact_type']))
            tables['contacts'] = rows
        except Exception as e:
            if verbose:
                print(f"    Warning: contacts failed: {e}")
            tables['contacts'] = []

    # ── INTERFACE ─────────────────────────────────────────────────────────────
    if 'interface' in analyses and chains_b is not None:
        if verbose:
            print("  [tabular] Interface...")
        try:
            ia = compute_interface(cx, chains_a, chains_b,
                                   cutoff=interface_cutoff,
                                   use_freesasa=False, ignore_h=True)
            rows = []
            for chain_id, res_list in ia.chain_data.items():
                for rd in res_list:
                    res = rd.residue
                    rows.append({
                        'residue_id':        res.id_str,
                        'chain_id':          res.chain_id,
                        'resname':           res.resname,
                        'resseq':            res.resseq,
                        'is_interface':      'Yes' if rd.is_interface else 'No',
                        'contact_count':     rd.contact_count,
                        'num_partners':      len(rd.partner_residues),
                        'partner_residue_ids': ';'.join(rd.partner_residues),
                    })
            rows.sort(key=lambda r: (r['chain_id'], r['resseq']))
            tables['interface'] = rows
        except Exception as e:
            if verbose:
                print(f"    Warning: interface failed: {e}")
            tables['interface'] = []

    # ── SASA ──────────────────────────────────────────────────────────────────
    if 'sasa' in analyses:
        if verbose:
            print("  [tabular] SASA...")
        try:
            sasa_results = compare_bound_unbound_sasa(cx, combined_chains)
            rows = []
            for res_id, vals in sasa_results.items():
                resname, chain_id, resseq = _parse_res_id(res_id)
                rows.append({
                    'residue_id':      res_id,
                    'chain_id':        chain_id,
                    'resname':         resname,
                    'resseq':          resseq,
                    'unbound_sasa_A2': _fmt(vals.get('unbound')),
                    'bound_sasa_A2':   _fmt(vals.get('bound')),
                    'delta_sasa_A2':   _fmt(vals.get('delta')),
                    'buried_fraction': _fmt(vals.get('buried_fraction')),
                    'is_interface':    'Yes' if vals.get('is_interface') else 'No',
                })
            rows.sort(key=lambda r: (r['chain_id'], r['resseq']))
            tables['sasa'] = rows
        except Exception as e:
            if verbose:
                print(f"    Warning: SASA failed: {e}")
            tables['sasa'] = []

    # ── GEOMETRY ──────────────────────────────────────────────────────────────
    if 'geometry' in analyses:
        if verbose:
            print("  [tabular] Geometry...")
        try:
            geom = analyze_geometry(cx, combined_chains,
                                    compute_backbone=True,
                                    compute_sidechains=True,
                                    compute_bends=True)
            res_data: Dict[str, Dict] = {}

            if geom.backbone:
                for ba in geom.backbone:
                    res_data.setdefault(ba.residue_id, {})
                    res_data[ba.residue_id].update({
                        'phi':   _fmt(ba.phi),
                        'psi':   _fmt(ba.psi),
                        'omega': _fmt(ba.omega),
                    })
            if geom.sidechains:
                for sc in geom.sidechains:
                    res_data.setdefault(sc.residue_id, {})
                    chis = sc.chis
                    res_data[sc.residue_id].update({
                        'chi1': _fmt(chis.get('chi1')),
                        'chi2': _fmt(chis.get('chi2')),
                        'chi3': _fmt(chis.get('chi3')),
                        'chi4': _fmt(chis.get('chi4')),
                    })
            if geom.bends:
                for b in geom.bends:
                    res_data.setdefault(b.residue_id, {})
                    res_data[b.residue_id]['bend_angle'] = _fmt(b.angle_deg)

            rows = []
            for res_id, d in res_data.items():
                resname, chain_id, resseq = _parse_res_id(res_id)
                rows.append({
                    'residue_id': res_id,
                    'chain_id':   chain_id,
                    'resname':    resname,
                    'resseq':     resseq,
                    'phi':        d.get('phi', ''),
                    'psi':        d.get('psi', ''),
                    'omega':      d.get('omega', ''),
                    'chi1':       d.get('chi1', ''),
                    'chi2':       d.get('chi2', ''),
                    'chi3':       d.get('chi3', ''),
                    'chi4':       d.get('chi4', ''),
                    'bend_angle': d.get('bend_angle', ''),
                })
            rows.sort(key=lambda r: (r['chain_id'], r['resseq']))
            tables['geometry'] = rows
        except Exception as e:
            if verbose:
                print(f"    Warning: geometry failed: {e}")
            tables['geometry'] = []

    # ── VDW ───────────────────────────────────────────────────────────────────
    if 'vdw' in analyses and chains_b is not None:
        if verbose:
            print("  [tabular] VDW...")
        if _HAS_VDW:
            try:
                vdw_a = per_residue_LJ_decomposition(cx, chains_a, chains_b, ignore_h=True)
                vdw_b = per_residue_LJ_decomposition(cx, chains_b, chains_a, ignore_h=True)
                combined_vdw = {**vdw_a, **vdw_b}
                rows = []
                for res_id, energy in combined_vdw.items():
                    resname, chain_id, resseq = _parse_res_id(res_id)
                    rows.append({
                        'residue_id':         res_id,
                        'chain_id':           chain_id,
                        'resname':            resname,
                        'resseq':             resseq,
                        'vdw_energy_kcal_mol': _fmt(energy),
                    })
                rows.sort(key=lambda r: (r['chain_id'], r['resseq']))
                tables['vdw'] = rows
            except Exception as e:
                if verbose:
                    print(f"    Warning: VDW failed: {e}")
                tables['vdw'] = []
        else:
            if verbose:
                print("    Warning: VDW module not available (vdw.py missing or not importable).")
            tables['vdw'] = []

    # ── DISTANCES ─────────────────────────────────────────────────────────────
    if 'distances' in analyses and chains_b is not None:
        if verbose:
            print("  [tabular] Distances...")
        try:
            def _atom_dist(a, b):
                dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
                return math.sqrt(dx*dx + dy*dy + dz*dz)

            resA = selection_residues(cx, select(cx, chains_a))
            resB = selection_residues(cx, select(cx, chains_b))

            def _dist_row(res, partner_list, seq_ctx):
                coords = [(a.x, a.y, a.z) for a in res.iter_atoms(ignore_h=True)]
                if not coords:
                    return None
                min_dist = float('inf')
                nearest = None
                ca_self = next((a for a in res.atoms if a.name.strip().upper() == 'CA'), None)
                for partner in partner_list:
                    pc = [(a.x, a.y, a.z) for a in partner.iter_atoms(ignore_h=True)]
                    if pc:
                        d = pairwise_min_distance(coords, pc)
                        if d < min_dist:
                            min_dist = d
                            nearest = partner
                ca_dist = None
                if nearest and ca_self:
                    ca_p = next((a for a in nearest.atoms if a.name.strip().upper() == 'CA'), None)
                    if ca_p:
                        ca_dist = _atom_dist(ca_self, ca_p)
                # Sequence neighbours (±2) in same chain
                avg_nb = None
                if ca_self:
                    idx = next((i for i, r in enumerate(seq_ctx) if r.id_str == res.id_str), None)
                    if idx is not None:
                        nb_dists = []
                        for off in (-2, -1, 1, 2):
                            j = idx + off
                            if 0 <= j < len(seq_ctx) and seq_ctx[j].chain_id == res.chain_id:
                                ca_nb = next((a for a in seq_ctx[j].atoms if a.name.strip().upper() == 'CA'), None)
                                if ca_nb:
                                    nb_dists.append(_atom_dist(ca_self, ca_nb))
                        avg_nb = sum(nb_dists) / len(nb_dists) if nb_dists else None
                resname, chain_id, resseq = _parse_res_id(res.id_str)
                return {
                    'residue_id':                 res.id_str,
                    'chain_id':                   chain_id,
                    'resname':                    resname,
                    'resseq':                     resseq,
                    'min_dist_to_partner_A':      _fmt(min_dist if min_dist != float('inf') else None),
                    'ca_dist_to_partner_A':       _fmt(ca_dist),
                    'avg_dist_to_seq_neighbors_A': _fmt(avg_nb),
                }

            rows = []
            for res in resA:
                row = _dist_row(res, resB, resA)
                if row:
                    rows.append(row)
            for res in resB:
                row = _dist_row(res, resA, resB)
                if row:
                    rows.append(row)
            rows.sort(key=lambda r: (r['chain_id'], r['resseq']))
            tables['distances'] = rows
        except Exception as e:
            if verbose:
                print(f"    Warning: distances failed: {e}")
            tables['distances'] = []

    return tables


def extract_combined_table(
    cx: Complex,
    group_a: Union[str, List[str]],
    group_b: Optional[Union[str, List[str]]] = None,
    analyses: Optional[List[str]] = None,
    interface_cutoff: float = 5.0,
    contact_params: Optional[ContactParameters] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Extract a unified per-residue table combining all selected analyses.

    Runs extract_residue_features() and filters the result to the requested
    analysis columns.  SASA and geometry always cover all chains from both
    groups; contacts / interface / vdw / distances require group_b.

    Returns:
        List of dicts (one per residue) sorted by chain → resseq.
    """
    if analyses is None:
        analyses = list(_ALL_ANALYSES)

    if isinstance(group_a, str):
        group_a = [group_a] if ',' not in group_a else group_a.split(',')
    chains_a = [s.strip() for s in group_a]

    if group_b is not None:
        if isinstance(group_b, str):
            group_b = [group_b] if ',' not in group_b else group_b.split(',')
        chains_b: Optional[List[str]] = [s.strip() for s in group_b]
    else:
        chains_b = None

    features = extract_residue_features(
        cx,
        selection_A=chains_a,
        selection_B=chains_b,
        interface_cutoff=interface_cutoff,
        contact_params=contact_params,
        compute_sasa='sasa' in analyses,
        compute_geometry='geometry' in analyses,
        compute_vdw='vdw' in analyses,
        verbose=verbose,
    )

    # Build ordered column list
    cols = list(_COMBINED_COLS['base'])
    for analysis in analyses:
        if analysis in _COMBINED_COLS:
            cols.extend(c for c in _COMBINED_COLS[analysis] if c not in cols)

    rows: List[Dict] = []
    for res_id in sorted(features.keys()):
        feat = features[res_id]
        raw = feat.to_dict()
        row: Dict = {}
        for col in cols:
            v = raw.get(col)
            if isinstance(v, float):
                v = round(v, 3)
            row[col] = v if v is not None else ''
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get('chain_id', '')), int(r.get('resseq', 0) or 0)))
    return rows


# ── HTML rendering ────────────────────────────────────────────────────────────

def _render_table_pane(name: str, rows: List[Dict], active: str) -> str:
    """Render one Bootstrap tab pane containing a sortable, filterable table."""
    tbl_id = f'tbl-{name}'
    flt_id = f'flt-{name}'

    if not rows:
        show = 'show active' if active else ''
        return (f'<div class="tab-pane fade {show}" id="tab-{name}" role="tabpanel">'
                f'<p class="text-muted mt-2">No data available.</p></div>')

    cols = list(rows[0].keys())

    header_cells = ''.join(
        f'<th class="sortable" onclick="sortTbl(\'{tbl_id}\',{i})">'
        f'{col.replace("_", " ")}<span class="si"></span></th>'
        for i, col in enumerate(cols)
    )

    body_rows = ''.join(
        '<tr>' + ''.join(f'<td>{row.get(col, "")}</td>' for col in cols) + '</tr>'
        for row in rows
    )

    show = 'show active' if active else ''
    return f'''<div class="tab-pane fade {show}" id="tab-{name}" role="tabpanel">
  <div class="d-flex align-items-center gap-2 my-2 flex-wrap">
    <input class="form-control form-control-sm" style="max-width:220px" type="text"
           id="{flt_id}" placeholder="&#128269; Search..."
           oninput="filterTbl('{flt_id}','{tbl_id}')">
    <button class="btn btn-sm btn-outline-success"
            onclick="dlCSV('{tbl_id}','{name}_data.csv')">
      &#8681; Download CSV
    </button>
    <span class="text-muted small">{len(rows)} rows &bull; {len(cols)} columns</span>
  </div>
  <div class="tbl-wrap">
    <table class="table table-sm table-bordered table-hover mb-0" id="{tbl_id}">
      <thead class="table-light sticky-top"><tr>{header_cells}</tr></thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>
</div>'''


def tables_to_html(
    tables: Dict[str, List[Dict]],
    title: str = "Tabular Data",
    subtitle: str = "",
) -> str:
    """
    Generate a self-contained, interactive HTML page from one or more tables.

    Features per table:
    - Sortable columns (click header)
    - Live text search / filter
    - Download as CSV

    Args:
        tables: Dict mapping table name → list of row dicts.
                Use ``{'combined': rows}`` for a single merged table.
        title: Page heading.
        subtitle: Optional description (e.g. structure name, groups).

    Returns:
        Self-contained HTML string (no external dependencies at runtime except CDN).
    """
    # Drop empty tables
    tables = {k: v for k, v in tables.items() if v}
    if not tables:
        return ("<html><body><p class='text-muted p-4'>"
                "No data available (all analyses returned empty results).</p></body></html>")

    tab_nav = ''
    tab_content = ''
    for i, (name, rows) in enumerate(tables.items()):
        active = 'active' if i == 0 else ''
        selected = 'true' if i == 0 else 'false'
        label = _TABLE_LABELS.get(name, name.replace('_', ' ').title())
        tab_nav += (
            f'<li class="nav-item" role="presentation">'
            f'<button class="nav-link {active}" id="tab-{name}-btn"'
            f' data-bs-toggle="tab" data-bs-target="#tab-{name}"'
            f' type="button" role="tab" aria-selected="{selected}">'
            f'{label} <span class="badge bg-secondary">{len(rows)}</span>'
            f'</button></li>\n'
        )
        tab_content += _render_table_pane(name, rows, active)

    subtitle_html = f'<p class="text-muted small mb-2">{subtitle}</p>' if subtitle else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
  crossorigin="anonymous">
<style>
body {{ font-size: 0.875rem; }}
.tbl-wrap {{ max-height: 65vh; overflow-y: auto; overflow-x: auto; }}
.tbl-wrap thead th {{ position: sticky; top: 0; z-index: 2;
  background: #f8f9fa; box-shadow: 0 1px 0 #dee2e6; }}
th.sortable {{ cursor: pointer; user-select: none; white-space: nowrap; }}
th.sortable:hover {{ background-color: #e9ecef !important; }}
th .si {{ font-size: 0.65rem; margin-left: 3px; color: #adb5bd; }}
th .si::after {{ content: "\\21C5"; }}
th.sa .si::after {{ content: "\\25B2"; color: #0d6efd; }}
th.sd .si::after {{ content: "\\25BC"; color: #0d6efd; }}
.table-sm td, .table-sm th {{ padding: 0.2rem 0.45rem; white-space: nowrap; }}
</style>
</head>
<body>
<div class="container-fluid py-3">
  <h5 class="mb-0 fw-semibold">{title}</h5>
  {subtitle_html}
  <ul class="nav nav-tabs mt-3" id="mainTabs" role="tablist">
    {tab_nav}
  </ul>
  <div class="tab-content border border-top-0 p-3 bg-white rounded-bottom">
    {tab_content}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
  crossorigin="anonymous"></script>
<script>
function sortTbl(id, col) {{
  var t = document.getElementById(id);
  var ths = t.querySelectorAll('thead th');
  var th = ths[col];
  var asc = !th.classList.contains('sa');
  ths.forEach(function(h) {{ h.classList.remove('sa', 'sd'); }});
  th.classList.add(asc ? 'sa' : 'sd');
  var tb = t.querySelector('tbody');
  var rows = Array.from(tb.querySelectorAll('tr'));
  rows.sort(function(a, b) {{
    var av = (a.querySelectorAll('td')[col] || {{}}).textContent || '';
    var bv = (b.querySelectorAll('td')[col] || {{}}).textContent || '';
    av = av.trim(); bv = bv.trim();
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(function(r) {{ tb.appendChild(r); }});
}}
function filterTbl(fId, tId) {{
  var q = document.getElementById(fId).value.toLowerCase();
  document.getElementById(tId).querySelectorAll('tbody tr').forEach(function(r) {{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function dlCSV(tId, fname) {{
  var t = document.getElementById(tId);
  var out = [];
  t.querySelectorAll('tr').forEach(function(r) {{
    var cells = r.querySelectorAll('th,td');
    out.push(Array.from(cells).map(function(c) {{
      return '"' + c.textContent.trim().replace(/"/g, '""') + '"';
    }}).join(','));
  }});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out.join('\\n')], {{type:'text/csv'}}));
  a.download = fname; a.click();
}}
</script>
</body>
</html>"""
