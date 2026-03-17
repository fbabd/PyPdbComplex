from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from .models import Complex, Residue, Chain
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
import math 

@dataclass
class ResidueInterfaceData:
    """Interface participation data for a single residue."""
    residue: Residue
    is_interface: bool = False
    contact_count: int = 0
    partner_residues: List[str] = field(default_factory=list)  # IDs of contacting residues
    sasa_details: Optional[Dict[str, float]] = None  # unbound, bound, delta, buried_fraction
    
    @property
    def id_str(self) -> str:
        return self.residue.id_str


@dataclass
class InterfaceAnalysis:
    """Complete interface analysis result with per-residue annotations."""
    
    # Per-chain residue data (chain_id -> list of ResidueInterfaceData)
    chain_data: Dict[str, List[ResidueInterfaceData]] = field(default_factory=dict)
    
    # Summary statistics
    total_contacts: int = 0
    bsa_total: Optional[float] = None
    cutoff: float = 5.0
    
    # Group labels (for display/export)
    group_A_chains: List[str] = field(default_factory=list)
    group_B_chains: List[str] = field(default_factory=list)
    
    # Metadata
    note: str = ""
    
    def get_residue_data(self, chain_id: str, resseq: int, icode: str = "") -> Optional[ResidueInterfaceData]:
        """Get interface data for a specific residue."""
        if chain_id not in self.chain_data:
            return None
        for rd in self.chain_data[chain_id]:
            r = rd.residue
            if r.resseq == resseq and r.icode == icode:
                return rd
        return None
    
    def get_interface_residues(self, chain_ids: Optional[Iterable[str]] = None) -> List[ResidueInterfaceData]:
        """Get all interface residues, optionally filtered by chain."""
        chains = set(chain_ids) if chain_ids else set(self.chain_data.keys())
        result = []
        for cid in chains:
            if cid in self.chain_data:
                result.extend([rd for rd in self.chain_data[cid] if rd.is_interface])
        return result
    
    def get_residue_contacts(self) -> Dict:
        interface_residues = self.get_interface_residues() 
        interface_res_data = {}
        for res in interface_residues:
            # print(res.id_str)
            interface_res_data[res.id_str] = {
                'partners': res.partner_residues,
                'contact_count': res.contact_count
            }
        def sort_key(resid):
            _, chain_pos = resid.split()
            chain, pos = chain_pos.split(":")
            num = int(''.join(filter(str.isdigit, pos)))
            ins = ''.join(filter(str.isalpha, pos))  # insertion code optional
            return (chain, num, ins) 

        sorted_result = dict(sorted(interface_res_data.items(), key=lambda x: sort_key(x[0])))
        return sorted_result 

    
    def get_group_summary(self, group: str = "A") -> Dict:
        """Get summary statistics for a group (A or B)."""
        chains = self.group_A_chains if group == "A" else self.group_B_chains
        interface_res = self.get_interface_residues(chains)
        
        total_residues = sum(len(self.chain_data.get(c, [])) for c in chains)
        interface_count = len(interface_res)
        total_contacts = sum(rd.contact_count for rd in interface_res)
        
        return {
            "chains": chains,
            "total_residues": total_residues,
            "interface_residues": interface_count,
            "interface_percentage": 100 * interface_count / total_residues if total_residues else 0,
            "total_contacts": total_contacts,
            "residue_ids": [rd.id_str for rd in interface_res]
        }
    
    def to_dataframe(self):
        """Export to pandas DataFrame for analysis (optional, if pandas available)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        rows = []
        for chain_id, res_data_list in self.chain_data.items():
            for rd in res_data_list:
                r = rd.residue
                rows.append({
                    "chain_id": chain_id,
                    "resname": r.resname,
                    "resseq": r.resseq,
                    "icode": r.icode,
                    "residue_id": rd.id_str,
                    "is_interface": rd.is_interface,
                    "contact_count": rd.contact_count,
                    "num_partners": len(rd.partner_residues),
                    "in_group_A": chain_id in self.group_A_chains,
                    "in_group_B": chain_id in self.group_B_chains,
                })
        return pd.DataFrame(rows)
    
    def annotate_complex(self, cx: Complex, 
                         interface_attr: str = "is_interface",
                         contact_attr: str = "contact_count") -> Complex:
        """
        Add interface annotations directly to Residue objects in the Complex.
        Returns the same complex (modified in-place) for chaining.
        """
        for chain_id, res_data_list in self.chain_data.items():
            if chain_id not in cx.chains:
                continue
            chain = cx.chains[chain_id]
            
            # Create lookup for fast access
            data_map = {(rd.residue.resseq, rd.residue.icode): rd 
                       for rd in res_data_list}
            
            for residue in chain.iter_residues():
                key = (residue.resseq, residue.icode)
                if key in data_map:
                    rd = data_map[key]
                    # Add attributes dynamically
                    setattr(residue, interface_attr, rd.is_interface)
                    setattr(residue, contact_attr, rd.contact_count)
                else:
                    setattr(residue, interface_attr, False)
                    setattr(residue, contact_attr, 0)
        
        return cx


def pairwise_min_distance(coordsA: List[Tuple[float,float,float]],
                          coordsB: List[Tuple[float,float,float]]) -> float:
    if not coordsA or not coordsB:
        return float("inf")
    if _HAS_NUMPY:
        A = np.array(coordsA, dtype=float)
        B = np.array(coordsB, dtype=float)
        diff = A[:, None, :] - B[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        return float(np.sqrt(d2.min()))
    best = float("inf")
    for (x1,y1,z1) in coordsA:
        for (x2,y2,z2) in coordsB:
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if d < best: best = d
    return best

def compute_interface(
    cx: Complex,
    selection_A: Iterable[str],
    selection_B: Iterable[str],
    cutoff: float = 5.0,
    use_freesasa: bool = False,
    ignore_h: bool = True,
    probe: float = 1.4,
    n_points: int = 960,
) -> InterfaceAnalysis:
    """
    Compute interface between two selections and return comprehensive analysis.
    
    Args:
        cx: Complex structure
        selection_A: Chain IDs for group A (e.g., ["H", "L"] for antibody)
        selection_B: Chain IDs for group B (e.g., ["A"] for antigen)
        cutoff: Distance cutoff in Angstroms
        use_freesasa: Calculate buried surface area
        ignore_h: Ignore hydrogens in distance calculations
        probe: Probe radius for SASA calculation
        n_points: Number of sphere points for approximate SASA
    
    Returns:
        InterfaceAnalysis with per-residue annotations for ALL chains in complex
    """
    
    from .sasa import _sasa_fs, _compute_bsa_fallback  
    
    # Convert to lists for consistent ordering
    chains_A = list(selection_A)
    chains_B = list(selection_B)
    
    # Collect ALL residues from the complex (not just A and B)
    all_chain_residues: Dict[str, List[Residue]] = {}
    for chain_id, chain in cx.chains.items():
        all_chain_residues[chain_id] = list(chain.iter_residues())
    
    # Separate A and B groups
    A_residues = []
    B_residues = []
    for cid in chains_A:
        if cid in all_chain_residues:
            A_residues.extend(all_chain_residues[cid])
    for cid in chains_B:
        if cid in all_chain_residues:
            B_residues.extend(all_chain_residues[cid])
    
    # Initialize interface data for ALL residues in complex
    chain_data: Dict[str, List[ResidueInterfaceData]] = {}
    for chain_id, residues in all_chain_residues.items():
        chain_data[chain_id] = [ResidueInterfaceData(residue=r) for r in residues]
    
    # Create lookup maps for quick access
    residue_to_data: Dict[int, ResidueInterfaceData] = {}
    for res_list in chain_data.values():
        for rd in res_list:
            residue_to_data[id(rd.residue)] = rd
    
    # Pre-compute coordinates
    A_coords = [[a.coord for a in r.iter_atoms(ignore_h=ignore_h)] for r in A_residues]
    B_coords = [[a.coord for a in r.iter_atoms(ignore_h=ignore_h)] for r in B_residues]
    
    total_contacts = 0
    
    # Contact detection
    for i, (res_A, coords_A) in enumerate(zip(A_residues, A_coords)):
        if not coords_A:
            continue
        
        for j, (res_B, coords_B) in enumerate(zip(B_residues, B_coords)):
            if not coords_B:
                continue
            
            dmin = pairwise_min_distance(coords_A, coords_B)
            if dmin <= cutoff:
                # Count atom-atom contacts
                contact_count = 0
                for (x1, y1, z1) in coords_A:
                    for (x2, y2, z2) in coords_B:
                        dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
                        if dx*dx + dy*dy + dz*dz <= cutoff * cutoff:
                            contact_count += 1
                
                # Update interface data
                rd_A = residue_to_data[id(res_A)]
                rd_B = residue_to_data[id(res_B)]
                
                rd_A.is_interface = True
                rd_B.is_interface = True
                rd_A.contact_count += contact_count
                rd_B.contact_count += contact_count
                rd_A.partner_residues.append(res_B.id_str)
                rd_B.partner_residues.append(res_A.id_str)
                
                total_contacts += contact_count
    
    # BSA calculation (optional)
    bsa = None
    note = ""
    if use_freesasa:
        try:
            A_atoms = [a for r in A_residues for a in r.iter_atoms(ignore_h=False)]
            B_atoms = [a for r in B_residues for a in r.iter_atoms(ignore_h=False)]

            sasaA = _sasa_fs(A_atoms)
            sasaB = _sasa_fs(B_atoms)
            sasaAB = _sasa_fs(A_atoms + B_atoms)
            bsa = (sasaA + sasaB - sasaAB) / 2.0
            note = "BSA calculated with FreeSASA"
        except Exception as e:
            A_atoms = [a for r in A_residues for a in r.iter_atoms(ignore_h=False)]
            B_atoms = [a for r in B_residues for a in r.iter_atoms(ignore_h=False)]
            bsa, note = _compute_bsa_fallback(A_atoms, B_atoms, probe=probe, n_points=n_points)

        # Per-residue SASA details (bound vs unbound for all chains)
        try:
            from .sasa_analysis import compare_bound_unbound_sasa
            all_chain_ids = list(cx.chains.keys())
            sasa_results = compare_bound_unbound_sasa(cx, all_chain_ids, probe=probe, ignore_h=ignore_h)
            for res_data_list in chain_data.values():
                for rd in res_data_list:
                    if rd.id_str in sasa_results:
                        rd.sasa_details = sasa_results[rd.id_str]
        except Exception:
            pass  # SASA details unavailable
    
    return InterfaceAnalysis(
        chain_data=chain_data,
        total_contacts=total_contacts,
        bsa_total=bsa,
        cutoff=cutoff,
        group_A_chains=chains_A,
        group_B_chains=chains_B,
        note=note
    )


