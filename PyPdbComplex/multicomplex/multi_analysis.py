"""
Modular Multi-Complex Analysis System
Allows users to select which properties to calculate and compare
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import traceback
from io import StringIO

# Import analysis modules
from ..pdbparser import parse_pdb
from ..sasa_analysis import compare_bound_unbound_sasa
from ..contacts import analyze_contacts
from ..interface import compute_interface
from ..vdw import per_residue_LJ_decomposition
# geometry module not used yet - will add RMSD later if needed


@dataclass
class AnalysisConfig:
    """Configuration for which analyses to run"""
    calculate_sasa: bool = True
    calculate_contacts: bool = True
    calculate_interface: bool = True
    calculate_vdw: bool = False
    # RMSD removed - not implemented yet
    
    # Advanced options
    sasa_probe_radius: float = 1.4
    contact_distance: float = 5.0
    vdw_cutoff: float = 5.0
    interface_distance: float = 5.0


@dataclass
class ComplexData:
    """Data container for a single complex"""
    name: str
    filepath: str
    structure: object = None

    # Superposed PDB text for 3D viewer (empty string = not yet aligned)
    aligned_pdb_content: str = ""

    # Analysis results (populated as needed)
    sasa_data: Dict = field(default_factory=dict)
    contact_data: Dict = field(default_factory=dict)
    interface_data: Dict = field(default_factory=dict)
    vdw_data: Dict = field(default_factory=dict)
    # rmsd_data removed - not implemented yet

    # Summary statistics
    summary: Dict = field(default_factory=dict)


class MultiComplexAnalyzer:
    """
    Main class for multi-complex comparison with modular analysis selection
    """
    
    def __init__(self, pdb_files: List[Tuple[str, str]]):
        """
        Initialize analyzer with PDB files
        
        Args:
            pdb_files: List of (name, filepath) tuples
        """
        self.complexes: Dict[str, ComplexData] = {}
        self.wt_name: Optional[str] = None
        self.chain_groups: Dict[str, List[str]] = {}
        
        # Load all structures
        print(f"\nLoading {len(pdb_files)} structures...")
        for name, filepath in pdb_files:
            try:
                structure = parse_pdb(filepath)
                self.complexes[name] = ComplexData(
                    name=name,
                    filepath=filepath,
                    structure=structure
                )
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
                raise
    
    
    def set_reference(self, wt_name: str):
        """Set the wild-type reference structure"""
        if wt_name not in self.complexes:
            raise ValueError(f"Reference '{wt_name}' not found in loaded structures")
        self.wt_name = wt_name
        print(f"Reference set to: {wt_name}")
    
    
    def set_chain_groups(self, chain_groups: Dict[str, List[str]]):
        """Set chain groupings for analysis"""
        self.chain_groups = chain_groups
        print(f"Chain groups configured: {list(chain_groups.keys())}")


    def _resolve_groups_for_structure(
        self, complex_data: ComplexData
    ) -> Optional[Tuple[List[str], List[str]]]:
        """
        Resolve user-defined chain groups to chains actually present in this structure.

        Group 0 becomes selection A; all remaining groups are combined into selection B.
        Chains in a group that do not exist in this structure are silently skipped,
        allowing structures with different chain naming conventions to be compared.

        Returns:
            (group_A_chains, group_B_chains) if both sides are non-empty, else None.
            None signals callers to fall back to the default first-chain-vs-rest logic.
        """
        if not self.chain_groups or len(self.chain_groups) < 2:
            return None

        available = set(complex_data.structure.chains.keys())
        group_names = list(self.chain_groups.keys())

        group_A = [c for c in self.chain_groups[group_names[0]] if c in available]
        group_B: List[str] = []
        for name in group_names[1:]:
            group_B.extend(c for c in self.chain_groups[name] if c in available)

        if not group_A or not group_B:
            missing_A = [c for c in self.chain_groups[group_names[0]] if c not in available]
            print(f"     ⚠ Chain group resolution failed for '{complex_data.name}': "
                  f"group_A missing {missing_A}, group_B resolved to {group_B}. "
                  f"Falling back to first-chain vs rest.")
            return None

        return group_A, group_B


    # -------------------------------------------------------------------------
    # Structural superposition (Biopython Superimposer)
    # -------------------------------------------------------------------------

    def superpose_to_reference(
        self, align_chains: Optional[List[str]] = None
    ) -> None:
        """
        Superpose all variant structures onto the WT reference using Cα atoms
        (via Biopython's Superimposer) and store the transformed PDB text in
        ComplexData.aligned_pdb_content.

        The WT reference is left in its original frame; all other structures
        are rotated and translated to minimise Cα RMSD against the WT.

        Chain matching strategy
        -----------------------
        Chains are paired **by position** within the alignment group rather
        than by chain ID.  This handles the common case where the same
        functional chain has a different letter in different structures
        (e.g., WT has chains H,L and a variant has A,B — chains are paired
        H↔A and L↔B by sorted alphabetical order).
        Within each pair, residues are matched by (resseq, icode) alone.
        """
        from Bio.PDB import PDBParser as BioPDBParser, Superimposer, PDBIO

        if self.wt_name is None:
            print("  ⚠ superpose_to_reference: no reference set, skipping.")
            return

        wt_data = self.complexes[self.wt_name]

        # WT: keep in its original frame (it is the target)
        wt_data.aligned_pdb_content = wt_data.structure.pdb_content

        # Parse WT once with Biopython — used as the fixed reference
        bio_parser = BioPDBParser(QUIET=True)
        wt_bio = bio_parser.get_structure("wt", wt_data.filepath)
        wt_align_chains = sorted(c.id for c in wt_bio.get_chains())

        print(f"\n  → Superposing structures onto '{self.wt_name}' "
              f"using all chains {wt_align_chains}...")

        def _ca_map(bio_chain):
            """Return {(resseq, icode): CA_atom} for a Biopython chain."""
            m = {}
            for res in bio_chain.get_residues():
                if 'CA' in res:
                    rid = res.get_id()
                    m[(rid[1], rid[2].strip())] = res['CA']
            return m

        for name, complex_data in self.complexes.items():
            if name == self.wt_name:
                continue

            var_bio = bio_parser.get_structure("var", complex_data.filepath)
            var_align_chains = sorted(c.id for c in var_bio.get_chains())

            n_pairs = min(len(wt_align_chains), len(var_align_chains))
            if n_pairs == 0:
                print(f"     ⚠ '{name}': no chains found — skipping superposition.")
                complex_data.aligned_pdb_content = complex_data.structure.pdb_content
                continue

            # Build matched CA atom lists using positional chain pairing
            fixed_atoms = []
            mobile_atoms = []

            for wt_cid, var_cid in zip(wt_align_chains[:n_pairs],
                                       var_align_chains[:n_pairs]):
                wt_ca  = _ca_map(wt_bio[0][wt_cid])
                var_ca = _ca_map(var_bio[0][var_cid])
                for rk in sorted(set(wt_ca.keys()) & set(var_ca.keys())):
                    fixed_atoms.append(wt_ca[rk])
                    mobile_atoms.append(var_ca[rk])

            if len(fixed_atoms) < 3:
                print(f"     ⚠ '{name}': only {len(fixed_atoms)} matching Cα "
                      f"pairs — skipping superposition (need ≥ 3).")
                complex_data.aligned_pdb_content = complex_data.structure.pdb_content
                continue

            # Superimpose: set_atoms(fixed, mobile), then apply to all variant atoms
            sup = Superimposer()
            sup.set_atoms(fixed_atoms, mobile_atoms)
            sup.apply(list(var_bio.get_atoms()))

            # Serialize the transformed structure to a PDB string
            output = StringIO()
            bio_io = PDBIO()
            bio_io.set_structure(var_bio)
            bio_io.save(output)
            complex_data.aligned_pdb_content = output.getvalue()

            print(f"     ✓ Superposed '{name}' onto '{self.wt_name}' "
                  f"({len(fixed_atoms)} Cα pairs, RMSD = {sup.rms:.3f} Å, "
                  f"chains {list(zip(wt_align_chains[:n_pairs], var_align_chains[:n_pairs]))})")


    def run_analysis(self, config: AnalysisConfig) -> Dict[str, ComplexData]:
        """
        Run selected analyses on all complexes
        
        Args:
            config: Configuration specifying which analyses to run
            
        Returns:
            Dictionary of complex name -> ComplexData with results
        """
        print("\n" + "="*60)
        print("STARTING MULTI-COMPLEX ANALYSIS")
        print("="*60)
        
        analyses_enabled = []
        if config.calculate_sasa:
            analyses_enabled.append("SASA")
        if config.calculate_contacts:
            analyses_enabled.append("Contacts")
        if config.calculate_interface:
            analyses_enabled.append("Interface")
        if config.calculate_vdw:
            analyses_enabled.append("VDW")
        
        print(f"Analyses enabled: {', '.join(analyses_enabled)}")
        print(f"Processing {len(self.complexes)} structures...")
        print()
        
        for name, complex_data in self.complexes.items():
            print(f"Analyzing: {name}")
            print("-" * 40)
            
            # Run each selected analysis
            if config.calculate_sasa:
                self._calculate_sasa(complex_data, config)
            
            if config.calculate_contacts:
                self._calculate_contacts(complex_data, config)
            
            if config.calculate_interface:
                self._calculate_interface(complex_data, config)
            
            if config.calculate_vdw:
                self._calculate_vdw(complex_data, config)
            
            print()
        
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return self.complexes
    
    
    def _calculate_sasa(self, complex_data: ComplexData, config: AnalysisConfig):
        """Calculate SASA for a complex"""
        try:
            print("  → Calculating SASA...")

            # Use only the chains from user-defined groups when available,
            # combining both sides so bound-vs-unbound is scoped to the selection.
            resolved = self._resolve_groups_for_structure(complex_data)
            if resolved:
                group_A, group_B = resolved
                chain_ids = group_A + group_B
                print(f"     Using groups: A={group_A}, B={group_B}")
            else:
                chain_ids = list(complex_data.structure.chains.keys())

            sasa_results = compare_bound_unbound_sasa(
                complex_data.structure,
                chain_ids,
                probe=config.sasa_probe_radius
            )
            
            complex_data.sasa_data = sasa_results
            
            # Calculate summary stats
            total_buried = sum(
                res_data.get('buried_sasa', 0) 
                for res_data in sasa_results.values()
            )
            complex_data.summary['total_buried_sasa'] = total_buried
            complex_data.summary['num_residues_with_sasa'] = len(sasa_results)
            
            print(f"     ✓ SASA calculated ({len(sasa_results)} residues)")
            
        except Exception as e:
            print(f"     ✗ SASA calculation failed: {e}")
            traceback.print_exc()
            complex_data.sasa_data = {}
    
    
    def _calculate_contacts(self, complex_data: ComplexData, config: AnalysisConfig):
        """Calculate inter-chain contacts"""
        try:
            print("  → Calculating contacts...")
            chain_ids = list(complex_data.structure.chains.keys())

            if len(chain_ids) < 2:
                print("     ⚠ Skipped (only 1 chain)")
                return

            # Use user-defined chain groups when available, else first chain vs rest
            resolved = self._resolve_groups_for_structure(complex_data)
            if resolved:
                group_A, group_B = resolved
                # Pass lists directly — select() in selection.py unions items in a list
                sel_A = group_A if len(group_A) > 1 else group_A[0]
                sel_B = group_B if len(group_B) > 1 else group_B[0]
                print(f"     Using groups: A={group_A}, B={group_B}")
            else:
                sel_A = chain_ids[0]
                sel_B = chain_ids[1:] if len(chain_ids) > 2 else chain_ids[1]

            contact_analysis = analyze_contacts(
                complex_data.structure,
                selection_A=sel_A,
                selection_B=sel_B
            )
            
            complex_data.contact_data = {
                'residue_summary': contact_analysis.get_residue_summary(),
                'bond_counts': contact_analysis.get_contact_counts()
            }
            
            # Store summary
            bond_counts = contact_analysis.get_contact_counts()
            for bond_type, count in bond_counts.items():
                complex_data.summary[f'contacts_{bond_type.name}'] = count
            
            total_contacts = sum(bond_counts.values())
            complex_data.summary['total_contacts'] = total_contacts
            
            print(f"     ✓ Contacts calculated ({total_contacts} total)")
            
        except Exception as e:
            print(f"     ✗ Contact calculation failed: {e}")
            traceback.print_exc()
            complex_data.contact_data = {}
    
    
    def _calculate_interface(self, complex_data: ComplexData, config: AnalysisConfig):
        """Calculate interface properties"""
        try:
            print("  → Calculating interface...")
            chain_ids = list(complex_data.structure.chains.keys())

            if len(chain_ids) < 2:
                print("     ⚠ Skipped (only 1 chain)")
                return

            # Use user-defined chain groups when available, else first chain vs rest
            resolved = self._resolve_groups_for_structure(complex_data)
            if resolved:
                group_A, group_B = resolved
                print(f"     Using groups: A={group_A}, B={group_B}")
            else:
                group_A = [chain_ids[0]]
                group_B = chain_ids[1:]

            interface_analysis = compute_interface(
                complex_data.structure,
                selection_A=group_A,
                selection_B=group_B,
                cutoff=config.interface_distance
            )
            
            # Convert InterfaceAnalysis object to dictionary format expected by dashboard
            interface_residues = interface_analysis.get_interface_residues()
            residues_dict = {}
            
            for res_data in interface_residues:
                # res_data is ResidueInterfaceData object
                res_id = res_data.id_str
                residues_dict[res_id] = {
                    'is_interface': res_data.is_interface,
                    'contact_count': res_data.contact_count,
                    'partner_residues': res_data.partner_residues
                }
            
            # Store as dictionary
            complex_data.interface_data = {
                'residues': residues_dict,
                'total_contacts': interface_analysis.total_contacts,
                'bsa_total': interface_analysis.bsa_total,
                'cutoff': interface_analysis.cutoff,
                'group_A_chains': interface_analysis.group_A_chains,
                'group_B_chains': interface_analysis.group_B_chains
            }
            
            # Store summary
            complex_data.summary['interface_residues'] = interface_analysis.total_contacts
            complex_data.summary['buried_surface_area'] = interface_analysis.bsa_total or 0.0
            
            
            print(f"     ✓ Interface calculated ({interface_analysis.total_contacts} contacts)")
            
        except Exception as e:
            print(f"     ✗ Interface calculation failed: {e}")
            traceback.print_exc()
            complex_data.interface_data = {}
    
    
    def _calculate_vdw(self, complex_data: ComplexData, config: AnalysisConfig):
        """Calculate VDW energies"""
        try:
            print("  → Calculating VDW energies...")
            chain_ids = list(complex_data.structure.chains.keys())

            if len(chain_ids) < 2:
                print("     ⚠ Skipped (only 1 chain)")
                return

            # Use user-defined chain groups when available, else each chain vs all others
            resolved = self._resolve_groups_for_structure(complex_data)
            if resolved:
                group_A, group_B = resolved
                print(f"     Using groups: A={group_A}, B={group_B}")
                all_vdw = {}
                vdw_group_a = per_residue_LJ_decomposition(
                    complex_data.structure,
                    group_A,
                    group_B,
                    cutoff=config.vdw_cutoff
                )
                all_vdw.update(vdw_group_a)
                vdw_group_b = per_residue_LJ_decomposition(
                    complex_data.structure,
                    group_B,
                    group_A,
                    cutoff=config.vdw_cutoff
                )
                all_vdw.update(vdw_group_b)
                
            else:
                all_vdw = {}
                for i, chain in enumerate(chain_ids):
                    group_a = [chain]
                    group_b = [ch for j, ch in enumerate(chain_ids) if j != i]
                    vdw_energies = per_residue_LJ_decomposition(
                        complex_data.structure,
                        group_a,
                        group_b,
                        cutoff=config.vdw_cutoff
                    )
                    all_vdw.update(vdw_energies)
            
            complex_data.vdw_data = all_vdw
            
            # Calculate summary
            total_vdw = sum(all_vdw.values())
            favorable_vdw = sum(e for e in all_vdw.values() if e < 0)
            
            complex_data.summary['total_vdw_energy'] = total_vdw
            complex_data.summary['favorable_vdw_energy'] = favorable_vdw
            complex_data.summary['num_residues_with_vdw'] = len(all_vdw)
            
            print(f"     ✓ VDW calculated ({len(all_vdw)} residues, total: {total_vdw:.2f} kcal/mol)")
            
        except Exception as e:
            print(f"     ✗ VDW calculation failed: {e}")
            traceback.print_exc()
            complex_data.vdw_data = {}
    
    
    def get_summary_table(self) -> Dict[str, Dict]:
        """
        Get summary statistics for all complexes
        
        Returns:
            Dictionary of complex name -> summary statistics
        """
        return {
            name: complex_data.summary
            for name, complex_data in self.complexes.items()
        }
    
    
    def export_results(self) -> Dict:
        """
        Export all results in format compatible with existing multi_dashboard.py
        
        Returns:
            Dictionary containing all analysis results in legacy format
        """
        # Convert to format expected by existing dashboard
        results = {}
        
        for name, complex_data in self.complexes.items():
            # Build residue-level data structure
            residues = {}
            
            # Collect all unique residues
            all_residue_ids = set()
            if complex_data.sasa_data:
                all_residue_ids.update(complex_data.sasa_data.keys())
            if complex_data.contact_data and 'residue_summary' in complex_data.contact_data:
                all_residue_ids.update(complex_data.contact_data['residue_summary'].keys())
            if complex_data.interface_data and 'residues' in complex_data.interface_data:
                all_residue_ids.update(complex_data.interface_data['residues'].keys())
            if complex_data.vdw_data:
                all_residue_ids.update(complex_data.vdw_data.keys())
            
            # Build per-residue data
            for res_id in all_residue_ids:
                residues[res_id] = {
                    'sasa': complex_data.sasa_data.get(res_id, {}),
                    'bonds': complex_data.contact_data.get('residue_summary', {}).get(res_id, {}),
                    'interface': complex_data.interface_data.get('residues', {}).get(res_id, 0),
                    'vdw_energy': complex_data.vdw_data.get(res_id, 0.0)
                }
            
            # Build summary data
            summary = complex_data.summary.copy()
            
            # Add contact counts if available
            if complex_data.contact_data and 'bond_counts' in complex_data.contact_data:
                for bond_type, count in complex_data.contact_data['bond_counts'].items():
                    summary[f'contacts_{bond_type}'] = count
            
            # Build result entry
            results[name] = {
                'residues': residues,
                'summary': summary
            }
        
        return results
    
    