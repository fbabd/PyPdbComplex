from __future__ import annotations
from typing import List, Tuple, Set, Dict, Union
import re
from .models import Complex, Selection, Residue

def _parse_residue_token(token: str) -> Tuple[int, str]:
    """Parse a residue token like '30' or '30A' into (resseq, icode)."""
    match = re.fullmatch(r"([0-9]+)([A-Za-z]?)", token)
    if not match:
        raise ValueError(f"Invalid residue token: {token}")
    return int(match.group(1)), match.group(2)

def _residue_in_range(
    resseq: int, 
    icode: str,
    start_resseq: int, 
    start_icode: str,
    end_resseq: int, 
    end_icode: str
) -> bool:
    """Check if a residue falls within the specified range."""
    res_key = (resseq, icode or "")
    start_key = (start_resseq, start_icode or "")
    end_key = (end_resseq, end_icode or "")
    return start_key <= res_key <= end_key

def _select_one(complex_obj: Complex, expr: str) -> Selection:
    """
    Parse a single selection expression and return matching atoms.
    
    Supported formats:
    - Chain: "A"
    - Residue: "A:30"
    - Range: "A:30-36"
    - Atom: "A:30.CA"
    - Named group: any key in complex_obj._named_groups
    """
    expr = (expr or "").strip()
    if not expr:
        return Selection([])

    # Named group?
    if expr in complex_obj._named_groups:
        return _select_one(complex_obj, complex_obj._named_groups[expr])

    # Chain only?
    chain_match = re.fullmatch(r"([A-Za-z0-9_])", expr)
    if chain_match:
        chain_id = chain_match.group(1)
        if chain_id not in complex_obj.chains:
            return Selection([])
        return Selection(list(complex_obj.chains[chain_id].iter_atoms(ignore_h=False)))

    # Chain:residue pattern
    pattern = r"([A-Za-z0-9_]):([0-9]+[A-Za-z]?)(?:-([0-9]+[A-Za-z]?))?(?:\.([A-Za-z0-9]+))?"
    match = re.fullmatch(pattern, expr)
    if not match:
        raise ValueError(f"Unrecognized selection expression: {expr}")
    
    chain_id, start_tok, end_tok, atom_name = match.groups()
    if chain_id not in complex_obj.chains:
        return Selection([])

    # Parse range
    start_resseq, start_icode = _parse_residue_token(start_tok)
    if end_tok:
        end_resseq, end_icode = _parse_residue_token(end_tok)
    else:
        end_resseq, end_icode = start_resseq, start_icode

    # Collect atoms
    atoms = []
    for residue in complex_obj.chains[chain_id].iter_residues():
        if _residue_in_range(
            residue.resseq, residue.icode,
            start_resseq, start_icode,
            end_resseq, end_icode
        ):
            if atom_name:
                # Filter by atom name (case-insensitive)
                atoms.extend(
                    atom for atom in residue.atoms 
                    if atom.name.strip().upper() == atom_name.upper()
                )
            else:
                atoms.extend(residue.atoms)

    return Selection(atoms)

# Type aliases
ExprOrSel = Union[str, Selection]
ManyExprOrSel = Union[ExprOrSel, List[ExprOrSel], Tuple[ExprOrSel, ...]]

def select(complex_obj: Complex, expr_or_list: ManyExprOrSel) -> Selection:
    """
    Flexible selection supporting multiple input types.
    
    Args:
        complex_obj: The protein complex to select from
        expr_or_list: Can be:
            - A string expression (e.g., "A:30-36.CA")
            - A Selection object (returned as-is)
            - A list/tuple of strings/Selections (union of all)
    
    Returns:
        Selection object with deduplicated atoms (by serial number)
    
    Examples:
        select(complex, "A:30")
        select(complex, ["A:30", "B:40-50"])
        select(complex, [existing_selection, "C:10.CA"])
    """
    # Single Selection passthrough
    if isinstance(expr_or_list, Selection):
        return expr_or_list

    # Single string
    if isinstance(expr_or_list, str):
        return _select_one(complex_obj, expr_or_list)

    # Iterable: union with deduplication
    if isinstance(expr_or_list, (list, tuple)):
        seen: Set[int] = set()
        atoms = []
        
        for item in expr_or_list:
            if isinstance(item, Selection):
                sel = item
            elif isinstance(item, str):
                sel = _select_one(complex_obj, item)
            else:
                raise TypeError(
                    f"Selection items must be str or Selection, got {type(item).__name__}"
                )
            
            for atom in sel.atoms:
                if atom.serial not in seen:
                    seen.add(atom.serial)
                    atoms.append(atom)
        
        return Selection(atoms)

    raise TypeError(
        f"Selection must be str, Selection, or list/tuple thereof, got {type(expr_or_list).__name__}"
    )

def selection_residues(complex_obj: Complex, sel: Selection) -> List[Residue]:
    """
    Extract unique residues from an atom selection.
    
    Args:
        complex_obj: The protein complex
        sel: Selection of atoms
    
    Returns:
        Sorted list of unique residues
    """
    seen: Set[Tuple[str, int, str]] = set()
    residues: List[Residue] = []
    
    for atom in sel.atoms:
        key = (atom.chain_id, atom.resseq, atom.icode)
        if key not in seen:
            seen.add(key)
            residue = complex_obj.chains[atom.chain_id].residues.get(key)
            if residue:
                residues.append(residue)
    
    residues.sort(key=lambda r: (r.chain_id, r.resseq, r.icode or ""))
    return residues

