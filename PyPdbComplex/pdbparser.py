from __future__ import annotations
from typing import Dict, List, Tuple
import re
from collections import defaultdict, OrderedDict
from .models import Atom, Chain, Complex

_LINE_RE = re.compile(r"^(ATOM  |HETATM)")

def parse_pdb(path: str,
              keep_hetatm: bool = False,
              altloc_policy: str = "occupancy_max") -> Complex:
    """
    Parse PDB (ATOM [+ optional HETATM]) into Complex.
    altloc_policy: 'occupancy_max' | 'A' | 'first'
    """
    atoms: List[Atom] = [] 
    pdb_lines: List[str] = [] 
    
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw_atoms: Dict[Tuple[str,int,str,str,str], List[Atom]] = defaultdict(list)
        for line in fh:
            pdb_lines.append(line) 
            
            if not _LINE_RE.match(line):
                continue
            if line.startswith("HETATM") and not keep_hetatm:
                continue
            try:
                serial  = int(line[6:11])
                name    = line[12:16].strip()
                altloc  = line[16].strip()
                resname = line[17:20].strip()
                chain   = line[21].strip() or "_"
                resseq  = int(line[22:26])
                icode   = line[26].strip()
                x       = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                occ     = float(line[56:60].strip() or 1.0)
                bfac    = float(line[60:66].strip() or 0.0)
                elem    = (line[76:78].strip() or name[0].upper() or "C")
            except Exception:
                continue
            a = Atom(serial, name, altloc, resname, chain, resseq, icode, x, y, z, occ, bfac, elem)
            key = (chain, resseq, icode, name, altloc or " ")
            raw_atoms[key].append(a)

        by_site = defaultdict(list)  # (chain, resseq, icode, name) -> atoms with altlocs
        for (chain, resseq, icode, name, altloc), lst in raw_atoms.items():
            by_site[(chain, resseq, icode, name)].extend(lst)

        for site_key, lst in by_site.items():
            if len(lst) == 1 or altloc_policy == "first":
                atoms.append(lst[0])
            elif altloc_policy == "A":
                chosen = next((a for a in lst if (a.altloc or " ") == "A"), lst[0])
                atoms.append(chosen)
            elif altloc_policy == "occupancy_max":
                atoms.append(max(lst, key=lambda a: a.occupancy))
            else:
                atoms.append(lst[0])

    chains: Dict[str, Chain] = {}
    for a in atoms:
        if a.chain_id not in chains:
            chains[a.chain_id] = Chain(a.chain_id, OrderedDict())
        chains[a.chain_id].add_atom(a)
    pdb_content = "".join(pdb_lines) 
    
    return Complex(chains=chains, source_path=path, pdb_content=pdb_content)

