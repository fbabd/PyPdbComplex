from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Iterator, Optional
from collections import OrderedDict

@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    altloc: str
    resname: str
    chain_id: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def resid_str(self) -> str:
        ic = self.icode.strip()
        return f"{self.chain_id}:{self.resseq}{ic}"

    @property
    def atom_id(self) -> str:
        ic = self.icode.strip()
        return f"{self.chain_id}:{self.resseq}{ic}.{self.name.strip()}"

@dataclass
class Residue:
    resname: str
    chain_id: str
    resseq: int
    icode: str
    atoms: List[Atom] = field(default_factory=list)

    def add_atom(self, a: Atom) -> None:
        self.atoms.append(a)

    def iter_atoms(self, ignore_h: bool = True) -> Iterator[Atom]:
        for a in self.atoms:
            if ignore_h and (a.element.upper() == "H" or a.name.strip().startswith("H")):
                continue
            yield a

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)

    @property
    def id_str(self) -> str:
        ic = self.icode.strip()
        return f"{self.resname} {self.chain_id}:{self.resseq}{ic}"

@dataclass
class Chain:
    chain_id: str
    residues: "OrderedDict[Tuple[str,int,str], Residue]" = field(default_factory=OrderedDict)

    def add_atom(self, a: Atom) -> None:
        k = (a.chain_id, a.resseq, a.icode)
        if k not in self.residues:
            self.residues[k] = Residue(a.resname, a.chain_id, a.resseq, a.icode, [])
        self.residues[k].add_atom(a)

    def iter_residues(self) -> Iterator[Residue]:
        return iter(self.residues.values())

    def iter_atoms(self, ignore_h: bool = True) -> Iterator[Atom]:
        for r in self.iter_residues():
            yield from r.iter_atoms(ignore_h=ignore_h)

@dataclass
class Selection:
    atoms: List[Atom]


@dataclass
class Complex:
    chains: Dict[str, Chain]
    source_path: str = ""
    pdb_content: str = "" 
    _named_groups: Dict[str, str] = field(default_factory=dict)

    # group aliases (e.g., CDRs)
    def add_group(self, name: str, expr: str) -> None:
        self._named_groups[name] = expr

    def list_groups(self) -> Dict[str, str]:
        return dict(self._named_groups)
    

@dataclass
class SimpleGraph:
    nodes: List[Tuple[str, Union[Atom, Residue]]] = field(default_factory=list)
    edges: List[Tuple[int, int, float, str]] = field(default_factory=list)  # i,j,weight,type

    def add_node(self, id_str: str, payload: Union[Atom, Residue]) -> int:
        self.nodes.append((id_str, payload))
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, weight: float, edge_type: str = "cross") -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        self.edges.append((i, j, float(weight), edge_type)) 

