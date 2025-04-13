import dataclasses
import hashlib
import os
import pathlib
from collections.abc import Iterable, Sequence
from functools import cache, cached_property, partial
from typing import Literal, overload

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Pharm2D import Generate as Generate2D
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

from .base import Drawable


@dataclasses.dataclass(frozen=True, eq=True, unsafe_hash=True)
class FingerprintOption:
    type: str = "morgan"
    # Morgan
    morgan_radius: int = 2
    morgan_n_bits: int = 256
    # RDKit
    rdkit_fp_size: int = 2048

    def __post_init__(self):
        supported_types = ("morgan", "rdkit", "gobbi_pharm2d")
        if self.type not in supported_types:
            raise ValueError(f"Unsupported fingerprint type: {self.type}")

    @classmethod
    def morgan_for_tanimoto_similarity(cls):
        return FingerprintOption(
            type="morgan",
            morgan_radius=2,
            morgan_n_bits=4096,
        )

    @classmethod
    def gobbi_pharm2d(cls):
        return FingerprintOption(
            type="gobbi_pharm2d",
        )

    @classmethod
    def morgan_for_building_blocks(cls):
        return FingerprintOption(
            type="morgan",
            morgan_radius=2,
            morgan_n_bits=256,
        )

    @classmethod
    def rdkit(cls):
        return FingerprintOption(
            type="rdkit",
        )

    @property
    def dim(self) -> int:
        if self.type == "morgan":
            return self.morgan_n_bits
        elif self.type == "rdkit":
            return self.rdkit_fp_size
        elif self.type == "gobbi_pharm2d":
            return 39972
        raise ValueError(f"Unsupported fingerprint type: {self.type}")


class Molecule(Drawable):
    def __init__(self, smiles: str, source: Literal["smiles", "fp", ''] = '') -> None:
        super().__init__()
        self._smiles = smiles.strip()
        self.meta_info = {}
        self._source = source

    @classmethod
    def from_rdmol(cls, rdmol: Chem.Mol) -> "Molecule":
        return cls(Chem.MolToSmiles(rdmol))

    def __getstate__(self):
        return self._smiles

    def __setstate__(self, state):
        self._smiles = state
        self._source = ''

    @property
    def smiles(self) -> str:
        return self._smiles
    
    @property
    def source(self) -> Literal["smiles", "fp", '']:
        return self._source

    @cached_property
    def _rdmol(self):
        return Chem.MolFromSmiles(self._smiles)

    @cached_property
    def _rdmol_no_hs(self):
        return Chem.RemoveHs(self._rdmol)

    @cached_property
    def is_valid(self) -> bool:
        return self._rdmol is not None

    @cached_property
    def csmiles(self) -> str:
        return Chem.MolToSmiles(self._rdmol, canonical=True, isomericSmiles=False)

    @cached_property
    def num_atoms(self) -> int:
        return self._rdmol.GetNumAtoms()

    def draw(self, size: int = 100, svg: bool = False):
        if svg:
            return Draw._moltoSVG(self._rdmol, sz=(size, size), highlights=[], legend=[], kekulize=True)
        else:
            return Draw.MolToImage(self._rdmol, size=(size, size), kekulize=True)

    def __hash__(self) -> int:
        return hash(self._smiles)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Molecule) and self.csmiles == __value.csmiles

    @cached_property
    def major_molecule(self) -> "Molecule":
        if "." in self.smiles:
            segs = self.smiles.split(".")
            segs.sort(key=lambda a: -len(a))
            return Molecule(segs[0])
        return self

    @overload
    def get_fingerprint(self, option: FingerprintOption) -> np.ndarray: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[True]) -> Sequence[Literal[0, 1]]: ...

    @overload
    def get_fingerprint(self, option: FingerprintOption, as_bitvec: Literal[False]) -> np.ndarray: ...

    def get_fingerprint(self, option: FingerprintOption, as_bitvec: bool = False):
        return self._get_fingerprint(option, as_bitvec)  # work-around for mypy check

    @cache
    def _get_fingerprint(self, option: FingerprintOption, as_bitvec: bool):
        if option.type == "morgan":
            bit_vec = AllChem.GetMorganFingerprintAsBitVect(self._rdmol, option.morgan_radius, option.morgan_n_bits)
        elif option.type == "rdkit":
            bit_vec = Chem.RDKFingerprint(self._rdmol, fpSize=option.rdkit_fp_size)
        elif option.type == "gobbi_pharm2d":
            bit_vec = DataStructs.cDataStructs.ConvertToExplicit(
                Generate2D.Gen2DFingerprint(self._rdmol, Gobbi_Pharm2D.factory)
            )
        else:
            raise ValueError(f"Unsupported fingerprint type: {option.type}")

        if as_bitvec:
            return bit_vec
        feat = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vec, feat)
        return feat

    @cached_property
    def scaffold(self) -> "Molecule":
        s = Molecule.from_rdmol(MurckoScaffold.GetScaffoldForMol(self._rdmol))
        if not s.is_valid:
            s = self
        return s

    def tanimoto_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        fp1 = self.get_fingerprint(fp_option, as_bitvec=True)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=True)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def dice_similarity(self, other: "Molecule", fp_option: FingerprintOption) -> float:
        fp1 = self.get_fingerprint(fp_option, as_bitvec=True)
        fp2 = other.get_fingerprint(fp_option, as_bitvec=True)
        return DataStructs.DiceSimilarity(fp1, fp2)

    @cache
    def sim(
        self,
        other: "Molecule",
        fp_option: FingerprintOption = FingerprintOption.morgan_for_tanimoto_similarity(),
    ) -> float:
        return self.tanimoto_similarity(other, fp_option)

    @cached_property
    def csmiles_md5(self) -> bytes:
        return hashlib.md5(self.csmiles.encode()).digest()

    @cached_property
    def csmiles_sha256(self) -> bytes:
        return hashlib.sha256(self.csmiles.encode()).digest()
    
def get_meta_info(mol):
        return {
            "id": mol.GetProp("id") if mol.HasProp("id") else None,
            "IUPAC Name": mol.GetProp("IUPAC Name") if mol.HasProp("IUPAC Name") else None,
            "CAS": mol.GetProp("CAS") if mol.HasProp("CAS") else None,
            "purity": mol.GetProp("purity") if mol.HasProp("purity") else None,
            "MDLNUMBER": mol.GetProp("MDLNUMBER") if mol.HasProp("MDLNUMBER") else None,
            "LogP": mol.GetProp("LogP") if mol.HasProp("LogP") else None,
            "URL": mol.GetProp("URL") if mol.HasProp("URL") else None,
            "avail_US_100mg": mol.GetProp("avail_US_100mg") if mol.HasProp("avail_US_100mg") else None,
            "avail_US_250mg": mol.GetProp("avail_US_250mg") if mol.HasProp("avail_US_250mg") else None,
            "avail_US_1g": mol.GetProp("avail_US_1g") if mol.HasProp("avail_US_1g") else None,
            "avail_US_2_5g": mol.GetProp("avail_US_2_5g") if mol.HasProp("avail_US_2_5g") else None
        }


def read_mol_file(
    path: os.PathLike,
    major_only: bool = True,
    drop_duplicates: bool = True,
    show_pbar: bool = True,
    smiles_col: str | None = None,
    pbar_fn=partial(tqdm, desc="Reading"),
) -> Iterable[Molecule]:
    path = pathlib.Path(path)
    if path.suffix == ".sdf":
        f = Chem.SDMolSupplier(str(path))
    elif path.suffix == ".smi":
        f = Chem.SmilesMolSupplier(str(path))
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if smiles_col is None:
            if "smiles" in df.columns:
                smiles_col = "smiles"
            elif "SMILES" in df.columns:
                smiles_col = "SMILES"
            else:
                raise ValueError(f"Cannot find SMILES column in {path}")
        f = (Chem.MolFromSmiles(smiles) for smiles in df[smiles_col])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    visited: set[str] = set()
    if show_pbar:
        f_iter = pbar_fn(f)
    else:
        f_iter = f
    for rdmol in f_iter:
        if rdmol is not None:
            meta_info = get_meta_info(rdmol)
            mol = Molecule.from_rdmol(rdmol)
            mol.meta_info = meta_info
            if major_only:
                mol = mol.major_molecule
            if drop_duplicates and mol.csmiles in visited:
                continue
            yield mol
            visited.add(mol.csmiles)


def write_to_smi(path: os.PathLike, mols: Sequence[Molecule]):
    with open(path, "w") as f:
        for mol in mols:
            f.write(f"{mol.smiles}\n")
