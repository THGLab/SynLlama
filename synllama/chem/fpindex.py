import dataclasses
import functools
import os, pickle
import pathlib
import tempfile
from collections.abc import Iterable, Sequence

import joblib
import numpy as np
import torch
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

from synllama.chem.mol import FingerprintOption, Molecule


@dataclasses.dataclass
class _QueryResult:
    index: int
    molecule: Molecule
    fingerprint: np.ndarray
    distance: float


def _fill_fingerprint(
    fp: np.memmap,
    offset: int,
    molecules: Iterable[Molecule],
    fp_option: FingerprintOption,
):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    for i, mol in enumerate(molecules):
        fp[offset + i] = mol.get_fingerprint(fp_option).astype(np.uint8)


def compute_fingerprints(
    molecules: Sequence[Molecule],
    fp_option: FingerprintOption,
    batch_size: int = 1024,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tempdir_s:
        temp_fname = pathlib.Path(tempdir_s) / "fingerprint"
        fp = np.memmap(
            str(temp_fname),
            dtype=np.uint8,
            mode="w+",
            shape=(len(molecules), fp_option.dim),
        )
        joblib.Parallel(n_jobs=joblib.cpu_count() // 2)(
            joblib.delayed(_fill_fingerprint)(
                fp=fp,
                offset=start,
                molecules=molecules[start : start + batch_size],
                fp_option=fp_option,
            )
            for start in tqdm(range(0, len(molecules), batch_size), desc="Fingerprint")
        )
        return np.array(fp)


class FingerprintIndex:
    def __init__(self, molecules: Iterable[Molecule], fp_option: FingerprintOption) -> None:
        super().__init__()
        self._molecules = tuple(molecules)
        self._fp_option = fp_option
        self._fp = self._init_fingerprint()
        self._tree = self._init_tree()

    @property
    def molecules(self) -> tuple[Molecule, ...]:
        return self._molecules

    @property
    def fp_option(self) -> FingerprintOption:
        return self._fp_option

    def _init_fingerprint(self, batch_size: int = 1024) -> np.ndarray:
        return compute_fingerprints(
            molecules=self._molecules,
            fp_option=self._fp_option,
            batch_size=batch_size,
        )

    def _init_tree(self) -> BallTree:
        tree = BallTree(self._fp, metric="manhattan")
        return tree

    def __getitem__(self, index: int) -> tuple[Molecule, np.ndarray]:
        return self._molecules[index], self._fp[index]
    
    def query_single(self, q: np.ndarray, k: int) -> list[_QueryResult]:
        dist, idx = self._tree.query(q.reshape([1, self._fp_option.dim]), k=k)
        results = []
        for distance, idx in zip(dist[0], idx[0]):
            results.append(_QueryResult(
                index=idx,
                molecule=self._molecules[idx],
                fingerprint=None,
                distance=distance
            ))
        return sorted(results, key=lambda x: x.distance)

    def query(self, q: np.ndarray, k: int) -> list[list[_QueryResult]]:
        """
        Args:
            q: shape (bsz, ..., fp_dim)
        """
        bsz = q.shape[0]
        dist, idx = self._tree.query(q.reshape([-1, self._fp_option.dim]), k=k)
        dist = dist.reshape([bsz, -1])
        idx = idx.reshape([bsz, -1])
        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results

    @functools.cache
    def fp_cuda(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._fp, dtype=torch.float, device=device)

    @torch.inference_mode()
    def query_cuda(self, q: torch.Tensor, k: int) -> list[list[_QueryResult]]:
        bsz = q.size(0)
        q = q.reshape([-1, self._fp_option.dim])
        pwdist = torch.cdist(self.fp_cuda(q.device), q, p=1)  # (n_mols, n_queries)
        dist_t, idx_t = torch.topk(pwdist, k=k, dim=0, largest=False)  # (k, n_queries)
        dist = dist_t.t().reshape([bsz, -1]).cpu().numpy()
        idx = idx_t.t().reshape([bsz, -1]).cpu().numpy()

        results: list[list[_QueryResult]] = []
        for i in range(dist.shape[0]):
            res: list[_QueryResult] = []
            for j in range(dist.shape[1]):
                index = int(idx[i, j])
                res.append(
                    _QueryResult(
                        index=index,
                        molecule=self._molecules[index],
                        fingerprint=self._fp[index],
                        distance=dist[i, j],
                    )
                )
            results.append(res)
        return results
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
if __name__ == "__main__":
    # Later, you can load the searcher without refitting
    loaded_searcher = FingerprintIndex.load("data/processed/fpindex.pkl")
    # Example search
    query_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    fingerprints = compute_fingerprints(tuple([Molecule(query_smiles)]), fp_option=FingerprintOption.morgan_for_building_blocks())
    results = loaded_searcher.query_single(fingerprints.reshape(1, -1), k=5)
    print(f"\nTop 5 similar SMILES to {query_smiles}:")
    for result in results:
        print(result.molecule.smiles)
