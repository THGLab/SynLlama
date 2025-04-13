import dataclasses
import itertools
import random
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from synllama.chem.matrix import ReactantReactionMatrix
from synllama.chem.mol import Molecule
from synllama.chem.reaction import Reaction
from synllama.chem.smiles_tfidf import sort_by_similarity

_NumReactants: TypeAlias = int
_MolOrRxnIndex: TypeAlias = int
_TokenType: TypeAlias = tuple[_NumReactants, _MolOrRxnIndex]


def _flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from _flatten(el)
        else:
            yield el


@dataclasses.dataclass
class _Node:
    mol: Molecule
    rxn: Reaction | None
    token: _TokenType
    children: list["_Node"]

    def to_str(self, depth: int) -> str:
        pad = " " * depth * 2
        lines = [f"{pad}{self.mol.smiles}"]
        if self.rxn is not None:
            for c in self.children:
                lines.append(f"{c.to_str(depth + 1)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Node(\n{self.to_str(1)}\n)"


class Stack:
    def __init__(self) -> None:
        super().__init__()
        self._mols: list[Molecule] = []
        self._rxns: list[Reaction | None] = []
        self._tokens: list[_TokenType] = []
        self._stack: list[set[Molecule]] = []

    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    @property
    def tokens(self) -> tuple[_TokenType, ...]:
        return tuple(self._tokens)

    def get_top(self) -> set[Molecule]:
        return self._stack[-1]

    def get_second_top(self) -> set[Molecule]:
        return self._stack[-2]
    
    def get_third_top(self) -> set[Molecule]:
        return self._stack[-3]

    def push_mol(self, mol: Molecule, index: int) -> None:
        self._mols.append(mol)
        self._rxns.append(None)
        self._tokens.append((-1, index))
        self._stack.append({mol})

    def push_rxn(self, rxn: Reaction, index: int, product_limit: int | None = None, product_template: str | None = None) -> bool:
        if len(self._stack) < rxn.num_reactants:
            return False

        prods: list[Molecule] = []
        if rxn.num_reactants == 1:
            for r in self.get_top():
                prods += rxn([r])
        elif rxn.num_reactants == 2:
            for r1, r2 in itertools.product(self.get_top(), self.get_second_top()):
                if product_limit is not None and len(prods) >= product_limit:
                    break
                prods += rxn([r1, r2]) + rxn([r2, r1])
        elif rxn.num_reactants == 3:
            for r1, r2, r3 in itertools.product(self.get_top(), self.get_second_top(), self.get_third_top()):
                if product_limit is not None and len(prods) >= product_limit:
                    break
                prods += (
                    rxn([r1, r2, r3])
                    + rxn([r1, r3, r2])
                    + rxn([r2, r1, r3])
                    + rxn([r2, r3, r1])
                    + rxn([r3, r2, r1])
                    + rxn([r3, r1, r2])
                )
        else:
            return False

        if len(prods) == 0:
            return False
        prod_dict = {m.smiles: m for m in prods}
        if product_template is not None:
            prod_sorted = sort_by_similarity(product_template, list(prod_dict.keys()))
            prod_sorted = [prod_dict[p] for p in prod_sorted]
        else:
            prod_sorted = prods
        if product_limit is not None:
            prod_sorted = prod_sorted[:product_limit]
        prod: Molecule = prod_sorted[0] if product_template is not None else random.choices(prod_sorted, weights=[len(p.smiles) for p in prod_sorted])[0]

        self._mols.append(prod) # need to look into this step for why there's wrong molecules, the prod here is not necessarily the product of the reaction
        self._rxns.append(rxn)
        self._tokens.append((rxn.num_reactants, index))
        for _ in range(rxn.num_reactants):
            self._stack.pop()
        self._stack.append(set([prod]))
        return True

    def get_tree(self) -> _Node:
        stack: list[_Node] = []
        for i in range(len(self._tokens)):
            token = self._tokens[i]
            n_react = token[0]
            if n_react > 0:
                item = _Node(self._mols[i], self._rxns[i], token, [])
                for _ in range(n_react):
                    item.children.append(stack.pop())
                stack.append(item)
            else:
                stack.append(_Node(self._mols[i], self._rxns[i], token, []))
        return stack[-1]

    def get_postfix_tokens(self) -> tuple[_TokenType, ...]:
        return tuple(self._tokens)

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> Molecule:
        return self._mols[index]

    def get_mol_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] == 0 else None for t in self.tokens]

    def get_rxn_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] > 0 else None for t in self.tokens]

    def count_reactions(self) -> int:
        cnt = 0
        for rxn in self._rxns:
            if rxn is not None:
                cnt += 1
        return cnt

    def get_state_repr(self) -> str:
        rl: list[str] = []
        for s in self._stack:
            sl = list(map(lambda m: m.smiles, s))
            sl.sort()
            rl.append(",".join(sl))
        return ";".join(rl)

    def get_action_string(self, delim: str = ";") -> str:
        tokens: list[str] = []
        for mol, (num_reactants, idx) in zip(self._mols, self._tokens):
            if num_reactants == -1:
                tokens.append(f"{mol.smiles}")
            else:
                tokens.append(f"R{idx}_{num_reactants}")
                tokens.append(f"{mol.smiles}")
        return delim.join(tokens)
    
    def get_source(self, delim: str = ";") -> str:
        return delim.join([mol.source for mol in self._mols])

def create_init_stack(matrix: ReactantReactionMatrix, rxn_count: dict[str, int], weighted_ratio: float = 0.0, prob_u_fp: str = None) -> Stack:
    stack = Stack()
    prob_u = np.ones(len(matrix.reactant_count)) / len(matrix.reactant_count)
    if prob_u_fp is not None:
        with open(prob_u_fp, "r") as f:
            prob_u = np.array(list(map(float, f.read().splitlines())), dtype=np.float32)
        prob_u = prob_u / prob_u.sum()
    prob_w = list(rxn_count.values())
    prob_w = np.array([prob_w[i] if i in matrix.seed_reaction_indices else 0 for i in range(len(matrix.reactant_count))], dtype=np.float32)
    prob_w = prob_w / prob_w.sum()
    prob = weighted_ratio * prob_w + (1 - weighted_ratio) * prob_u
    prob = prob / np.sum(prob)
    rxn_index: int = np.random.choice(np.arange(len(matrix.reactions)), p=prob)
    rxn_col = matrix.matrix[:, rxn_index]
    rxn = matrix.reactions[rxn_index]

    if rxn.num_reactants == 2:
        m1 = np.random.choice(np.bitwise_and(rxn_col, 0b01).nonzero()[0])
        m2 = np.random.choice(np.bitwise_and(rxn_col, 0b10).nonzero()[0])
        if random.randint(0, 1) % 2 == 1:
            m1, m2 = m2, m1
        stack.push_mol(matrix.reactants[m1], m1)
        stack.push_mol(matrix.reactants[m2], m2)
    elif rxn.num_reactants == 1:
        m = np.random.choice(rxn_col.nonzero()[0])
        stack.push_mol(matrix.reactants[m], m)
    elif rxn.num_reactants == 3:
        m1 = np.random.choice(np.bitwise_and(rxn_col, 0b001).nonzero()[0])
        m2 = np.random.choice(np.bitwise_and(rxn_col, 0b010).nonzero()[0])
        m3 = np.random.choice(np.bitwise_and(rxn_col, 0b100).nonzero()[0])
        m1, m2, m3 = random.sample([m1, m2, m3], 3)
        stack.push_mol(matrix.reactants[m1], m1)
        stack.push_mol(matrix.reactants[m2], m2)
        stack.push_mol(matrix.reactants[m3], m3)

    stack.push_rxn(rxn, rxn_index)
    return stack


def expand_stack(stack: Stack, matrix: ReactantReactionMatrix):
    matches = matrix.reactions.match_reactions(random.choice(list(stack.get_top())))
    if len(matches) == 0:
        return stack, False
    rxn_index = random.choice(list(matches.keys()))
    reactant_flag = 1 << matches[rxn_index][0]

    rxn_col = matrix.matrix[:, rxn_index]
    if np.any(rxn_col >= 4):
        # Case of tri-mol reaction
        all_reactants = 0b111
        remaining_reactants = all_reactants ^ reactant_flag
        reactant_1 = remaining_reactants & 0b001  # Isolate the 001 bit
        reactant_2 = remaining_reactants & 0b010  # Isolate the 010 bit
        reactant_3 = remaining_reactants & 0b100  # Isolate the 100 bit
        valid_reactants = [reactant for reactant in [reactant_1, reactant_2, reactant_3] if reactant != 0]
        s_indices_1 = np.logical_and(rxn_col != 0, (rxn_col & valid_reactants[0]) == valid_reactants[0]).nonzero()[0]
        s_indices_2 = np.logical_and(rxn_col != 0, (rxn_col & valid_reactants[1]) == valid_reactants[1]).nonzero()[0]
        s_indices_1, s_indices_2 = random.sample([s_indices_1, s_indices_2], 2)
        s_index1 = np.random.choice(s_indices_1)
        stack.push_mol(matrix.reactants[s_index1], s_index1)
        s_index2 = np.random.choice(s_indices_2)
        stack.push_mol(matrix.reactants[s_index2], s_index2)
        rxn_success = stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
    else:
        s_indices = np.logical_and(rxn_col != 0, rxn_col != reactant_flag).nonzero()[0]
        if len(s_indices) == 0:
            stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
            return stack, True
        s_index = np.random.choice(s_indices)
        stack.push_mol(matrix.reactants[s_index], s_index)
        rxn_success = stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
        if not rxn_success:
            # pop the last pushed molecule and end the reaction
            stack._mols.pop()
            stack._rxns.pop()
            stack._tokens.pop()
            stack._stack.pop()
    return stack, rxn_success


def create_stack(
    matrix: ReactantReactionMatrix,
    rxn_count: dict[str, int],
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    prob_u_fp: str = None,
) -> Stack:
    stack = create_init_stack(matrix, rxn_count=rxn_count, weighted_ratio=init_stack_weighted_ratio, prob_u_fp=prob_u_fp)
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if not changed:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break
    return stack


def create_stack_step_by_step(
    matrix: ReactantReactionMatrix,
    rxn_count: dict[str, int],
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
    prob_u_fp: str = None,
) -> Iterable[Stack]:
    stack = create_init_stack(matrix, rxn_count=rxn_count, weighted_ratio=init_stack_weighted_ratio, prob_u_fp=prob_u_fp)
    yield stack
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if changed:
            yield stack
        else:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break