# Preprocessing step 2: Generate prompt-response pairs of reactions for LLM fine-tuning.

import pickle, random, json, os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import click
import multiprocessing as mp
from joblib import Parallel, delayed
from collections import defaultdict
from synllama.chem.matrix import ReactantReactionMatrix
from synllama.chem.stack import create_stack
from synllama.chem.reaction import Reaction
from synllama.llm.vars import TEMPLATE, BB_BASE, REACTION_BASE_MAX2, REACTION_BASE_MAX3

def rebuild_response(synthesis_route, rxn_mapping, max_reactants = 3):
    if max_reactants == 2:
        reaction_base = REACTION_BASE_MAX2
    elif max_reactants == 3:
        reaction_base = REACTION_BASE_MAX3
    else:
        raise ValueError(f"Invalid number of reactants: {max_reactants}")
    
    synthesis = synthesis_route.replace("\\", "\\\\").split(";")[::-1] # fix json escaping
    target_smiles = synthesis[0]
    rxn_positions = [i for i, s in enumerate(synthesis) if s.startswith("R")]
    bb_list = [target_smiles]
    rxn_list = []
    
    for j, rxn_pos in enumerate(rxn_positions):
        product = synthesis[rxn_pos-1]
        rxn_idx = j+1
        if j+1 < len(rxn_positions):
            reactants = synthesis[rxn_pos+1:rxn_positions[j+1]]
        else:
            reactants = synthesis[rxn_pos+1:]
        reactants_padded = reactants + [""] * (max_reactants - len(reactants))
        rxn_copy = deepcopy(reaction_base)
        rxn_copy = rxn_copy.replace('PRODUCT', product)
        rxn_copy = rxn_copy.replace("RXN_TEMPLATE", rxn_mapping[int(synthesis[rxn_pos][1:].split("_")[0])])
        rxn_copy = rxn_copy.replace("REACTION_NUM", str(rxn_idx))
        rxn_copy = rxn_copy.replace("REACTANT1", reactants_padded[-1])
        rxn_copy = rxn_copy.replace("REACTANT2", reactants_padded[-2])
        if max_reactants == 3:
            rxn_copy = rxn_copy.replace("REACTANT3", reactants_padded[-3])
        rxn_list.append(rxn_copy)
        bb_list.remove(product)
        bb_list.extend(reactants)

    bb_list_formatted = [deepcopy(BB_BASE).replace("Building_Block", bb) for bb in bb_list]
    
    template_copy = deepcopy(TEMPLATE)
    template_copy['input'] = template_copy['input'].replace("SMILES_STRING", target_smiles)
    template_copy['output'] = template_copy['output'].replace("REACTIONS", ", ".join(rxn_list))
    template_copy['output'] = template_copy['output'].replace("BUILDING_BLOCKS", ", ".join(bb_list_formatted))
    output_dict = json.loads(template_copy['output'])
    return template_copy

def generate_reaction_data(matrix: ReactantReactionMatrix, rxn_mapping, rxn_count, init_stack_weighted_ratio, prob_u_fp, max_num_reactions=5, max_num_atoms=80):
    stack = create_stack(
        matrix,
        rxn_count,
        max_num_reactions=max_num_reactions,
        max_num_atoms=max_num_atoms,
        init_stack_weighted_ratio=init_stack_weighted_ratio,
        prob_u_fp=prob_u_fp,
    )
    rebuilt_response = rebuild_response(stack.get_action_string(), rxn_mapping)
    return rebuilt_response

def generate_reaction_chunk(matrix, rxn_mapping, rxn_count, num_reactions, init_stack_weighted_ratio, prob_u_fp, max_num_reactions=5, max_num_atoms=80):
    reactions_dict = defaultdict(int)
    all_reactions = []
    while len(all_reactions) < num_reactions:
        try:
            stack = create_stack(
                matrix,
                rxn_count,
                max_num_reactions=max_num_reactions,
                max_num_atoms=max_num_atoms,
                init_stack_weighted_ratio=init_stack_weighted_ratio,
                prob_u_fp=prob_u_fp,
            )
            all_reactions.append(rebuild_response(stack.get_action_string(), rxn_mapping))
            rxns = [r for r in stack.get_action_string().split(";") if r.startswith("R")]
            for rxn in rxns:
                reactions_dict[rxn] += 1
        except Exception as e:
            continue
    print(sorted(reactions_dict.items(), key=lambda x: int(x[0][1:])))
    return all_reactions

@click.command()
@click.option("--matrix_path", type=click.Path(exists=True, path_type=Path), required=True, default="data/91_rxns/processed/test/reaction_matrix_test.pkl")
@click.option("--rxn_mapping_path", type=click.Path(exists=True, path_type=Path), required=True, default="data/91_rxns/rxn_embeddings/reaction_smarts_map.pkl")
@click.option("--prob_u_fp", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--init_stack_weighted_ratio", type=float, required=True, default=0.8)
@click.option("--name", default=None)
@click.option("--num_reactions", type=int, required=True, default=2000000)
@click.option("--batch_size", type=int, required=True, default=1000)
@click.option("--write_for_benchmark", is_flag=True)

def main(matrix_path, rxn_mapping_path, prob_u_fp, num_reactions, init_stack_weighted_ratio=0.8, name=None, batch_size=1000, write_for_benchmark=False):
    matrix: ReactantReactionMatrix = ReactantReactionMatrix.load(matrix_path)
    reaction_smarts_dict = pickle.load(open(rxn_mapping_path, "rb"))
    rxn_mapping = {k: v[0] for k, v in reaction_smarts_dict.items()}
    rxn_count = {k: v[1] for k, v in reaction_smarts_dict.items()}
    prob_u_fp = prob_u_fp
    num_cores = mp.cpu_count()
    num_batches = num_reactions // batch_size // num_cores
    remainder = num_reactions - num_batches * batch_size * num_cores
    if name is None:
        name = f"{num_reactions/1000000:.1f}m_reactions"
    
    # check if the file exists
    if os.path.exists(f"data/{name}.jsonl"):
        print(f"File {name}.jsonl already exists. Deleting to start fresh...")
        os.remove(f"data/{name}.jsonl")
    
    for batch_num in range(num_batches):
        with tqdm(total=batch_size, desc=f"Generating reactions batch {batch_num+1} of {num_batches}") as pbar:
            with open(f"data/{name}.jsonl", "a") as f:
                results = Parallel(n_jobs=num_cores)(
                delayed(generate_reaction_chunk)(matrix, rxn_mapping, rxn_count, batch_size, init_stack_weighted_ratio, prob_u_fp)
                for _ in range(num_cores)
                )
                for result in results:
                    for r in result:
                        json.dump(r, f)
                        f.write("\n")

    if remainder > 0:
        with tqdm(total=remainder, desc=f"Generating reactions batch {num_batches+1} of {num_batches}") as pbar:
            results = Parallel(n_jobs=num_cores)(
                delayed(generate_reaction_chunk)(matrix, rxn_mapping, rxn_count, remainder // num_cores + 1, init_stack_weighted_ratio, prob_u_fp)
                for _ in range(num_cores)
            )
            results = [r for rr in results for r in rr if r is not None]
            results = results[:remainder]
            with open(f"data/{name}.jsonl", "a") as f:
                for result in results:
                    json.dump(result, f)
                    f.write("\n")
    
    if write_for_benchmark:
        with open(f"data/{name}.jsonl", "r") as f:
            reactions = [json.loads(line) for line in f]
        reactions_dict = {r['input'].split("SMILES string:")[1].strip(): [json.loads(r['output'])] for r in reactions}
        with open(f"data/{name}_benchmark.pkl", "wb") as f:
            pickle.dump(reactions_dict, f)
        with open(f"data/{name}.smi", "w") as f:
            for r in reactions:
                f.write(r['input'].split("SMILES string:")[1].strip() + "\n")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
