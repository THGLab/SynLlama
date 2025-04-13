import pickle, argparse, copy, glob, os
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from synllama.chem.smiles_tfidf import SmilesSimilaritySearch
from synllama.chem.fpindex import FingerprintIndex
from synllama.chem.mol import FingerprintOption, Molecule
from synllama.chem.reaction import Reaction
from synllama.chem.smiles_tfidf import find_closest_match, string_similarity
from synllama.chem.stack import Stack

def load_results(file_path):
    """Load the results from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def analyze_results(result_file_path, total_num_mols, top_n_rows = 1):
    """Perform analysis on the reconstruction results."""
    file_name = result_file_path[:-4].split("/")[-1]
    results = load_results(result_file_path)
    max_similarity = []
    total_failure_rate = []
    total_reconstruction_rate = []
    failure_rate_within_group = []
    reconstruction_rate_within_group = []
    scf_sim_all = []
    pharm2d_sim_all = []
    rdkit_sim_all = []
    scf_sim_no_reconstruction = []
    pharm2d_sim_no_reconstruction = []
    rdkit_sim_no_reconstruction = []
    average_number_of_steps = []
    morgan_no_reconstruction = []

    max_rows_df = pd.DataFrame()
    
    for df in results:
        # Calculate average maximum similarity
        max_similarity.append(df['score'].max())
        max_row = df.loc[[df['score'].idxmax()]] if top_n_rows == 1 else df.drop_duplicates(subset=['smiles']).nlargest(top_n_rows, 'score')
        # remove response_num column
        if 'response_num' in max_row.columns: max_row = max_row.drop(columns=['response_num'])
        max_rows_df = pd.concat([max_rows_df, max_row])
        failure_rate_within_group.append(df['score'].isna().mean())
        total_failure_rate.append(all(df['score'].isna()))
        # Calculate reconstruction rate (where similarity == 1)
        reconstruction_rate_within_group.append((df['score'] == 1).mean())
        total_reconstruction_rate.append(any(df['score'] == 1))
        scf_sim_all.append(max_row['scf_sim'].values[0] if not max_row['scf_sim'].isna().any() else np.nan)
        pharm2d_sim_all.append(max_row['pharm2d_sim'].values[0] if not max_row['pharm2d_sim'].isna().any() else np.nan)
        rdkit_sim_all.append(max_row['rdkit_sim'].values[0] if not max_row['rdkit_sim'].isna().any() else np.nan)
        synthesis_steps = max_row['num_steps'].values[0]
        average_number_of_steps.append(synthesis_steps)
        if df['score'].max() < 1:
            morgan_no_reconstruction.append(df['score'].max())
            scf_sim_no_reconstruction.append(max_row['scf_sim'].values[0] if not max_row['scf_sim'].isna().any() else np.nan)
            pharm2d_sim_no_reconstruction.append(max_row['pharm2d_sim'].values[0] if not max_row['pharm2d_sim'].isna().any() else np.nan)
            rdkit_sim_no_reconstruction.append(max_row['rdkit_sim'].values[0] if not max_row['rdkit_sim'].isna().any() else np.nan)
    result_file_folder = os.path.dirname(result_file_path)
    max_rows_df.to_csv(os.path.join(result_file_folder, f"{file_name}_enamine_reconstruct.csv"), index=False)
    
    return {
        "file_name": file_name,
        "max_similarity": np.mean(max_similarity),
        "total_failure_rate %": round((1 - (len(results) - np.sum(total_failure_rate)) / total_num_mols) * 100, 2),
        "total_reconstruction_rate %": round((np.sum(total_reconstruction_rate) / total_num_mols) * 100, 2),
        "scf_sim_including_reconstruction": np.nanmean(scf_sim_all),
        "pharm2d_sim_including_reconstruction": np.nanmean(pharm2d_sim_all),
        "avg_rxn_steps": np.nanmean(average_number_of_steps),
        "morgan_no_reconstruction": np.nanmean(morgan_no_reconstruction),
        "scf_sim_no_reconstruction": np.nanmean(scf_sim_no_reconstruction),
        "pharm2d_sim_no_reconstruction": np.nanmean(pharm2d_sim_no_reconstruction),
    }

def similarity_score(product_template, stack_prod_smiles):
    if not Chem.MolFromSmiles(product_template):
        return string_similarity(product_template, stack_prod_smiles)
    else:
        return Molecule(product_template).sim(Molecule(stack_prod_smiles), FingerprintOption.morgan_for_tanimoto_similarity())

def get_top_k_smiles(input_smiles, smiles_searcher, fp_searcher, k=10):
    """
    get the top k smiles from the smiles_searcher and fp_searcher.
    
    Args:
        input_smiles (str): the smiles of the input molecule
        smiles_searcher (SmilesSimilaritySearch): the smiles searcher
        fp_searcher (FingerprintIndex): the fingerprint searcher
        k (int, optional): the number of top smiles to return. Defaults to 10.
    """
    # check if smiles is valid
    input_mol = Chem.MolFromSmiles(input_smiles)
    if input_mol is None:
        searched_smiles = smiles_searcher.query(input_smiles, k=k*2)
        results = [result.molecule.smiles for result in searched_smiles]
        result_mols = [Molecule(s, source="smiles") for s in results]
    else:
        searched_smiles = smiles_searcher.query(input_smiles, k=k)
        results = [result.molecule.smiles for result in searched_smiles]
        result_mols = [Molecule(s, source="smiles") for s in results]
        fingerprints = Molecule(input_smiles).get_fingerprint(FingerprintOption.morgan_for_building_blocks(), as_bitvec=False)
        fp_searched_results = fp_searcher.query_single(np.array([fingerprints]), k=k)
        results.extend([result.molecule.smiles for result in fp_searched_results])
        result_mols.extend([Molecule(s, source="fp") for s in results])
    return list(set(result_mols))

def match_two_reactants(reactant1_list, reactant2_list, rxn, continue_rxn = False):
    valid_combinations = []
    for reactant1 in reactant1_list:
        for reactant2 in reactant2_list:
            reactant_combo1 = [reactant1, reactant2]
            reactant_combo2 = [reactant2, reactant1]
            if rxn(reactant_combo1) or rxn(reactant_combo2):
                if continue_rxn:
                    valid_combinations.append(reactant2)
                else:
                    valid_combinations.append(reactant_combo1)
    return valid_combinations

def match_three_reactants(reactant1_list, reactant2_list, reactant3_list, rxn, continue_rxn = False):
    valid_combinations = []
    for reactant1 in reactant1_list:
        for reactant2 in reactant2_list:
            for reactant3 in reactant3_list:
                reactant_combo1 = [reactant1, reactant2, reactant3]
                reactant_combo2 = [reactant1, reactant3, reactant2]
                reactant_combo3 = [reactant2, reactant1, reactant3]
                reactant_combo4 = [reactant2, reactant3, reactant1]
                reactant_combo5 = [reactant3, reactant1, reactant2]
                reactant_combo6 = [reactant3, reactant2, reactant1]
                if rxn(reactant_combo1) or rxn(reactant_combo2) or rxn(reactant_combo3) or rxn(reactant_combo4) or rxn(reactant_combo5) or rxn(reactant_combo6):
                    if continue_rxn:
                        valid_combinations.append([reactant2, reactant3])
                    else:
                        valid_combinations.append(reactant_combo1)
    return valid_combinations
                
def reconstruct_single_rxn(smiles_to_search, product_template, smiles_searcher, fp_searcher, template, rxn_idx, stacks = None, k=5, n_stacks=25, product_limit = 3):
    """
    Reconstruct a single reaction from a list of building blocks and reactants.
    
    Args:
        smiles_list (list): a list of tuples of (smiles, is_building_block).
        product_template (str): the product template.
        smiles_searcher (SmilesSimilaritySearch): the smiles searcher.
        fp_searcher (FingerprintIndex): the fingerprint searcher.
        template (str): the reaction template.
        rxn_idx (int): the reaction index.
        stack (Stack): the stack to push the reactants.
        k (int, optional): the number of top smiles to return. Defaults to 10.
    """
    # check if reaction template is in the reaction_templates
    rxn = Reaction(template)
    new_stacks = []
    if len(stacks) > 0 and len(stacks[0]) > 0:
        scores = []
        for stack in stacks:
            prev_mol = list(stack.get_top())
            # see how many reactants are needed
            if rxn.num_reactants == 1:
                assert len(smiles_to_search) == 0
                success = stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                if success:
                    new_stacks.append(stack)
            elif rxn.num_reactants == 2:
                assert len(smiles_to_search) == 1
                top_bbs_reactants = get_top_k_smiles(smiles_to_search[0], smiles_searcher, fp_searcher, k)
                valid_mols = match_two_reactants(prev_mol, top_bbs_reactants, rxn, continue_rxn = True)
                for mol in valid_mols:
                    new_stack = copy.deepcopy(stack)
                    new_stack.push_mol(mol, 0)
                    success = new_stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                    if success:
                        scores.append(similarity_score(product_template, new_stack[-1].smiles))
                        new_stacks.append(new_stack)
            elif rxn.num_reactants == 3:
                assert len(smiles_to_search) == 2
                top_bbs_reactants1 = get_top_k_smiles(smiles_to_search[0], smiles_searcher, fp_searcher, k)
                top_bbs_reactants2 = get_top_k_smiles(smiles_to_search[1], smiles_searcher, fp_searcher, k)
                valid_mols = match_three_reactants(prev_mol, top_bbs_reactants1, top_bbs_reactants2, rxn, continue_rxn = True)
                for mol1, mol2 in valid_mols:
                    new_stack = copy.deepcopy(stack)
                    new_stack.push_mol(mol1, 0)
                    new_stack.push_mol(mol2, 0)
                    success = new_stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                    if success:
                        scores.append(similarity_score(product_template, new_stack[-1].smiles))
                        new_stacks.append(new_stack)
    else:
        scores = []
        if rxn.num_reactants == 3:
            assert len(smiles_to_search) == 3
            top_bbs_reactants1 = get_top_k_smiles(smiles_to_search[0], smiles_searcher, fp_searcher, k // 2 + 1)
            top_bbs_reactants2 = get_top_k_smiles(smiles_to_search[1], smiles_searcher, fp_searcher, k // 2 + 1)
            top_bbs_reactants3 = get_top_k_smiles(smiles_to_search[2], smiles_searcher, fp_searcher, k // 2 + 1)
            valid_mols = match_three_reactants(top_bbs_reactants1, top_bbs_reactants2, top_bbs_reactants3, rxn, continue_rxn = False)
            for mol1, mol2, mol3 in valid_mols:
                new_stack = Stack()
                new_stack.push_mol(mol1, 0)
                new_stack.push_mol(mol2, 0)
                new_stack.push_mol(mol3, 0)
                success = new_stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                if success:
                    scores.append(similarity_score(product_template, new_stack[-1].smiles))
                    new_stacks.append(new_stack)
            
        elif rxn.num_reactants == 2:
            assert len(smiles_to_search) == 2
            top_bbs_reactants1 = get_top_k_smiles(smiles_to_search[0], smiles_searcher, fp_searcher, k)
            top_bbs_reactants2 = get_top_k_smiles(smiles_to_search[1], smiles_searcher, fp_searcher, k)
            valid_mols = match_two_reactants(top_bbs_reactants1, top_bbs_reactants2, rxn, continue_rxn=False)
            for mol1, mol2 in valid_mols:
                new_stack = Stack()
                new_stack.push_mol(mol1, 0)
                new_stack.push_mol(mol2, 0)
                success = new_stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                if success:         
                    scores.append(similarity_score(product_template, new_stack[-1].smiles))
                    new_stacks.append(new_stack)
        
        elif rxn.num_reactants == 1:
            assert len(smiles_to_search) == 1
            top_bbs_reactants = get_top_k_smiles(smiles_to_search[0], smiles_searcher, fp_searcher, k)
            for mol in top_bbs_reactants:
                new_stack = Stack()
                new_stack.push_mol(mol, 0)
                success = new_stack.push_rxn(rxn, rxn_idx, product_template=product_template, product_limit=product_limit)
                if success:
                    scores.append(similarity_score(product_template, new_stack[-1].smiles))
                    new_stacks.append(new_stack)
    
    new_stacks = [stack for stack in new_stacks if stack is not None and len(stack) > 0]
    if len(new_stacks) == 0:
        return None
    if len(new_stacks) > n_stacks:
        new_stacks = sorted(new_stacks, key=lambda x: scores[new_stacks.index(x)], reverse=True)[:n_stacks]
    return new_stacks

def reconstruct_all_rxns(output, reaction_idx_map, embedding_path, k, n_stacks):
    """
    Reconstruct all reactions from a list of building blocks and reactants.
    
    Args:
        output (dict): the output from the LLM.
        reaction_idx_map (dict): the reaction idx map.
        embedding_path (str): the path to the smiles embedding.
        k (int, optional): the number of top smiles to return. Defaults to 5.
        n_stacks (int, optional): the number of stacks to return. Defaults to 50.
    """
    if 'reactions' not in output or 'building_blocks' not in output: return None
    building_blocks = [bb.split("<bb>")[-1].split("</bb>")[0] for bb in output['building_blocks']]
    reactions = output['reactions']
    stacks = [Stack()]
    for i, reaction in enumerate(reactions[::-1]):
        if 'reaction_template' not in reaction or 'reactants' not in reaction or 'product' not in reaction: continue
        template = reaction['reaction_template'].split('<rxn>')[1].split('</rxn>')[0]
        if template not in reaction_idx_map:
            template = find_closest_match(template, list(reaction_idx_map.keys()))
        rxn_idx = reaction_idx_map[template]
        reactants = reaction['reactants']
        product_template = reaction['product']
        smiles_to_search = [s for s in reactants if s in building_blocks]
        smiles_searcher = SmilesSimilaritySearch.load(f"{embedding_path}/smiles_tfidf_{rxn_idx}.pkl")
        fp_searcher = FingerprintIndex.load(f"{embedding_path}/fpindex_{rxn_idx}.pkl")
        stacks = reconstruct_single_rxn(smiles_to_search, product_template, smiles_searcher, fp_searcher, template, rxn_idx, stacks, k, n_stacks)
        if stacks is None:
            print(f"Error reconstructing reaction {i}")
            return None
    return stacks

def reaction_scorer(stacks, target_mol, num_calc_extra_metrics: int = 10) -> pd.DataFrame:
    """
    Score the reactions by their similarity to the target molecule.
    
    Args:
        stacks (list[Stack]): the stacks to score.
        target_mol (Molecule): the target molecule.
        num_calc_extra_metrics (int, optional): the number of extra metrics to calculate. Defaults to 10.

    Returns:
        pd.DataFrame: a dataframe with the scores and extra metrics.
    """
    rows: list[dict[str, str | float]] = []
    smiles_to_mol: dict[str, Molecule] = {}
    if not stacks:
        return pd.DataFrame()
    for stack in stacks:
        product_mol = stack[-1]
        rows.append(
            {
                "target": target_mol.smiles,
                "smiles": product_mol.smiles,
                "score": target_mol.sim(product_mol, FingerprintOption.morgan_for_tanimoto_similarity()),
                "synthesis": stack.get_action_string(),
                # "source": stack.get_source(), # for checking the source of the bb generation
                "num_steps": stack.count_reactions(),
            }
        )
        smiles_to_mol[product_mol.smiles] = product_mol
    rows.sort(key=lambda r: r["score"], reverse=True)
    for row in rows[:num_calc_extra_metrics]:
        mol = smiles_to_mol[str(row["smiles"])]
        row["scf_sim"] = target_mol.scaffold.tanimoto_similarity(
            mol.scaffold,
            fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
        )
        row["pharm2d_sim"] = target_mol.dice_similarity(mol, fp_option=FingerprintOption.gobbi_pharm2d())
        row["rdkit_sim"] = target_mol.tanimoto_similarity(mol, fp_option=FingerprintOption.rdkit())
    df = pd.DataFrame(rows)
    return df

def result_generator(smiles, llama_answer, reaction_smarts_dict_path, embedding_path, k, n_stacks, num_calc_extra_metrics=10):
    """
    Generate results by finding top k SMILES strings for building blocks
    and building products from reactants and reaction templates.
    
    Args:
        smiles (str): The product SMILES string.
        llama_answer (dict): The output containing reactants and reaction templates.
        reaction_smarts_dict_path (str): The path to the reaction smarts map.
        embedding_path (str): The path to the smiles embedding.
        k (int, optional): The number of top smiles to return. Defaults to 10.
    """
    reaction_smarts_dict = pickle.load(open(reaction_smarts_dict_path, "rb"))
    reaction_idx_map = {v[0]: k for k, v in reaction_smarts_dict.items()}
    print(f"Loaded reaction smarts dict with {len(reaction_idx_map)} reactions")
    product_mol = Molecule(smiles)
    df_product = pd.DataFrame()
    for i, output in enumerate(llama_answer):
        try:
            stacks = reconstruct_all_rxns(output, reaction_idx_map, embedding_path, k, n_stacks)
            if stacks is None:
                continue
            df = reaction_scorer(stacks, product_mol, num_calc_extra_metrics)
            df_product = pd.concat([df_product, df])
        except Exception as e:
            print(e)
            continue
    print("Finished processing all reactions for " + smiles)
    return df_product.sort_values(by=["score", "rdkit_sim", "scf_sim", "pharm2d_sim"], ascending=[False, False, False, False]).reset_index(drop=True).iloc[:n_stacks] if len(df_product) > 0 else None

def result_generator_wrapper(args):
    """Wrapper function to unpack arguments for result_generator."""
    return result_generator(*args)

def run_enamine_reconstruct(llama_output_path, embedding_path, reaction_smarts_dict_path, save_path, k, n_stacks):
    # Load data
    llama_outputs = pickle.load(open(llama_output_path, "rb"))
    tasks = [(smiles, llama_answer, reaction_smarts_dict_path, embedding_path, k, n_stacks) for smiles, llama_answer in list(llama_outputs.items())]
    
    # Use multiprocessing
    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool:
        # Create a tqdm progress bar
        with tqdm(total=len(tasks)) as pbar:
            results = []
            for result in pool.imap_unordered(result_generator_wrapper, tasks):
                results.append(result)
                pbar.update()
    
    results = [r for r in results if r is not None]
    print(f"Found {len(results)} results")
    pickle.dump(results, open(save_path, "wb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_folder", type=str, default="../synllama-data/results/table_2_syn_planning_91rxns")
    parser.add_argument("--embedding_path", type=str, default="../synllama-data/inference/reconstruction/91rxns/rxn_embeddings")
    parser.add_argument("--n_stacks", type=int, default=25)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--total_num_mols", type=int, default=1000)
    parser.add_argument("--top_n_rows", type=int, default=1)

    args = parser.parse_args()
    args.reaction_smarts_dict_path = os.path.join(args.embedding_path, "reaction_smarts_map.pkl")
    mp.set_start_method('spawn')

    llama_output_paths = glob.glob(os.path.join(args.llama_folder, "*.pkl"))
    if args.save_path is None:
        args.save_path = os.path.join(args.llama_folder, "enamine_reconstruct") 
    os.makedirs(args.save_path, exist_ok=True)
    
    for llama_output_path in llama_output_paths:
        print(f"Processing {llama_output_path}")
        name = llama_output_path[:-4].split("/")[-1]
        save_path = os.path.join(args.save_path, f"{name}.pkl")
        if os.path.exists(save_path):
            print(f"Skipping {name} because it already exists")
            continue
        run_enamine_reconstruct(llama_output_path, args.embedding_path, args.reaction_smarts_dict_path, save_path, args.k, args.n_stacks)
    
    results_folder = os.path.join(args.llama_folder, "enamine_reconstruct")
    results_file_paths = glob.glob(os.path.join(results_folder, "*.pkl"))
    final_df = pd.DataFrame()
    for results_file_path in results_file_paths:
        result = analyze_results(results_file_path, args.total_num_mols, args.top_n_rows)
        df = pd.DataFrame.from_dict(result, orient="index").T
        final_df = pd.concat([final_df, df])
    
    final_df.sort_values(by="file_name", ascending=True, inplace=True)
    final_df.to_csv(os.path.join(results_folder, "enamine_reconstruct_analysis.csv"), index=False)

if __name__ == "__main__":
    main()
