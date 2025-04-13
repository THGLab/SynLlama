import pickle, argparse, os, glob, csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from synllama.chem.reaction import Reaction
from synllama.chem.fpindex import FingerprintIndex, compute_fingerprints
from synllama.chem.mol import FingerprintOption, Molecule
import multiprocessing as mp

def arrange_reactants_and_react_synllama(template, reactant_mols):
    rxn = Reaction(template)
    if len(reactant_mols) != rxn.num_reactants:
        return None, False
    if len(reactant_mols) == 1:
        product = rxn(reactant_mols)
        if len(product) == 0:
            return None, False
    elif len(reactant_mols) == 2:
        product = []
        product.extend(rxn([reactant_mols[0], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[0]]))
        if len(product) == 0:
            return None, False
    elif len(reactant_mols) == 3:
        product = []
        product.extend(rxn([reactant_mols[0], reactant_mols[1], reactant_mols[2]]))
        product.extend(rxn([reactant_mols[0], reactant_mols[2], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[0], reactant_mols[2]]))
        product.extend(rxn([reactant_mols[1], reactant_mols[2], reactant_mols[0]]))
        product.extend(rxn([reactant_mols[2], reactant_mols[0], reactant_mols[1]]))
        product.extend(rxn([reactant_mols[2], reactant_mols[1], reactant_mols[0]]))
        if len(product) == 0:
            return None, False
    else:
        return None, False
    return product, True

def filter_raw_output(llama_output, reaction_idx_map):
    
    successful_synthesis = defaultdict(list)
    for product_smiles, example_data in tqdm(llama_output.items()):
        if type(example_data) == str: continue
        for output in example_data:
            if type(output) == str: continue
            try:
                assert 'reactions' in output and 'building_blocks' in output
                reactions = output['reactions']
                building_blocks = output['building_blocks']
                reactant_stack = []
                reaction_strings = []
                reactant_stack.append(product_smiles)
                reaction_strings.append(product_smiles)
            
                for reaction in reactions:
                    assert 'reaction_template' in reaction and 'reactants' in reaction and 'product' in reaction
                    product = reaction['product']
                    assert product in reactant_stack
                    reactant_stack.remove(product)
                    reaction_strings.remove(product)
                    reaction_strings.append(product)
                    template = reaction['reaction_template'].split('<rxn>')[1].split('</rxn>')[0]
                    assert template in reaction_idx_map
                    reaction_strings.append(f"R{reaction_idx_map[template]}")
                    reactants = reaction['reactants']
                    reactants = [reactant.split("<bb>")[-1].split("</bb>")[0] if "<bb>" in reactant else reactant for reactant in reactants]
                    reactant_stack.extend(reactants)
                    reactant_mols = []
                    for reactant in reactants:
                        if reactant == '': continue
                        mol = Molecule(reactant, source="smiles")
                        if not mol.is_valid:
                            raise ValueError(f"Invalid molecule {reactant}")
                        reactant_mols.append(mol)
                        reaction_strings.append(reactant)
                    product_mol = Molecule(product, source="smiles")
                    if not product_mol.is_valid:
                        raise ValueError(f"Invalid molecule {product}")
                    product_from_rxn, matched = arrange_reactants_and_react_synllama(template, reactant_mols)
                    assert matched
                    product_from_rxn = [prod.csmiles for prod in product_from_rxn if prod is not None]
                    assert product_mol.csmiles in product_from_rxn
                
                bbs = []
                for bb in building_blocks:
                    bb_clean = bb.split("<bb>")[-1].split("</bb>")[0]
                    assert bb_clean in reactant_stack
                    reactant_stack.remove(bb_clean)
                    bbs.append(bb_clean)
                
                successful_synthesis[product_smiles].append({
                    "reaction_strings": ";".join(reaction_strings[::-1]),
                    "bbs": bbs,
                })
                    
            except Exception as e:
                continue
    
    return successful_synthesis

def check_bb_in_enamine(args):
    bb, fp_searcher = args
    fingerprints = Molecule(bb).get_fingerprint(FingerprintOption.morgan_for_building_blocks(), as_bitvec=False)
    fp_searched_results = fp_searcher.query_single(np.array([fingerprints]), k=10)
    fp_searched_mols = [result.molecule for result in fp_searched_results]
    return np.max([Molecule(bb).sim(mol, FingerprintOption.morgan_for_tanimoto_similarity()) for mol in fp_searched_mols])

def check_bbs_in_enamine_parallel(bbs, fp_searcher, num_cores):
    
    bb_mols = [Molecule(bb, source="smiles") for bb in bbs]
    fingerprints = compute_fingerprints(bb_mols, FingerprintOption.morgan_for_building_blocks(), batch_size=1024)
    fp_searched_results = fp_searcher.query(fingerprints, k=10)
    bbs_similarity = []
    for bb, result in zip(bbs, fp_searched_results):
        bbs_similarity.append(np.max([Molecule(bb).sim(r.molecule, FingerprintOption.morgan_for_tanimoto_similarity()) for r in result]))

    # Create a dictionary with bb as the key and its similarity score as the value
    bb_similarity_dict = {bb: similarity for bb, similarity in zip(bbs, bbs_similarity)}
    return bb_similarity_dict

def convert_smiles_dict(successful_synthesis, save_folder, file_name, fp_searcher, num_cores):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # collect all bbs to search in enamine
    all_bbs = []
    for _, value in successful_synthesis.items():
        for v in value:
            all_bbs.extend(v['bbs'])
    all_bbs = list(set(all_bbs))
    bb_similarity_dict = check_bbs_in_enamine_parallel(all_bbs, fp_searcher, num_cores)
    
    for _, value in successful_synthesis.items():
        for v in value:
            v['bbs_similarity'] = [bb_similarity_dict[bb] for bb in v['bbs']]
            v['bbs_not_in_enamine'] = [bb for bb, sim in zip(v['bbs'], v['bbs_similarity']) if sim < 1]
            v['bbs_in_enamine'] = [bb for bb, sim in zip(v['bbs'], v['bbs_similarity']) if sim == 1]
    # save the successful_synthesis to a pickle file
    with open(os.path.join(save_folder, f"{file_name}_successful_synthesis.pkl"), "wb") as f:
        pickle.dump(successful_synthesis, f)
    
    all_bbs_not_in_enamine = []
    all_bbs_in_enamine = []
    for key, value in successful_synthesis.items():
        for v in value:
            bbs_not_in_enamine = v['bbs_not_in_enamine']
            all_bbs_not_in_enamine.extend(bbs_not_in_enamine)
            bbs_in_enamine = v['bbs_in_enamine']
            all_bbs_in_enamine.extend(bbs_in_enamine)
    
    smiles_list = list(set(all_bbs_not_in_enamine))
    file_count = 1
    for i in range(0, len(smiles_list), 10000):
        with open(os.path.join(save_folder, f"{file_name}_successful_bbs_not_in_enamine_{file_count}.txt"), "w") as f:
            f.write("\n".join(smiles_list[i:i+10000]))
        file_count += 1
    
    smiles_list = list(set(all_bbs_in_enamine))
    file_count = 1
    for i in range(0, len(smiles_list), 10000):
        with open(os.path.join(save_folder, f"{file_name}_successful_bbs_in_enamine_{file_count}.txt"), "w") as f:
            f.write("\n".join(smiles_list[i:i+10000]))
        file_count += 1
    
def calc_benchmark_rxn(llama_output, reaction_idx_map):
    # check the correctness in general with rdkit functions
    total_trials = len(llama_output) * len([k for k in llama_output.values() if type(k) != str][0])
    successful_trials = 0
    template_obedience = defaultdict(int)
    reactant_matched = defaultdict(int)
    product_obedience = defaultdict(int)
    bb_obedience = defaultdict(list)
    total_reactions = 0
    successful_reactions = 0
    product_not_in_reactant_stack = 0
    invalid_smiles = []
    total_molecules = 0
    failed_structured_output = []
    template_no_rxn_tag = 0
    failed_cases = [] # template, reactants, product
    total_success_formats = defaultdict(int)
    total_success_reactions = defaultdict(int)

    for product_smiles, example_data in tqdm(llama_output.items()):
        if type(example_data) == str: 
            print(example_data)
            continue
        format_success = True
        reaction_success = True
        for output in example_data:
            if 'reactions' not in output or 'building_blocks' not in output:
                failed_structured_output.append(output)
                format_success = False
                continue
            reactions = output['reactions']
            building_blocks = output['building_blocks']
            reactant_stack = []
            reactant_stack.append(product_smiles)
            successful_trials += 1
            
            for reaction in reactions:
                # Extract the reaction template between <rxn> tags
                if 'reaction_template' not in reaction or 'reactants' not in reaction or 'product' not in reaction:
                    successful_trials -= 1
                    format_success = False
                    failed_structured_output.append(reaction)
                    break
                try:
                    template = reaction['reaction_template'].split('<rxn>')[1].split('</rxn>')[0]
                except:
                    template_no_rxn_tag += 1
                    successful_trials -= 1
                    format_success = False
                    break
                total_reactions += 1
                if template not in reaction_idx_map:
                    template_obedience['not_in_template'] += 1
                    print(f"Template {template} not found in reaction_templates")
                    failed_cases.append((reaction, "template"))
                    format_success = False
                    continue
                else:
                    template_obedience[template] += 1
                # check if reactants can form the product through the reaction template
                reactants = reaction['reactants']
                reactants = [reactant.split("<bb>")[-1].split("</bb>")[0] if "<bb>" in reactant else reactant for reactant in reactants]
                reactant_stack.extend(reactants)
                product = reaction['product']
                if product not in reactant_stack:
                    product_not_in_reactant_stack += 1
                else:
                    reactant_stack.remove(product)
                reactant_mols = []
                total_molecules += len(reactants)
                for reactant in reactants:
                    if not Molecule(reactant, source="smiles").is_valid:
                        invalid_smiles.append(reactant)
                    elif reactant == '':
                        total_molecules -= 1
                        continue
                    else:
                        reactant_mols.append(Molecule(reactant, source="smiles"))
                product_mol = Molecule(product, source="smiles")
                total_molecules += 1
                if not product_mol.is_valid:
                    invalid_smiles.append(product)
                product_from_rxn, matched = arrange_reactants_and_react_synllama(template, reactant_mols)
                if template in reaction_idx_map:
                    reactant_matched[template] += int(matched)
                if product_from_rxn is None:
                    failed_cases.append((reaction, "reactants"))
                    print(f"Reactants {reactants} cannot react through template {template}")
                    reaction_success = False
                    continue
                product_from_rxn = [prod.csmiles for prod in product_from_rxn]
                successful_reactions += 1
                if not product_mol.is_valid or product_mol.csmiles not in product_from_rxn:
                    failed_cases.append((reaction, "product"))
                    print(f"Product {product} not in product_from_rxn {product_from_rxn}")
                    reaction_success = False
                else:
                    product_obedience[template] += 1
            
            bb_count = 0
            for bb in building_blocks:
                bb_clean = bb.split("<bb>")[-1].split("</bb>")[0]
                if bb_clean not in reactant_stack:
                    print(f"Building block {bb_clean} not found in reactant stack")
                    format_success = False
                else:
                    bb_count += 1
            if len(building_blocks) > 0:
                bb_obedience[product_smiles].append(bb_count / len(building_blocks))
            else:
                bb_obedience[product_smiles].append(1)
        
        bb_obedience[product_smiles] = sum(bb_obedience[product_smiles]) / len(bb_obedience[product_smiles]) if len(bb_obedience[product_smiles]) > 0 else 1
        total_success_formats[product_smiles] = format_success
        total_success_reactions[product_smiles] = reaction_success and format_success
    
    stats_rxn = {
        "total_trials": total_trials,
        "failed_structured_output": len(failed_structured_output),
        "template_no_rxn_tag": template_no_rxn_tag,
        "valid_responses": round(successful_trials / total_trials * 100, 2),
        "valid_smiles": round((1 - len(invalid_smiles) / total_molecules) * 100, 2),
        "recycled_bbs": round(sum(bb_obedience.values()) / len(bb_obedience) * 100, 2),
        "template_memorization": round((1 - template_obedience['not_in_template'] / total_reactions) * 100, 2),
        "matched_reactants": round(successful_reactions / total_reactions * 100, 2),
        "good_products": round(sum(product_obedience.values()) / successful_reactions * 100, 2),
        "total_success_formats": round(sum(total_success_formats.values()) / len(llama_output) * 100, 2),
        "total_success_reactions": round(sum(total_success_reactions.values()) / len(llama_output) * 100, 2),
    }
    return stats_rxn, failed_structured_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_folder", type=str, default = "../synllama-data/results/table_2_syn_planning_91rxns")
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--raw_output_only", action="store_true")
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument("--rxn_mapping_path", type=str, default="../synllama-data/inference/reconstruction/91rxns/rxn_embeddings/reaction_smarts_map.pkl")
    parser.add_argument("--fp_searcher_path", type=str, default="../synllama-data/inference/reconstruction/91rxns/processed/fpindex.pkl")
    args = parser.parse_args()
    
    reaction_smarts_dict = pickle.load(open(args.rxn_mapping_path, "rb"))
    reaction_idx_map = {v[0]: k for k, v in reaction_smarts_dict.items()}
    if args.save_folder is None:
        args.save_folder = os.path.join(args.llama_folder, "synllama_reconstruct")
    os.makedirs(args.save_folder, exist_ok=True)
    fp_searcher = FingerprintIndex.load(args.fp_searcher_path)
    
    file_list = glob.glob(os.path.join(args.llama_folder, "*.pkl"))
    all_data = []
    for file in file_list:
        file_name = file.split("/")[-1][:-4]
        with open(file, "rb") as f:
            llama_output = pickle.load(f)
        if not args.raw_output_only:
            stats_rxn, failed_cases = calc_benchmark_rxn(llama_output, reaction_idx_map)
            combined_stats = {**stats_rxn}
            combined_stats['file_name'] = file_name  # Add the file name as an index
            # with open(f"{args.llama_folder}/failed_cases_{file_name}.pkl", "wb") as f:
            #     pickle.dump(failed_cases, f)
            all_data.append(combined_stats)
        if not args.benchmark_only:
            successful_synthesis = filter_raw_output(llama_output, reaction_idx_map)
            if not args.raw_output_only:
                combined_stats['total_success_molecules'] = len(successful_synthesis)
            convert_smiles_dict(successful_synthesis, args.save_folder, file_name, fp_searcher, mp.cpu_count() // 2)

    if not args.raw_output_only:
        all_keys = ['file_name', 'total_trials', 'valid_responses', 'template_memorization', 'recycled_bbs', 'valid_smiles', 'matched_reactants', 'good_products', 'total_success_formats', 'total_success_reactions']
        if not args.benchmark_only:
            all_keys.append('total_success_molecules')
        df = pd.DataFrame(all_data, columns=all_keys)
        df.sort_values(by="file_name", ascending=True, inplace=True)
        df.to_csv(os.path.join(args.save_folder, "llm_benchmark_stats.csv"), index=False)

if __name__ == "__main__":
    main()

