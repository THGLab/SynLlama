# This script is used to reconstruct the raw output from MolPort

import os, sys, argparse, glob
import pickle
import pandas as pd
from collections import defaultdict

# Once the raw output (*_bbs_1.txt & *_successful_synthesis.pkl) is generated, please first go to Molport (https://www.molport.com/shop/swl-step-1)
# to do a "list search" for available building blocks and then run this script to reconstruct the raw output.

# Upon the completion of the Molport search, please download the list search results under the "Selected Items" column.
# This file contains the list of building blocks that are available in the market based on the amount requested.

# Once the file is downloaded, please rename it to "*_molport_ls.xlsx" and put it in the same folder as the raw output files.
# Then, run this script to reconstruct the raw output.

def extract_best_csv(total_reconstruction, save_path):
    df = pd.DataFrame(columns = ["target","smiles","score","synthesis","num_steps","scf_sim","pharm2d_sim","rdkit_sim"])
    for k, v in total_reconstruction.items():
        item = v[0]
        df = pd.concat([df, pd.DataFrame.from_dict({
            "target": k,
            "smiles": k,
            "score": 1.0,
            "synthesis": item['synthesis'],
            "num_steps": sum(["R" in r for r in item['synthesis'].split(";")]),
            "scf_sim": 1.0,
            "pharm2d_sim": 1.0,
            "rdkit_sim": 1.0
            }, orient="index").T])
    df.to_csv(save_path, index=False)
    return df

def find_synllama_reconstruction(success_raw_syn_path, molport_ls_path):
    successful_synthesis = pickle.load(open(success_raw_syn_path, "rb"))
    search_results = pd.read_excel(molport_ls_path)
    found_bbs = search_results['Search Criteria'].tolist()
    enamine_reconstruction = defaultdict(list)
    non_enamine_reconstruction = defaultdict(list)
    total_reconstruction = defaultdict(list)
    total_building_blocks = 0
    non_enamine_building_blocks = 0
    for k, v in successful_synthesis.items():
        for item in v:
            total_building_blocks += len(item['bbs'])
            non_enamine_building_blocks += len(item['bbs_not_in_enamine'])
            if all(bb in found_bbs for bb in item['bbs_not_in_enamine']):
                total_reconstruction[k].append({
                    "bbs": item['bbs'],
                    "synthesis": item['reaction_strings']
                })
                if len(item['bbs_not_in_enamine']) > 0:
                    non_enamine_reconstruction[k].append({
                        "bbs": item['bbs'],
                        "synthesis": item['reaction_strings']
                    })
                else:
                    enamine_reconstruction[k].append({
                        "bbs": item['bbs'],
                        "synthesis": item['reaction_strings']
                    })
    enamine_save_path = success_raw_syn_path.replace("_successful_synthesis.pkl", "_enamine_synllama_reconstruct.csv")
    extract_best_csv(enamine_reconstruction, enamine_save_path)
    non_enamine_save_path = success_raw_syn_path.replace("_successful_synthesis.pkl", "_non_enamine_synllama_reconstruct.csv")
    extract_best_csv(non_enamine_reconstruction, non_enamine_save_path)
    all_synllama_reconstruct_path = success_raw_syn_path.replace("_successful_synthesis.pkl", "_all_synllama_reconstruct.csv")
    extract_best_csv(total_reconstruction, all_synllama_reconstruct_path)
    return len(set(total_reconstruction.keys())), len(set(enamine_reconstruction.keys())), len(set(non_enamine_reconstruction.keys())), total_building_blocks, non_enamine_building_blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_folder", type=str, default="../synllama-data/results/table_2_syn_planning_91rxns")
    args = parser.parse_args()
    
    raw_output_folder = os.path.join(args.llama_folder, "synllama_reconstruct")
    raw_output_files = glob.glob(os.path.join(raw_output_folder, "*_successful_synthesis.pkl"))
    for raw_output_file in raw_output_files:
        molport_ls_path = raw_output_file.replace("_successful_synthesis.pkl", "_molport_ls.xls")
        synllama_reconstruct, enamine_synllama_reconstruct, non_enamine_synllama_reconstruct, total_building_blocks, non_enamine_building_blocks = find_synllama_reconstruction(raw_output_file, molport_ls_path)
        print(f"{raw_output_file} has {synllama_reconstruct} total successful syntheses")
        print(f"{raw_output_file} has {enamine_synllama_reconstruct} enamine successful syntheses")
        print(f"{raw_output_file} has {non_enamine_synllama_reconstruct} non-enamine successful syntheses")
        print(f"{raw_output_file} has {total_building_blocks} total building blocks")
        print(f"{raw_output_file} has {non_enamine_building_blocks} non-enamine building blocks")
        print(f"{raw_output_file} has {(1 - non_enamine_building_blocks / total_building_blocks)*100:.2f}% enamine building blocks")
if __name__ == "__main__":
    main()
