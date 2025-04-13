# This script is used to reconstruct the raw output from MolPort

import os, sys, argparse, glob
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def combine_stats(enamine_reconstruct_csv, all_synllama_reconstruct_csv, non_enamine_synllama_reconstruct_csv, enamine_synllama_reconstruct_csv, total_num_mols, llama_folder = None):
    enamine_reconstruct_df = pd.read_csv(enamine_reconstruct_csv)
    enamine_reconstruct_mol = np.sum(enamine_reconstruct_df.groupby('target')['score'].max() == 1)
    # remove the rows in enamine reconstruct that have the same target as in raw output
    synllama_enamine_reconstruct_df = pd.read_csv(enamine_synllama_reconstruct_csv)
    synllama_all_reconstruct_df = pd.read_csv(all_synllama_reconstruct_csv)
    synllama_non_enamine_reconstruct_df = pd.read_csv(non_enamine_synllama_reconstruct_csv)
    non_enamine_reconstruct_mol = np.sum(synllama_non_enamine_reconstruct_df.groupby('target')['score'].max() == 1)
    
    enamine_all_df = pd.concat([enamine_reconstruct_df, synllama_enamine_reconstruct_df], ignore_index=True)
    enamine_reconstruct_mol = np.sum(enamine_all_df.groupby('target')['score'].max() == 1)
    
    enamine_reconstruct_filtered = enamine_reconstruct_df[~enamine_reconstruct_df['target'].isin(synllama_all_reconstruct_df['target'])]
    combined_df = pd.concat([synllama_all_reconstruct_df, enamine_reconstruct_filtered], ignore_index=True)
    no_recon_combined_df = combined_df[combined_df['score'] < 1]
    combined_stats = {
        "file_name": enamine_reconstruct_csv[:-4].split("/")[-1].split("_enamine_reconstruct")[0],
        "total_failure_rate %": round((1 - (len(combined_df) - np.sum(combined_df['score'].isna())) / total_num_mols) * 100, 2),
        "total_enamine_reconstruct_rate %": round((enamine_reconstruct_mol / total_num_mols) * 100, 2),
        "total_non_enamine_reconstruct_rate %": round((non_enamine_reconstruct_mol / total_num_mols) * 100, 2),
        "total_all_reconstruction_rate %": round((np.sum(combined_df['score'] == 1) / total_num_mols) * 100, 2),
        "morgan_sim": combined_df['score'].mean(),
        "scf_sim": combined_df['scf_sim'].mean(),
        "pharm2d_sim": combined_df['pharm2d_sim'].mean(),
        "avg_rxn_steps": combined_df['num_steps'].mean(),
        "morgan_sim_no_recon": no_recon_combined_df['score'].mean(),
        "scf_sim_no_recon": no_recon_combined_df['scf_sim'].mean(),
        "pharm2d_sim_no_recon": no_recon_combined_df['pharm2d_sim'].mean(),
        "avg_rxn_steps_no_recon": no_recon_combined_df['num_steps'].mean(),
    }
    combined_df.to_csv(os.path.join(llama_folder, f"{file_name}_final_reconstruct_stats.csv"), index=False)
    return combined_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_folder", type=str, default="../synllama-data/results/table_2_syn_planning_91rxns")
    parser.add_argument("--total_num_mols", type=int, default=1000)
    
    args = parser.parse_args()

    enamine_reconstruct_paths = glob.glob(os.path.join(args.llama_folder, "enamine_reconstruct", "*_enamine_reconstruct.csv"))
    final_df = pd.DataFrame()
    for enamine_reconstruct_path in enamine_reconstruct_paths:
        # if you follow the default naming convention, you can use this line
        file_name = enamine_reconstruct_path[:-4].split("/")[-1].split("_enamine_reconstruct")[0]
        enamine_synllama_reconstruct_path = os.path.join(args.llama_folder, "synllama_reconstruct", f"{file_name}_enamine_synllama_reconstruct.csv")
        all_synllama_reconstruct_path = os.path.join(args.llama_folder, "synllama_reconstruct", f"{file_name}_all_synllama_reconstruct.csv")
        non_enamine_synllama_reconstruct_path = os.path.join(args.llama_folder, "synllama_reconstruct", f"{file_name}_non_enamine_synllama_reconstruct.csv")
        result = combine_stats(enamine_reconstruct_path, all_synllama_reconstruct_path, non_enamine_synllama_reconstruct_path, enamine_synllama_reconstruct_path, total_num_mols=args.total_num_mols, llama_folder = args.llama_folder)
        df = pd.DataFrame.from_dict(result, orient="index").T
        final_df = pd.concat([final_df, df])
    final_df.sort_values(by="file_name", ascending=True, inplace=True)
    final_df.to_csv(os.path.join(args.llama_folder, "combined_final_stats.csv"), index=False)
    