# Preprocessing step 1: Extract metadata and create fingerprint index and reactant reaction matrices
import pathlib, shutil
from sklearn.cluster import KMeans
from synllama.chem.mol import read_mol_file, FingerprintOption
from synllama.chem.fpindex import FingerprintIndex
from synllama.chem.matrix import ReactantReactionMatrix, ReactionContainer
from synllama.chem.reaction import read_reaction_file
from synllama.chem.smiles_tfidf import SmilesSimilaritySearch
import numpy as np
import pickle, click

# As noted in the README, the Enamine data needs to be downloaded separately. If you want to use the default data, 
# please download the data from the following links: https://enamine.net/building-blocks/building-blocks-catalog

# If you want to request these exact files, please contact me at kysun@berkeley.edu or leave an issue on the GitHub repo.

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_253345cmpd_20250212.sdf")
_default_reaction_path = pathlib.Path("data/91_rxns/91_rxn_templates.txt")
_default_data_folder = pathlib.Path("data/91_rxns/")
_default_testing_data_path = pathlib.Path("data/13k_unseen_enamine_bbs.smi")
_random_state = 0
np.random.seed(_random_state) # for reproducibility of the test set bb if no testing data is provided

@click.command()
@click.option(
    "--data_folder",
    type=click.Path(path_type=pathlib.Path),
    default=_default_data_folder,
    help="Path to the data folder."
)
@click.option(
    "--bb_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
    help="Path to the input building blocks SDF file."
)
@click.option(
    "--rxn_template_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_reaction_path,
    help="Path to the input reaction templates file."
)
@click.option(
    "--testing_data_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_testing_data_path,
    help="Path to the testing data file."
)

def run_all_preprocessing(data_folder, bb_path, rxn_template_path, testing_data_path = None):
    processed_folder = data_folder / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)
    testing_folder = data_folder / "testing_data"
    testing_folder.mkdir(parents=True, exist_ok=True)
    molecules = list(read_mol_file(bb_path))
    print(f"Generating fingerprints for {bb_path}")
    generate_morgan_fingerprints(molecules, processed_folder / "fpindex.pkl", processed_folder / "enamine_metadata.csv")
    # print(f"Generating smiles embedding for {bb_path}")
    # generate_smiles_embedding(molecules, data_folder / "smiles_embedding.pkl", smiles_vocab_path)
    if testing_data_path is None:
        print(f"Clustering fingerprints")
        knn_clustering(processed_folder / "fpindex.pkl", testing_folder, n_clusters=128, random_state=_random_state)
    else:
        shutil.copy(testing_data_path, testing_folder / "test_bb.smi")
    print(f"Creating reactant reaction matrix cache")
    create_reactant_reaction_matrix_cache(molecules, rxn_template_path, processed_folder / "all" / "reaction_matrix.pkl")
    create_reactant_reaction_matrix_cache(molecules, rxn_template_path, processed_folder / "train" / "reaction_matrix_train.pkl", testing_folder / "test_bb.smi")
    create_reactant_reaction_matrix_cache(molecules, rxn_template_path, processed_folder / "test" / "reaction_matrix_test.pkl", testing_folder / "test_bb.smi", test_only=True)

def generate_morgan_fingerprints(molecules, out, meta_data_path):
    """Generate Morgan fingerprints from the specified SDF file and save the FingerprintIndex."""
    # Define the fingerprint option
    fp_option = FingerprintOption.morgan_for_building_blocks()
    if meta_data_path:
        import csv
        with open(meta_data_path, 'w', newline='') as csvfile:
            fieldnames = ['SMILES'] + list(molecules[0].meta_info.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for mol in molecules:
                row = {'SMILES': mol.smiles}
                row.update(mol.meta_info)
                writer.writerow(row)
    # Generate fingerprints
    fp_index = FingerprintIndex(molecules, fp_option)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp_index.save(out)

# don't need this for now because we are using reaction template-based smiles embedding
def generate_smiles_embedding(molecules, out, smiles_vocab_path):
    """Generate smiles embedding from the specified SDF file and save the SmilesSimilaritySearch."""
    smiles_tokens = [line.strip() for line in open(smiles_vocab_path)]
    smiles_searcher = SmilesSimilaritySearch(token_list=smiles_tokens)
    smiles_searcher.fit(molecules, save_ngram=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    smiles_searcher.save(out)

def knn_clustering(fp_index_path, out, n_clusters=128, random_state=_random_state):
    """Find the smallest cluster in the FingerprintIndex and save the SMILES of the molecules in the cluster to the output file."""
    fp_index = FingerprintIndex.load(fp_index_path)
    fp = fp_index._fp
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(fp)
    for i in range(n_clusters):
        print(f"Cluster {i} has {np.sum(kmeans.labels_ == i)} molecules")
        cluster_idx = np.where(kmeans.labels_ == i)[0]
        cluster_smiles = [fp_index.molecules[i].smiles for i in cluster_idx]
        if i == 0:
            cluster_out_path = out / f"test_bb.smi"
            with open(cluster_out_path, "w") as f:
                for smi in cluster_smiles:
                    f.write(smi + "\n")

def create_reactant_reaction_matrix_cache(molecules, reaction_path, cache_path, excl_path = None, test_only= False):
    """Create a reactant reaction matrix cache for reaction generation."""
    rxns = ReactionContainer(read_reaction_file(reaction_path))
    if test_only and excl_path is None:
        raise ValueError("test_only is True but excl_path is None")
    if test_only:
        mols = list(read_mol_file(excl_path))
    else:
        mols = molecules
        if excl_path is not None:
            excl_mols = list(read_mol_file(excl_path))
            excl_smiles = {m.smiles for m in excl_mols}
            mols = [m for m in mols if m.smiles not in excl_smiles]
    m = ReactantReactionMatrix(mols, rxns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(cache_path)

if __name__ == "__main__":
    run_all_preprocessing()
    # old_bbs = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_243540cmpd_20240806.sdf")
    # new_bbs = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_253345cmpd_20250212.sdf")
    # old_mols = list(read_mol_file(old_bbs))
    # new_mols = list(read_mol_file(new_bbs))
    # old_smiles = {m.smiles for m in old_mols}
    # new_smiles_list = []
    # for mol in new_mols:
    #     if mol.smiles not in old_smiles:
    #         new_smiles_list.append(mol.smiles)
    # with open("data/new_test_smiles_list.smi", "w") as f:
    #     for smi in new_smiles_list:
    #         f.write(smi + "\n")
    # molecules = list(read_mol_file("data/new_test_smiles_list.smi"))
    # rxn_template_path = pathlib.Path("data/91_rxns/reaction_templates_hb.txt")
    # processed_folder = pathlib.Path("data/91_rxns/")
    # create_reactant_reaction_matrix_cache(molecules, rxn_template_path, processed_folder / "test" / f"reaction_matrix_test_new_enamine.pkl","data/new_test_smiles_list.smi", test_only=True)