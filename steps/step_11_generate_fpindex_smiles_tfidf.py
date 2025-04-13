import pickle
from pathlib import Path
import click
from synllama.chem.fpindex import FingerprintIndex
from synllama.chem.smiles_tfidf import SmilesSimilaritySearch
from synllama.chem.mol import FingerprintOption
from synllama.chem.matrix import ReactantReactionMatrix

_default_matrix_file = "data/91_rxns/processed/all/reaction_matrix.pkl"
_default_output_dir = "data/91_rxns/rxn_embeddings"
_default_token_list_path = "data/smiles_vocab.txt"

@click.command()
@click.option("--matrix_file", type=click.Path(exists=True, path_type=Path), default=_default_matrix_file)
@click.option("--output_dir", type=click.Path(path_type=Path), default=_default_output_dir)
@click.option("--token_list_path", type=click.Path(exists=True, path_type=Path), default=_default_token_list_path)

def main(matrix_file: Path, output_dir: Path, token_list_path: Path):
    # Load the reactant-reaction matrix
    with open(matrix_file, 'rb') as f:
        reactant_reaction_matrix: ReactantReactionMatrix = pickle.load(f)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to map reaction index to reaction SMARTS
    reaction_smarts_map = {}

    # Iterate over each reaction
    for reaction_idx, reaction in enumerate(reactant_reaction_matrix.reactions):
        # Find reactants that can participate in this reaction
        reactant_indices = reactant_reaction_matrix.matrix[:, reaction_idx].nonzero()[0]
        reactants = [reactant_reaction_matrix.reactants[i] for i in reactant_indices]

        # Generate FingerprintIndex
        fp_option = FingerprintOption.morgan_for_building_blocks()
        fp_index = FingerprintIndex(molecules=reactants, fp_option=fp_option)

        # Save FingerprintIndex
        fp_index_file = output_dir / f"fpindex_{reaction_idx}.pkl"
        fp_index.save(fp_index_file)

        # Generate SmilesSimilaritySearch
        smiles_search = SmilesSimilaritySearch(token_list_path=token_list_path)
        smiles_search.fit(molecules=reactants)

        # Save SmilesSimilaritySearch
        smiles_search_file = output_dir / f"smiles_tfidf_{reaction_idx}.pkl"
        smiles_search.save(smiles_search_file)

        # Map reaction index to reaction SMARTS
        reaction_smarts_map[reaction_idx] = (reaction.smarts, len(reactants))  # Assuming `reaction` has a `smarts` attribute
        print(f"Processed reaction {reaction_idx}: {len(reactants)} reactants")

    # Save the reaction SMARTS map
    smarts_map_file = output_dir / "reaction_smarts_map.pkl"
    with open(smarts_map_file, 'wb') as f:
        pickle.dump(reaction_smarts_map, f)

    print("All reactions processed and SMARTS map saved.")

if __name__ == "__main__":
    main()
