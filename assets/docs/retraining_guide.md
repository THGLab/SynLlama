## 🔍 Retraining Guide

This section provides a guide for retraining the SynLlama model. You will first need to generate your fine-tuning data by first accessing Enamine BBs and then generating synthetic pathways. After that, you can perform supervised fine-tuning with the Axolotl package.

### 📦 Enamine Synthetic Pathway Generation

**Step 1:** Since Enamine BBs are not publicly available, you will need to access them first. Please refer to the [Enamine BBs](https://enamine.net/building-blocks/building-blocks-catalog) and follow the necessary steps to create an account and download the BBs from the **US Stock**. After downloading the BBs, you can place the file under the `data/` directory and run the following command to prepare the BBs for the pathway generation. We still use the 91 reaction templates from the original SynLlama paper as an example:

```bash
cd SynLlama
python steps/step_10_calc_embedding.py \
    --data_folder data/91_rxns \
    --bb_path ENAMINE_FILE_PATH \ # replace with your downloaded BBs file path
    --rxn_template_path data/91_rxns/91_rxn_templates.txt \
    --testing_data_path TESTING_DATA_PATH # don't need this if you don't have a predefined testing .smi file
```

After this step, you will have a `data/91_rxns/processed` folder containing the reaction matrices for pathway generation. 

**Step 2: [Optional]** If you are downloading the most updated Enamine BBs, you will have more than 230k BBs as specified in the paper. Therefore, you should calculate all the reaction embeddings with your new BBs with the following command:

```bash
python steps/step_11_generate_fpindex_smiles_tfidf.py \
    --matrix_file data/91_rxns/processed/all/reaction_matrix.pkl \
    --output_dir data/91_rxns/rxn_embeddings \
    --token_list_path data/smiles_vocab.txt
```

After this step, you will have a `data/91_rxns/rxn_embeddings` folder containing the reaction embeddings for inference. In this case, you don't need to download the figshare data as specified in the [Inference Guide](inference_guide.md). 

**Step 3:** Finally, you can generate your fine-tuning data in [alpaca format](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/inst_tune.html#alpaca) with the following command:

```bash
python steps/step_20_generate_reactions.py \
    --matrix_path data/91_rxns/processed/train/reaction_matrix_train.pkl \ # change it to testing/all reaction matrix file if needed
    --rxn_mapping_path data/91_rxns/rxn_embeddings/reaction_smarts_map.pkl \
    --num_reactions NUM_REACTIONS \ # replace with your desired number of reactions
    --name NAME \ # replace with your desired name
```

This step will generate a `data/NAME.jsonl` file containing the fine-tuning data, which will be used for the next step.

### 📦 Supervised Fine-Tuning (SFT)

Here, we provide instructions to reproduce the fine-tuning results in the paper using a package called [Axolotl](https://github.com/axolotl-ai-cloud/axolotl). Axolotl is a user-friendly tool that simplifies the process of fine-tuning large language models. It provides:

- Easy configuration through YAML files
- Support for multiple model architectures
- Efficient training with various optimization techniques
- Comprehensive documentation and examples

#### Installation

For detailed instructions on fine-tuning the model, please refer to the [Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl). We strongly recommend creating a separate conda environment for fine-tuning to avoid dependency conflicts. Please follow the installation and usage guides in the Axolotl repository to fine-tune the model for your specific needs. Make sure to activate your dedicated fine-tuning environment before proceeding to the following steps.

#### Supervised Finetuning

Axolotl uses a configuration file that we provide `synllama_sft.yml` to specify training parameters and data paths. After generating your fine-tuning data following previous steps, to perform SFT, you'll need to update the provided [config file](../../synllama/llm/sft/synllama_sft.yml) with:

- The path to your generated training data
- The path to save the prepared dataset
- The path to save the outputs
- [Optional] The project name and run id for logging ([wandb](https://wandb.ai/site))

Make sure to **review** and **modify** the provided [config file](../../synllama/llm/sft/synllama_sft.yml) according to your specific training requirements before proceeding with the fine-tuning process.

**Step 1:** To preprocess the data before fine-tuning, run the following command:

```bash
source activate axolotl # activate the fine-tuning environment
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.preprocess synllama_sft.yml
```

**Step 2:** To perform supervised finetuning with multiple GPUs, run the following command:

```bash
source activate axolotl # activate the fine-tuning environment
accelerate launch -m axolotl.cli.train synllama_sft.yml
```

**Step 3:** To merge the LoRA weights with the base model, run the following command:

```bash
source activate axolotl # activate the fine-tuning environment
python -m axolotl.cli.merge_lora synllama_sft.yml --lora_model_dir=CHANGE_TO_YOUR_OUTPUT_PATH
```

Once the merging is done, you can use the merged model for inference following the instructions in [Inference Guide](inference_guide.md).