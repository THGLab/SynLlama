TEMPLATE = {
"instruction": "You are an expert synthetic organic chemist. Your task is to design a synthesis pathway for a given target molecule using common and reliable reaction templates and building blocks. Follow these instructions:\n\n1. **Input the SMILES String:** Read in the SMILES string of the target molecule and identify common reaction templates that can be applied.\n\n2. **Decompose the Target Molecule:** Use the identified reaction templates to decompose the target molecule into different intermediates.\n\n3. **Check for Building Blocks:** For each intermediate:\n   - Identify if it is a building block. If it is, wrap it in <bb> and </bb> tags and save it for later use.\n   - If it is not a building block, apply additional reaction templates to further decompose it into building blocks.\n\n4. **Document Reactions:** For each reaction documented in the output, wrap the reaction template in <rxn> and </rxn> tags.\n\n5. **Repeat the Process:** Continue this process until all intermediates are decomposed into building blocks, and document each step clearly in a structured JSON format.",
"input": "Provide a synthetic pathway for this SMILES string: SMILES_STRING",
"output": "{\"reactions\": [REACTIONS], \"building_blocks\": [BUILDING_BLOCKS]}"
}
BB_BASE = "\"<bb>Building_Block</bb>\""
REACTION_BASE_MAX2 = "{\"reaction_number\": REACTION_NUM, \"reaction_template\": \"<rxn>RXN_TEMPLATE</rxn>\", \"reactants\": [\"REACTANT1\", \"REACTANT2\"], \"product\": \"PRODUCT\"}"
REACTION_BASE_MAX3 = "{\"reaction_number\": REACTION_NUM, \"reaction_template\": \"<rxn>RXN_TEMPLATE</rxn>\", \"reactants\": [\"REACTANT1\", \"REACTANT2\", \"REACTANT3\"], \"product\": \"PRODUCT\"}"

sampling_params_frugal = [
    {"temp": 0.1, "top_p": 0.1, "repeat": 1, "name": "frozen"},
    {"temp": 0.6, "top_p": 0.5, "repeat": 1, "name": "low"},
    {"temp": 1.0, "top_p": 0.7, "repeat": 1, "name": "medium"},
    {"temp": 1.5, "top_p": 0.9, "repeat": 1, "name": "high"}
]

sampling_params_greedy = [
    {"temp": 0.1, "top_p": 0.1, "repeat": 1, "name": "frozen"},
    {"temp": 0.6, "top_p": 0.5, "repeat": 2, "name": "low"},
    {"temp": 1.0, "top_p": 0.7, "repeat": 3, "name": "medium"},
    {"temp": 1.5, "top_p": 0.9, "repeat": 4, "name": "high"}
]

sampling_params_frozen_only = [
    {"temp": 0.1, "top_p": 0.1, "repeat": 1, "name": "frozen"}
]

sampling_params_low_only = [
    {"temp": 0.6, "top_p": 0.5, "repeat": 5, "name": "low"}
]

sampling_params_medium_only = [
    {"temp": 1.0, "top_p": 0.7, "repeat": 5, "name": "medium"}
]

sampling_params_high_only = [
    {"temp": 1.5, "top_p": 0.9, "repeat": 5, "name": "high"}
]