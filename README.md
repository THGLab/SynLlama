# SynLlama: Generating Synthesizable Molecules and Their Analogs with Large Language Models 🧬
[![License](https://img.shields.io/github/license/THGLab/SynLlama)](LICENSE)
[![DOI](https://img.shields.io/badge/ACS%20Central%20Science-10.1021%2Facscentsci.5c01285-003A8F)](https://doi.org/10.1021/acscentsci.5c01285)


## 📖 Overview
![SynLlama](assets/toc.png)
SynLlama is a fine-tuned version of Meta's Llama3 large language models that generates synthesizable analogs of small molecules by creating full synthetic pathways using commonly accessible building blocks and robust organic reaction templates, offering a valuable tool for drug discovery with strong performance in bottom-up synthesis, synthesizable analog generation, and hit expansion.

## 💡 Usage

### Prerequisites
Ensure you have `conda` installed on your system. All additional dependencies will be managed via the `environment.yml` file.

### Installation
To get started with SynLlama, follow these steps:
```bash
git clone https://github.com/THGLab/SynLlama
cd SynLlama
conda env create -f environment.yml
conda activate synllama
pip install -e .
```

### Inference
To perform inference using the already trained SynLlama, download the trained models and relevant files from [here](https://figshare.com/s/39a37d31cea2c190498d) and follow the instructions in the [Inference Guide](assets/docs/inference_guide.md).

### Retraining
If you are interested in retraining the model, please refer to the [Retraining Guide](assets/docs/retraining_guide.md) for detailed instructions.

## 📄 License
See the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments
This project is built on top of the [ChemProjector Repo](https://github.com/luost26/ChemProjector). We thank the authors for building such a user-friendly github!

## 📝 Citation
If you use this code in your research, please cite:

```bibtex
@article{sun_synllama_2025,
	title = {{SynLlama}: {Generating} {Synthesizable} {Molecules} and {Their} {Analogs} with {Large} {Language} {Models}},
	volume = {11},
	copyright = {https://creativecommons.org/licenses/by/4.0/},
	issn = {2374-7943, 2374-7951},
	shorttitle = {{SynLlama}},
	url = {https://pubs.acs.org/doi/10.1021/acscentsci.5c01285},
	doi = {10.1021/acscentsci.5c01285},
	language = {en},
	number = {11},
	urldate = {2026-04-11},
	journal = {ACS Central Science},
	author = {Sun, Kunyang and Bagni, Dorian and Cavanagh, Joseph M. and Wang, Yingze and Sawyer, Jacob M. and Zhou, Bo and Gritsevskiy, Andrew and Zhang, Oufan and Head-Gordon, Teresa},
	month = nov,
	year = {2025},
	pages = {2108--2120},
}
```


