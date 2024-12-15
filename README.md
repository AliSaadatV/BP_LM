# Fine-Tuning RNA Language Models to Predict Branch Points
This repository focuses on fine-tuning RNA language models for predicting branch points within intronic sequences. The models are fine-tuned using the [MultiMolecule library](https://multimolecule.danling.org/) and evaluated on experimental datasets.

The following RNA language models were fine-tuned:
- SpliceBERT
- RNABERT
- RNA-FM
- RNA-MSM
- ERNIE-RNA
- UTR-LM

The dataset contains **177980 samples** and is an experimental-data only subset of the dataset used to train [BPHunter](https://www.pnas.org/doi/abs/10.1073/pnas.2211194119?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed).

It has been split into approximately **80/10/10 train/validation/test** by chromosome type:
- Train: `chr1`, `chr2`, `chr3`, `chr4`, `chr5`, `chr6`, `chr7`, `chr12`, `chr13`, `chr14`, `chr15`, `chr16`, `chr17`, `chr18`, `chr19`, `chr20`, `chr21`, `chr22`, `chrX`, `chrY`,
- Validation: `chr9`, `chr10`
- Test: `chr8`, `chr11`

## Repository Structure

The repository is organized as follows:
### Main Code
- `master_training.ipynb`: The main notebook for fine-tuning MultiMolecule RNA language models. Change the model name and hyperparameters in the first cell and set the dataset path in the second cell.

### Scripts
Located in the `scripts/` folder 
  - `compute_metrics.py`: Computes performance metrics during training and testing.
  - `data_preprocessing.py`: Preprares and preeprocesses the dataset.
  - `model_choice.py`: Defines the avaiable models and parameters for fine-tuning.
  - `trainer_dataset_creation.py`: Creates datasets for the HuggingFace trainer.
 
### Supporting Notebooks
- `naive_statistical_approch.ipynb` : Implements a simple statistical model for branch point prediction as a baseline.
- `data_exploration.ipynb`: Analyzes the dataset and visualizes patterns or trends.
- `curve_plotting.ipynb`: Plots training and evaluation metrics (e.g., loss, accuracy).

### Archived Code 
- `archived_code`: Contains early experiments and legacy code: initial training on SpliceBERT and notebooks for early work in the MultiMolecule library.

## Using this Repository
In `master_training.ipynb`, simply change the desired model and hyperparameters in the first cell and the dataset path in the second cell. 

## Credits
This is an ML4Science project done for EPFL's CS433 Machine Learning Course in collaboration with the [Fellay Lab - Human Genomics of Infection and Immunity](https://www.epfl.ch/labs/fellay-lab/).

Contributors (in alphabetical order):
- Pablo Rodenas (EPFL)
- Oliver Smedt (EPFL)
- Timothy Tran (EPFL, University of Washington)

Huge thanks to [Ali Sadaat](https://people.epfl.ch/ali.saadat) who organized this project and was an amazing adivsor for this project.
