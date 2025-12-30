# Binovo
## A novel bi-direction transformer model for denovo protein sequence identification
**Binovo** is a state-of-the-art bi-directional transformer model designed for accurate peptide sequence prediction from DIA (Data-Independent Acquisition) mass spectrometry data. It introduces a novel architecture that combines feature encoding with dual decoding strategies to significantly improve de novo peptide sequencing.

## Installation
Create a new Conda environment named binovo with Python 3.10:
```sh
conda create --name binovo python=3.10
```
Activate the environment:
```sh
conda activate binovo
```
Install PyTorch with CUDA 12.4:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
install NumPy:
```sh
conda install numpy
```
isntall Pyteomics:
```sh
conda install bioconda::pyteomics
```
install Spectrum Utils:
```sh
conda install -c bioconda spectrum_utils
```
install Einops:
```sh
conda install conda-forge::einops
```
## Pretrained model
Binovo requires pretrained models for peptide sequence generation:

- **model_plasma.pth**: Trained on the plasma dataset (as described in the paper)
- **model_oc.pth**: Trained on the OC dataset
- **model_uti.pth**: Trained on the UTI dataset

## Dataset
Feature and spectrum files are provided by **DeepNovo-DIA** and can be downloaded from [MassIVE MSV000082368](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=88f95e8494cc4feeb3610e59f07f1d41).

## Usage
### 1. Preprocess Mass Spectrum Data
You can run the preprocessing script to filter out noisy data and normalize the intensity of the mass spectrum data (mgf format).
```sh
python preprocess.py --infile [ input.mgf ] --outfile [ output.mgf ] 
```
### 2. Train a Model
Train a new transformer model using labeled training and validation datasets:
```sh
python train.py --type train --spect_file [ input.mgf ] --train_file [ train_feature.csv ] --val_file [ val_feature.csv ] --checkpoint_path [ model_save_path.pth ]
```
### 3. Generate Peptide Sequences
Generate peptide sequences using a trained model:
```sh
python train.py --type gen --model_path [ trained_model.pth ] --out [ output.tsv ] --spect_file [ input.mgf ] --test_file [ features.csv ]
```
If you want to evaluate the model, include the correct peptide sequence in the seq column of the feature file.
### 4. Evaluate Model Performance
Evaluate the generated peptide sequences:
```sh
python evaluation.py --in_file [ generated_peptides.tsv ] --ppm [ ppm_value ]
```
