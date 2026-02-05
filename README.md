# Motif-DiffMol: Synergistic Local-Global Molecular Modeling via Coarse-Grained Discrete Diffusion


This is the official implementation of the paper **"Motif-DiffMol: Synergistic Local-Global Molecular Modeling via Coarse-Grained Discrete Diffusion"**. We propose a discrete diffusion model based on Motifs, designed for efficient and controllable drug molecule generation and optimization.

<p align="center">
  <img src="./img/motif-diffmol.png" alt="Motif-DiffMol Architecture" width="60%">
  <br>

</p>


##  Quick Start

### 1. Environment Setup
We provide a comprehensive Conda configuration file to streamline the installation process.

```bash
# 1. Create the environment
conda env create -f environment.yaml

# 2. Activate the environment
conda activate motfi-diffmol
```

### 2. Data Preparation (SAFE Dataset)

Motif-DiffMol utilizes the [SAFE (Sequential Attachment-based Fragment Embedding)](https://huggingface.co/datasets/datamol-io/safe-drugs) dataset for training. 
The code is configured to use the SAFE-GPT dataset.

---

##  Training


All training parameters are located in `configs/train.yaml`. You can modify this file directly to suit your hardware resources.

**Key Parameter:**
* **`unit_size`**: **(Core Parameter)** Controls the granularity/length of the Motifs. A larger value prompts the model to focus on larger substructures, while a smaller value focuses on finer atomic combinations.

### Launch Training
Once configured, initiate the training process with the following command:

```bash
python train.py
```

> **Note**: Training logs and checkpoints will be automatically saved to the directory specified by the `hydra` configuration.

---

## Experiments

This framework supports various generative tasks for drug discovery. 

> **Note:** The source code and execution scripts for the experiments listed below will be made fully publicly available upon the acceptance of the manuscript.

### 1. Unconditional Generation (*De Novo*)
Explore the chemical space without any prior constraints.

### 2. Fragment-Constrained Generation
Generate molecules by extending from specific molecular fragments (scaffolds/substructures).

### 3. Property Optimization (PMO)
Generate high-scoring molecules targeted at specific properties (e.g., QED, DRD2 activity), including optimization workflows and evaluation metrics.

### 4. Lead Optimization
Fine-tune a starting hit molecule to improve its properties while maintaining structural similarity.

---
