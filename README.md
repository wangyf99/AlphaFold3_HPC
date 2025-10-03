# AlphaFold3 on Minerva (Singularity Workflow)

This repository contains the setup and workflow for running **AlphaFold3** on **Minerva** using Singularity (not Docker).

---

## Repository Contents (AF3_Materials)

The `AF3_Materials` directory contains the essential files and templates for running AlphaFold3 on Minerva:  

- **Singularity Container (`alphafold3.sif`)**  
  - Prebuilt container for running AlphaFold3.  
  - Minerva only supports **Singularity images** (not Docker).  
  - Converted from a Docker image (as of June 11, 2025).  
  - ⚠️ Due to its size (~14 GB), the container is **not stored in this repository**, but it is available in the `AF3_Materials` directory on Minerva.  
  - To use: copy to your working directory and **gunzip** if needed.  

- **fold_input.json** *(example: `EXAMPLE_SLC6A14.json`)*  
  - Template JSON file containing basic input information for running AlphaFold3.  
  - Configured for **a single protein monomer with one ligand**.  
  - ⚠️ Must be edited by the user to include:  
    - A name for the prediction.  
    - The amino acid sequence of the protein.  
    - The ligand SMILES string.  
  - For other complex types, see:  
    [AlphaFold3 Input Guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#user-provided-ccd)  

- **input_run_af3_singularity.lsf**  
  - LSF batch submission script.  
  - Runs the Singularity container with user-provided input/output paths.  
  - ⚠️ Must be edited by the user:  
    - Replace `"COMPLEX"` with your prediction name.  
    - Point to correct paths for input/output/parameters.  
    - Adjust resources for larger or multimeric complexes.  

- **Files in Singularity Folder.png**  
  - Screenshot showing all files in the **`standard_workflow`** directory.  
  - Provides a reference for how the workflow files should be organized when running jobs.  

- **README.md (this file)**  
  - Documentation of the workflow steps.  

- **AF3 Code**  
  - `run_alphafold.py`  
  - `src/` (supporting code for AlphaFold3 execution).  

---

## Workflow Steps

### 1. Duplicate AF3 Materials
Copy the shared **AF3_Materials** directory to your own scratch folder:
```bash
cp -r /sc/arion/projects/schlea02a/AF3_Materials /sc/arion/scratch/$USER/
```
This ensures you have the necessary container, templates, and scripts in your personal space.

### 2. Upload AlphaFold3 Parameters
- Create a directory for AlphaFold3 parameters.  
- **Recommended location:** `standard_workflow/batch_workflow` inside your working directory.

### 3. Modify `fold_input.json`
- Define a **name** for your prediction.  
- Set the number of seeds:  
  - `numSeeds` = number of random seeds (default: 5 models × 1 seed).  
- Provide inputs:  
  - **Protein sequence** (amino acid FASTA format).  
  - **Ligand SMILES string**.  
- Copy/paste sequence and ligand SMILES into their respective fields.
- If you want to see one running successfully in the singularity case, simply upload the EXAMPLE_SLC6A14.json from the repository to the standard_workflow directory.

### 4. Modify `input_run_af3_singularity.lsf`
- Replace all instances of `COMPLEX` with your prediction name.  
- Create an **output directory** for results. In the sample in this directory, a directory called "out"" was created, which you can see in the file Files in Singularity Folder.png.
- Update paths for:  
  - Input JSON file.  
  - Output directory.  
  - AlphaFold3 parameters.  
- Adjust resource requests for larger/multimeric complexes.
- If you want to see one running successfully in the singularity case, simply upload the input_run_af3_singularity.lsf from the repository to the standard_workflow directory.


### 5. Submit the Job
Run:
```bash
bsub < input_run_af3_singularity.lsf
```

---

## Runtime and Outputs

- Typical runtime:  
  - ~20–30 minutes for a monomer:ligand complex.  
  - 2 hours of resources requested by default.  

- Key output files:  
  - `*_model.cif` → Most confident model.  
  - `*_summary_confidences.json` → Confidence metrics for best model.  
  - Seed directories: contain all models and their individual `summary_confidences.json`.

---

## Notes
- Ensure all file paths are set correctly before submission.  
- Adjust resource requests in `.lsf` for larger/multimeric systems.  
- Store output results in scratch to avoid filling up project space.  

---

🎉 **Happy Folding!** 🎉  
