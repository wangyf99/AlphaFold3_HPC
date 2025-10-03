# AlphaFold3 on Minerva (Singularity Workflow)

This repository contains the setup and workflow for running **AlphaFold3** on **Minerva** using Singularity (not Docker).

---

## Repository Contents

- **Singularity Container**  
  Prebuilt container for running AlphaFold3.  
  - Minerva only supports **Singularity images**.  
  - Converted from a Docker image (as of June 11, 2025).  
  - To use: create a copy in your setup directory and **gunzip** it.

- **fold_input.json**  
  Input JSON template for running AlphaFold3.  
  - Currently configured for **a single protein monomer with one ligand**.  
  - ⚠️ **Must be modified by the user.**  
  - See official documentation for other use cases:  
    [AlphaFold3 Input Guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#user-provided-ccd)

- **input_run_af3_singularity.lsf**  
  LSF job submission script for Minerva.  
  - Runs AlphaFold3 using the Singularity container.  
  - ⚠️ **Must be modified by the user.**  
  - Adjust resource requests for larger or multimeric complexes.

---

## Workflow Steps

### 1. Upload AlphaFold3 Parameters
- Create a directory and upload AlphaFold3 parameters.  
- This location can be anywhere on Minerva.

### 2. Modify `fold_input.json`
- Define a **name** for your prediction.  
- Set the number of seeds:  
  - `numSeeds` = number of random seeds (default: 5 models × 1 seed).  
- Provide inputs:  
  - **Protein sequence** (amino acid FASTA format).  
  - **Ligand SMILES string**.  
- Copy/paste sequence and ligand SMILES into their respective fields.

### 3. Modify `input_run_af3_singularity.lsf`
- Replace all instances of `COMPLEX` with your prediction name.  
- Create an **output directory** for results.  
- Update paths for:  
  - Input JSON file.  
  - Output directory.  
  - AlphaFold3 parameters.

### 4. Submit the Job
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

---

🎉 **Happy Folding!** 🎉  
