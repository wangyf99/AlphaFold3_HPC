{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;\f1\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Email: noah.herrington@mssm.edu\
\
##################################################\
\
Contained in this gist are three files, besides the README.md and AF3 code (run_alphafold.py & src/):\
\
1) 1: Singularity container for running the AlphaFold3 program\
	Minerva only supports running singularity images (not docker).\
	Converted to from Docker image created from repository downloaded June 11, 2025.\
	Create a copy of this file in your setup directory and gunzip it.\
\
2) fold_input.json: Input JSON file containing basic input information for running AlphaFold3\
	This template is designed only for modeling a single protein monomer\
	with a single ligand. For information on modeling other types of complexes,\
	see: https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#user-provided-ccd\
	!!!NEEDS TO BE MODIFIED BY THE USER!!!\
\
3) input_run_af3_singularity.lsf: LSF submission script. Contains path information for running the \
	singularity container. Adjust resources as needed for larger complexes, especially\
	multimeric complexes.\
	!!!NEEDS TO BE MODIFIED BY THE USER!!!\
\
Steps for running this workflow:\
--------------------------------\
\
1) Upload your AlphaFold3 parameters:\
	- Create a directory somewhere and upload your parameters to it.\
	- The location for this file can be anywhere.\
\
2) Modify fold_input.json:\
	- Create a name for your prediction.\
	- 'numSeeds' controls the number of random seeds used for a prediction (Default: 5 models x 1 seed).\
	- Obtain the amino acid sequence of the protein you wish to model.\
	- Obtain the ligand SMILES of the ligand you wish to model.\
	- Copy and paste the amino acid sequence and SMILES in their proper places in the file.\
\
3) Modify input_run_af3_singularity.lsf:\
	- Replace "COMPLEX" everywhere with some name for your prediction.\
	- Create a new directory (anywhere) to hold your output files.\
	- Define the paths to directories (containing input JSON file; output files; parameters).\
\
4) Run the workflow:\
	- Enter: bsub < run_af3_singularity.lsf\
	- Whole workflow typically takes ~20-30 minutes or less for a monomer:ligand complex, but 2 hours are allowed in the resources.\
	- All output data goes to the output directory you defined.\
	- "_model.cif" is the most confident model. All models, including this one, are stored in the separate seed directories.\
	- "_summary_confidences.json" contains performance information about the most confident model.\
	- Separate "summary_confidences.json" are also stored in the separate seed directories.*\
\
HAPPY FOLDING!!!
\f1\fs26 \
}