# ProteomicSignaturePred
Machine Learning Based Prediction of Proteomic Signatures using  H\&amp;E Histology and Pathomic Features.

The pipeline performs the following steps:

Stain deconvolution on whole slide images (WSI) to separate hematoxylin and eosin stains.
Thresholding and cleaning of the hematoxylin channel to create a binary mask.
Nuclei segmentation on the DAPI channel to create a binary mask.
Registration of the H&E and DAPI masks.
Feature extraction from the CODEX channels for each segmented nucleus.
Dimensionality reduction and clustering of the feature vectors.
Analysis of cluster characteristics and intra-cluster variability.

![Alt Text](/images/diagrams/HighLevelView.svg)

![Alt Text](/images/results/Dir_gen.png)


# Direction 1:
• HE nuclei-level is noisy.

• Exploring whether we can **aggregate** CODEX data on additional levels of abstraction (e.g., an functional tissue unit with glom and 1st order neighbor tubules.)

• We want to **find **signal between HE and CODEX. 

![Alt Text](/images/results/Dir1.png)

![Alt Text](/images/results/Dir1_tubules.png)

picture

# Direction 2:
• We want to study methodology for **fusing** the histology and proteomic imaging data.

• Independently from nuclei, glom, FTU’s, and neighborhoods, we want to integrate features from histology onto CODEX optimaly.

• We want to **see** how the H&E modality affects CODEXUMAP clusters.

![Alt Text](/images/results/HEWeighted.png)

![Alt Text]/images/results/(Dir2.png)

# Structural Cytometry Interface

![Alt Text](/images/results/StructureCytometry.png)

# Dependencies
Linux

Python 3.7

TensorFlow 2.7

OpenCV

Tiffslide

shapely

skimage

# Usage

Clone the repository:

git clone https://github.com/maragall/ProteomicSignaturePred.git

Install the required dependencies:
pip install -r requirements.txt

Prepare your input data:
Place your WSI images in the data/wsi directory.
Place their corresponding CODEX pair images in the data/codex directory.

Run the pipeline:
BaselinePipeline.py

The results will be saved in the results directory:
results/masks: Binary masks for H&E and DAPI.
results/features.csv: Extracted features for each nucleus.
results/umap.png: UMAP projection of the feature vectors.
results/clusters.png: Hierarchical clustering of the feature vectors.
results/analysis.csv: Cluster analysis results.

# SLURMScripts
Please see runPipelineSlurm.sh to run using SLURM on Compute Cluster

# Contact
Please contact me for any additional questions: j.maragall@ufl.edu
