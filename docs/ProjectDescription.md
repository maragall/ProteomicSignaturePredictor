**Define Computational Tasks**:
  * **Segmentation of kidney nephron**: Utilize machine learning to accurately segment nephron components from H&E and CODEX stained slides, facilitating detailed analysis of nephron components and disease markers.
  * **Cytometry of neighborhood:** Develop algorithms to classify kidney cell subtypes based on integrated analysis of histological, proteomic, and genomic data, within the first-order glomerular neighborhood.
  * **Prediction of proteomic features**: Create predictive models that leverage multimodal data to forecast proteomic signatures.
* **Set Baselines**:
  * For **segmentation**: Compare againstl U-Net and threshold-based segmentation (Canny) to demonstrate improvements in accuracy and specificity for nephron tissue components.
  * For **Cytometric Classification**: Compare to Louvain Clustering of manifol of extracted features from nucleus space.
  * For **prediction**: Benchmark against biological inference via kidney Atlas.
