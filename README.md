# Unraveling-Exercise-Resilience-Multi-Omics-Meets-Machine-Learning

## Introduction

This repository contains the code and data for the hackathon project "Unraveling Exercise Resilience: Multi-Omics Meets
Machine Learning". The project aims to evaluate the readiness of MoTrPAC data for machine learning, and explore the
relationship between exercise and various omics data using machine learning techniques.

The key challenge was determining which machine learning approach would be most effective given the dataset's complexity, including time-series omics measurements, sex-specific responses, and physiological outcomes.

Why we want to solve this problem:

Understanding how molecular changes (e.g., gene expression, protein levels) relate to exercise-induced physiological adaptations (e.g., fat loss, VO₂max improvement) could bridge the gap between omics data and whole-body health. Successfully modeling these relationships could help predict exercise benefits and inform personalized training strategies.

These ideas were:

1) Leverage the time-course nature of the data (with measurements at 1, 2, 4, and 8 weeks) to build models that predict
   the future molecular state of tissues based on early training responses. This could involve using recurrent neural
   networks (LSTM/GRU) or other time-series forecasting methods.
2) Explore the sexual dimorphism inherent in the dataset by building separate models for male and female rats. The goal
   would be to identify sex-specific molecular patterns that drive the exercise adaptation and possibly predict
   physiological outcomes such as fat loss or VO₂max improvement.
3) Correlate multi-omic profiles with measured physiological phenotypes (e.g., changes in VO₂max, body composition).
   Develop regression models that can predict these outcomes based solely on the molecular data, helping to bridge the
   gap between omics and whole-body physiology.

After some trial-and-error, and a few dead ends, we decided to focus on the third idea. The first idea was too complex,
and the _n_ (number of samples) was too small to build a model that could generalize well. The second idea was too
complex, and much of the statistical analysis that could be done, had already been done. The third idea was the most
straightforward, and the most likely to yield results.

We decided to focus on the following:

- Correlate multi-omic profiles with measured physiological phenotypes (e.g., changes in VO₂max, body composition).
- Develop regression models that can predict these outcomes based solely on the molecular data, helping to bridge the
  gap between omics and whole-body physiology.

## Data

The data used in this project is from the MoTrPAC study, which is a large-scale study of exercise and its effects on
molecular and physiological outcomes. The specific data we used came from the PASS1B-06 sub-study. The MoTrPAC PASS1B-06
study is a multi-omics investigation of exercise training in rats. Adult Fischer 344 (F344) rats (6-month-old) were
either sedentary (0 weeks) or underwent endurance treadmill training for 1, 2, 4, or 8 weeks, after which tissues were
collected 48 hours
post-training [Study Design](https://motrpac.github.io/MotrpacRatTraining6moData/articles/MotrpacRatTraining6moData.html#study-design).
Multiple “omics” assays were performed, including RNA sequencing (transcriptomics), ATAC-seq (chromatin accessibility),
and RRBS (DNA methylation), across a broad set of tissues. Body weight and composition (lean/fat mass) were measured
throughout the study as primary phenotypes. The multi-omics data and phenotypes are publicly available via the MoTrPAC
Data Hub and GEO (accession GSE242358). Given this rich dataset, the goal is to build models that predict body weight
from transcriptomic and epigenomic features. Below we outline a detailed plan, covering tissue selection, data
integration, model types, preprocessing, validation, and confounders.

Data was originally derived from the officially support [
`MoTrPACRatTraining6Mo`](https://github.com/MoTrPAC/MoTrPACRatTraining6Mo) and [
`MoTrPACRatTraining6MoData`](https://github.com/MoTrPAC/MoTrPACRatTraining6MoData) repositories.
The data is also available in the [MoTrPAC Data Portal](https://motrpac-data.org/).

MoTrPAC profiled 18 tissues in total, but not all assays were done in every tissue. We will focus on tissues that are
both biologically relevant to body weight regulation and have all three omics data (RNA-seq, ATAC-seq, RRBS) available (
the main multi-omic subset was 8 tissues ￼).

### Selected Tissues for Body Weight Prediction

- Subcutaneous White Adipose Tissue (WAT-SC) – As the primary fat storage tissue, WAT is directly linked to body weight
  and adiposity. In this study, changes in WAT gene expression showed a positive correlation with changes in body weight
  and fat mass. Notably, 8-week trained rats had significantly lower body weight and fat mass than controls (especially
  in
  males), [indicating WAT remodeling during weight loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC9882056/#:~:text=match%20at%20L1088%20patterns%20depending,Remarkably%2C%20liver).
  Including WAT allows the model to capture signals of fat accumulation or loss that strongly influence total weight.
- Brown Adipose Tissue (BAT) – BAT is important for thermogenesis and energy expenditure, which can affect weight
  gain. While not explicitly highlighted in the MoTrPAC results, BAT activity can counteract weight gain by burning
  calories. We include BAT to capture epigenomic and transcriptomic changes related to metabolic rate and fat burning
  capacity (e.g. Ucp1 and other thermogenic genes).
- Skeletal Muscle (Gastrocnemius, SKM-GN) – Skeletal muscle is a major component of lean body mass and a primary
  target of exercise. Muscle growth or atrophy will alter body weight. Gastrocnemius muscle transcriptomic changes were
  observed with training, and importantly, many differentially expressed genes in muscle correlate with body weight
  changes. For example, exercise-induced genes in SKM-GN were positively correlated with weight change [(MyoD target
  genes were linked to weight gain in responders)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9882056/#:~:text=In%20SKM,The).
  By including muscle, the model can capture lean mass contributions to weight and the molecular signals of muscle
  hypertrophy or atrophy.
- Liver – The liver is central to metabolism (glucose, lipid handling) and can indirectly influence body weight (
  through metabolic rate and fat storage in other tissues). In the MoTrPAC data, the liver exhibited many strong
  correlations between its molecular changes and systemic phenotypes: in fact, liver had the highest frequency of
  [features strongly correlated](https://pmc.ncbi.nlm.nih.gov/articles/PMC9882056/#:~:text=body%20fat%20in%20WAT,proteins%2C%20and%20specifically%20TFs%2C%20with)
  with body fat changes among all tissues. This suggests liver epigenomic/transcriptomic markers (e.g. metabolic
  regulators) could improve weight predictions. We include liver so the model can leverage signals of metabolic state
  that may reflect or predict weight gain (for instance, lipid metabolism genes or methylation of metabolic gene
  promoters).
- Heart – Although heart size contributes minimally to total body weight, exercise can induce cardiac hypertrophy which
  might slightly increase heart weight. More importantly, heart tissue transcriptomic responses can indicate aerobic
  fitness and endurance capacity, which might correlate with weight changes (e.g. fitter animals may lose more fat). We
  include heart for completeness of multi-tissue integration, but we expect its direct contribution to body weight
  prediction to be smaller than fat or muscle. Heart omics data can still be informative as a systemic indicator of
  training status or health.

### Strategy for Integrating Multi-Omics Data (RNA-seq, ATAC-seq, RRBS)

We linked the three omics by genomic location utilizing MoTrPAC's Feature to Gene mapping table. This
compresses the data into a multi-omic profile per gene. 

## Initial Lasso Model For Weight Loss 

To evaluate how readily available our prepared data set is for Machine Learning, we applied a simple ML model to the prepared transciptome data.  Given the most common association with excercise is weight loss, we examined how weight loss is related to changes in the transcriptome.  The dataset provided an intial weight for every rat and a terminal weight, as well as the cohort the rat was a part of.  We were able to calculate a normalized weight loss rate by calculating the difference between terminal and initial weight divided by the number of weeks the rats exercised.  After researching a literature review of ML methods in transcriptomics (https://doi.org/10.1016/j.bbrc.2024.150225) we identified Lasso Regression as a commonly applied method.  After some difficulty with getting Lasso to converge, we pivoted to Lasso LARS (Least Angle Regression), as it is purported to perform well on high dimensional datasets.  In our final model, it regressed the 22000+ features to 8 informative features. 

The pipeline to prepare the data for the model was to calculate the goal physiological parameters.  We use scikit-learn to stratify the data set into 20% test data, with groups stratified across sacrifice week and sex. Because there are a significant number of nulls in the transcriptome dataset, we had to use K-nearest neighbors imputation to complete the dataset.  Prior to regression, data was mean-centered and normalized.  Regression was performed using SciKit-Learns LassoLars method varying alpha between 0.001 and 0.01.  By looking at MSE for each alpha, an optimal alpha that minimizes MSE was chosen. Predicted vs. Actuals makes sense, but shows bunching around 0,0.  This should be ok because we are trying to identify genes that are present in large weight loss rates. 

Simplifying weight loss to weight loss rate ignores the prior observation that females reach a maintenance weight in the middle of the trials.  A more sophisticated measure of weight loss would be needed to capture that observation.  It is possible that splitting the model between males and females would also be productive. 
