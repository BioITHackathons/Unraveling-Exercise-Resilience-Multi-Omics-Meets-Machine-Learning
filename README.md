# Unraveling-Exercise-Resilience-Multi-Omics-Meets-Machine-Learning




## Initial Lasso Model For Weight Loss 

To evaluate how readily available our prepared data set is for Machine Learning, we applied a simple ML model to the prepared transciptome data.  Given the most common association with excercise is weight loss, we examined how weight loss is related to changes in the transcriptome.  The dataset provided an intial weight for every rat and a terminal weight, as well as the cohort the rat was a part of.  We were able to calculate a normalized weight loss rate by calculating the difference between terminal and initial weight divided by the number of weeks the rats exercised.  After researching a literature review of ML methods in transcriptomics (https://doi.org/10.1016/j.bbrc.2024.150225) we identified Lasso Regression as a commonly applied method.  After some difficulty with getting Lasso to converge, we pivoted to Lasso LARS (Least Angle Regression), as it is purported to perform well on high dimensional datasets.  In our final model, it regressed the 22000+ features to 8 informative features. 

The pipeline to prepare the data for the model was to calculate the goal physiological parameters.  We use scikit-learn to stratify the data set into 20% test data, with groups stratified across sacrifice week and sex. Because there are a significant number of nulls in the transcriptome dataset, we had to use K-nearest neighbors imputation to complete the dataset.  Prior to regression, data was mean-centered and normalized.  Regression was performed using SciKit-Learns LassoLars method varying alpha between 0.001 and 0.01.  By looking at MSE for each alpha, an optimal alpha that minimizes MSE was chosen. Predicted vs. Actuals makes sense, but shows bunching around 0,0.  This should be ok because we are trying to identify genes that are present in large weight loss rates. 

Simplifying weight loss to weight loss rate ignores the prior observation that females reach a maintenance weight in the middle of the trials.  A more sophisticated measure of weight loss would be needed to capture that observation.  It is possible that splitting the model between males and females would also be productive. 
