# Model Card

## Model Details
* This model perdicts whether an individuals salary is greater or less than $50,000.
* Model date: 3 Feb 2022.
* Model type: Logistic Regression from scikit-learn.

## Intended Use
* This model was developed as a part of Udacity's Machine Learning DevOps Engineer Nanodegree.

## Training Data
* This model is trained on US Census Dataset available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data
* 20% of the cleaned data was held back for testing.

## Metrics
* The model is evaluated by precision, recall, and F1 scores:
    * Precision: ```0.7078```
    * Recall: ```0.5976```
    * F1 Score: ```0.6481```
* A breakdown of model performance on slices of the dataset can be found [here](https://github.com/abhishekanirudhan/udacity-deploy-ml/blob/main/data/slice_output.txt)

## Ethical Considerations
* Given that we're dealing with demographic data including sex and race, we should be vary of the model picking up inherent bias.

## Caveats and Recommendations
* This is a first-pass model, and can be made robust.
* It might be worthwhile to invest in augmenting our dataset with more features and records.