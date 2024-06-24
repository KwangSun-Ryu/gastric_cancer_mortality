# Predicting Mortality in Young Patients with Gastric Cancer

 
### Description

This study aim to develop a predictive model for mortality in young patients with gastric cancer. 
Using machine learning techniques, we preprocess the data, extract important features, and build a model to predict outcomes.

### Framework
The figure below illustrates the framework of our study:
![Figure1](https://github.com/KwangSun-Ryu/gastric_cancer_mortality/assets/122508289/22d56a19-9356-47ac-9f5a-c13ecdd81cd1)


Description of the Workflow
- Data Source: The data is sourced from the CPSD (2013-2015).
- Pre-processing: The raw data undergoes preprocessing to clean and prepare it for modeling.
- Data Splitting: The experimental data is split into a training set (70%) and a test set (30%).
- Model Training: Three survival machine learning models are trained:
  1. Random Survival Forest
  2. Gradient Boost Survival Analysis
  2. Extra Survival Tree
- Mortality Prediction: The models predict mortality at 1 year, 3 years, and 5 years.
- Model Evaluation: The models' performance is evaluated using the C-index.



### Table of Contents
- Installation
- Usage
- Files Description


### Installation
To use this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter


### Usage
- Preprocess the Data: Use the GC_data_preprocessing.py script to clean and preprocess the dataset.
- Feature Importance: Run the GC_feature_importance_rank.py script to determine the most important features for the model.
- Model Training and Prediction: Open the Jupyter notebook Prediction_Mortality_Gastric_Cancer_Young_patients.ipynb to train the model and make predictions.

### Example
To preprocess the data
```
python GC_data_preprocessing.py
```

To rank feature importance:
```
python GC_feature_importance_rank.py
```
To train the model and predict mortality, follow the steps in the Jupyter notebook.

### File Description
- GC_data_preprocessing.py: This script contains the code for data cleaning and preprocessing.
- GC_feature_importance_rank.py: This script ranks the features based on their importance to the predictive model.
- Prediction_Mortality_Gastric_Cancer_Young_patients.ipynb: A Jupyter notebook that includes steps for training the machine learning model and making predictions.
