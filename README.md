This repository contains a Jupyter Notebook (`Credit Risk Default.ipynb`) that analyzes credit risk default using a dataset. The notebook covers various aspects of data preprocessing, exploratory data analysis (EDA), and predictive modeling for credit risk assessment. In this notebook, the aim is to predict credit risk default using a dataset containing information about individuals' demographics, financial status, and loan details. The notebook follows a structured approach to analyze the data and build predictive models using machine learning algorithms.

## Requirements
To run the notebook, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- imbalanced-learn
- xgboost

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost
```

## Dataset
The dataset (`credit_risk_dataset.csv`) contains information on various attributes such as age, income, employment length, loan amount, interest rate, and loan status (whether the loan is defaulted or not).

## Notebook Contents

1. **Data Loading and Cleaning**: The dataset is loaded and initial cleaning steps such as handling missing values and removing outliers are performed

2. **Exploratory Data Analysis (EDA)**:
   - The distribution of numerical features by loan status is analyzed
   - Skewness, kurtosis, and statistical descriptions of the dataset are checked
   - Relationships among numerical features are visualized using scatterplots and correlations are explored using a correlation matrix

3. **Feature Engineering**:
   - The relationship between loan status and age/income is analyzed by grouping data into buckets and visualizing failure to repay by age and income groups

4. **Modeling**:
   - The dataset is split into training and testing sets
   - Class imbalance is handled using Synthetic Minority Over-sampling Technique (SMOTE)
   - Two predictive models are built: XGBoost Classifier and Random Forest Classifier
   - Model performance is evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC

## Running the Notebook
To run the notebook, simply open it in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab) and execute the cells sequentially. Make sure to have the required libraries installed.
