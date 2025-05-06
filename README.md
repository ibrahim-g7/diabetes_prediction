# Diabetes Prediction
![](./img/diabetes-528678_1280.jpg)

* [Source Code](./src/notenook.ipynb)
* [Report](./doc/report.pdf)
* [Installation Instruction](#running-instructions)

# Problem Statement
Develop a multi-class classification model to accurately predict the type of diabetes (from a set of 13 distinct types) based on a comprehensive dataset of patient medical data. The solution must involve the implementation, evaluation, and comparison of multiple supervised machine learning algorithms, along with appropriate feature selection techniques and hyperparameter tuning strategies, to achieve robust and reliable performance. The ultimate goal is to create a model capable of providing accurate and informative predictions for diverse forms of diabetes in a real-world healthcare setting.
# Dataset 
Publicly available data set from (Kaggle)[https://www.kaggle.com/datasets/ankitbatra1210/diabetes-dataset?resource=download] that provide extensive overview of different type of diabetes, it was obtained 

# Objective 

The objective of this assignment is to apply supervised machine learning techniques to develop a multi-class classification model capable of predicting the type of diabetes based on patient data.

### Model Configurations
Five classification models were implemented and evaluated:

1. **Logistic Regression**:
   - Penalty: L2 regularization
   - Solver: SAGA (suitable for multiclass problems)
   - Max iterations: 5000

2. **Support Vector Machine (Linear SVC)**:
   - Dual: False (optimized for cases where n_samples > n_features)
   - Max iterations: 2000

3. **Decision Tree**:
   - Max depth: 5
   - Min samples per leaf: 5
   - Cost-complexity pruning alpha: 0.01

4. **Random Forest** (bagging ensemble):
   - Initial configuration:
     - Max depth: 7
     - Min samples per leaf: 5
   - Optimized configuration (after grid search):
     - Bootstrap: False
     - Max depth: 15
     - Feature selection: sqrt
     - Min samples split: 6
     - Min samples leaf: 1
     - Number of estimators: 600

5. **AdaBoost** (boosting ensemble):
   - Default configuration with base estimator

All models were evaluated using 5-fold cross-validation to ensure robust performance assessment.

## Model Comparison

### Performance Metrics
The models were evaluated using two key metrics:

1. **F1-Score (Macro)**: This metric provides a balanced measure of precision and recall, especially important for multi-class classification with potentially imbalanced classes.Running instructions Train) | F1-Score (Test) | Accuracy (Train) | Accuracy (Test) |
|-------------------------------------|------------------|-----------------|------------------|-----------------|
| Logistic Regression                 | 0.760            | 0.754           | 0.761            | 0.756           |
| Linear SVC                          | 0.708            | 0.705           | 0.711            | 0.708           |
| Decision Tree                       | 0.463            | 0.463           | 0.538            | 0.538           |
| Random Forest (initial)             | 0.844            | 0.845           | 0.845            | 0.847           |
| Random Forest (with feature selection) | 0.854         | 0.857           | 0.855            | 0.858           |
| Random Forest (optimized)           | 0.926            | 0.906           | 0.926            | 0.908           |
| AdaBoost                            | 0.161            | 0.161           | 0.267            | 0.267           |


The Random Forest model consistently outperformed other algorithms across all evaluation metrics. The performance improved significantly after feature selection and hyperparameter optimization, achieving a final test F1-score of 0.906 and accuracy of 0.908.

# Running instructions 

1. Create a python cirtual environment: 

```python3 -m venv .venv ```

2. Activate the environment: 

- On Linux and MacOS:

``` source .venv/bin/activate ```

- On Windows

``` .venv\Scripts\activate ``` 

3. Install the requirments/dependencies: 

``` pip install -r requirements.txt```

4. Run the code.

5. Deactivate when finished 

```shell
deactivate
```