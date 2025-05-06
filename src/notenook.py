import marimo

__generated_with = "0.13.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Assignment 2

    1. Dataset: Comprehensive dataset of medical data associated with 13 distinct type of disbetes.
    2. Objective: Apply different classification models to the dataset, while applying cross validation methods and feature selection technique if neede to improve the performance of the models.
    3. Model Evluation.
    4. Report.
    """
    )
    return


@app.cell
def _():
    # Import packages
    import os, warnings
    import numpy as np
    import pandas as pd 
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    plt.style.use('default')
    warnings.filterwarnings('ignore')
    return (
        AdaBoostClassifier,
        DecisionTreeClassifier,
        GridSearchCV,
        LinearSVC,
        LogisticRegression,
        OneHotEncoder,
        RandomForestClassifier,
        StandardScaler,
        accuracy_score,
        cross_validate,
        f1_score,
        mo,
        np,
        os,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(
    AdaBoostClassifier,
    DecisionTreeClassifier,
    LinearSVC,
    LogisticRegression,
    OneHotEncoder,
    RandomForestClassifier,
    StandardScaler,
    accuracy_score,
    cross_validate,
    f1_score,
    np,
    pd,
    plt,
):
    # Used functions
    # train, validaition, and test split

    def get_scores(y_actual, y_pred):
        f1 = f1_score(y_actual, y_pred, average="macro")
        acc = accuracy_score(y_actual, y_pred)
        return f1, acc


    # Preprocessing 
    def preprocessing(train, validation):
        X_train = train.copy()
        X_val = validation.copy()
        cate_features = X_train.select_dtypes(include="object").columns.to_list()
        num_features = X_train.select_dtypes(include="int64").columns.to_list()
        # Initilize scaler
        scaler = StandardScaler()

        X_train[num_features] = scaler.fit_transform(X_train[num_features])
        X_val[num_features] = scaler.transform(X_val[num_features])

        # Initialize encoder 
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Transform training set
        encoded_train_data = encoder.fit_transform(X_train[cate_features])
        # Make encoded data frame for training set
        encoded_train = pd.DataFrame(
            encoded_train_data,
            columns=encoder.get_feature_names_out(cate_features),
            index=X_train.index
        )
        # Apply the training encoding and make encoded dataframe of the validation set
        encoded_val_data = encoder.transform(X_val[cate_features])
        encoded_val = pd.DataFrame(
            encoded_val_data,
            columns=encoder.get_feature_names_out(cate_features),
            index=X_val.index
        )

        # Combine the numerical and encoded columns for both the training and validation set. 
        X_train_processed = pd.concat([X_train[num_features], encoded_train], axis=1)
        X_val_processed = pd.concat([X_val[num_features], encoded_val], axis=1)

        return X_train_processed, X_val_processed

    def cross_validation(X_train, y_train):
        models = {
            "Logistic Regresion": LogisticRegression(penalty="l2", 
                                                     solver="saga",
                                                     max_iter=5000,
                                                     random_state=76, n_jobs=-1),
            "Linear SVC": LinearSVC(dual=False, 
                             max_iter=2000,
                             random_state=76),
            "Decision Tree": DecisionTreeClassifier(max_depth=5,
                                                    min_samples_leaf=5,
                                                    ccp_alpha=0.01,
                                                    random_state=76),
            "Random Forest": RandomForestClassifier(max_depth=7, 
                                                    min_samples_leaf=5,
                                                    random_state=76, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(random_state=76)
        }

        for name, model in models.items():
            results = cross_validate(
                model, 
                X_train, y_train,
                cv=5, 
                scoring = ["f1_macro", "accuracy"],
                n_jobs=-1,
                return_train_score=True
            )
            print(f"{name}:")
            print("  F1 (val):", results['test_f1_macro'])
            print("  F1 (train):", results['train_f1_macro'])
            print("  Acc (val):", results['test_accuracy'])
            print("  Acc (train):", results['train_accuracy'])
            print(f"------------------------------------")
            f1_scores = {
                "Training": results["train_f1_macro"],
                "Test": results["test_f1_macro"]
            }
            acc_scores = {
                "Training": results["train_accuracy"],
                "Test": results["test_accuracy"]
            }

            x = np.arange(5)
            width = 0.3

            fig, axes =  plt.subplots(1, 2, figsize=(18,6))
            fig.suptitle(f"5-Fold scores using {name} model.")
            axes[0].set_title("F1 Scores")
            axes[0].set_xlabel("Folds")
            axes[0].set_ylabel("F1 Score")
            axes[0].bar(x-0.2, f1_scores["Training"], width, label="Training")
            axes[0].bar(x+0.2, f1_scores["Test"], width, label="Validation")
            axes[0].set_ylim(0,1)
            axes[0].legend()

            axes[1].set_title("Accuracy Score")
            axes[1].set_xlabel("Folds")
            axes[1].set_ylabel("Accuracy")
            axes[1].bar(x-0.2, acc_scores["Training"], width, label="Training")
            axes[1].bar(x+0.2, acc_scores["Test"], width, label="Validation")
            axes[1].set_ylim(0,1)
            axes[1].legend()
            plt.show()
    return cross_validation, get_scores, preprocessing


@app.cell
def _(os):
    # Setup directories 
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_file_path, os.pardir, "data")
    data_path = os.path.join(data_dir, "diabetes_dataset00.csv")
    return (data_path,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Datasets: 


    1. The dataset contains 34 features, and 70000 observation with no null values, it includes 13 numerical features and 21 categorical features.
    2. We note the the distribution of classes in the target feature is mostly evenly distributed, ~ 5000 observation for each class.
    """
    )
    return


@app.cell
def _(data_path, pd):
    df = pd.read_csv(data_path)
    df.info()
    return (df,)


@app.cell
def _(df, plt, sns):
    _count = df["Target"].value_counts()
    plt.figure(figsize=(12, 6))  # width=10, height=6 inches
    sns.barplot(_count)
    plt.xticks(rotation=45, ha='right')  # Rotate labels 45 degrees, align right
    plt.tight_layout()                   # Adjust layout to prevent label cutoff
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # define numerical, and categorical features
    _cate_features = df.select_dtypes(include="object").columns.to_list()
    _num_features = df.select_dtypes(include="int64").columns.to_list()
    _features = []
    _count = []
    for feature in _cate_features: 
        print(f"The feature {feature} has {df[feature].nunique()} unique entries.")
        _features.append(feature)
        _count.append(df[feature].nunique())

    plt.figure(figsize=(12,6))
    sns.barplot(x= _features,
               y= _count)
    plt.xticks(rotation=45, ha="right")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data preperation. 

    1. Given that all categorical columns are low in cardinality, i will use one hot encoder to them all, and standard scaler for numerical features.
    2. I will do 68%, 17%, 15% training, validation and test split. Spliting the training and validation set into 5 equal folds after keeping 15% as test.
    """
    )
    return


@app.cell
def _(cross_validation, df, preprocessing, train_test_split):
    X, y = df.drop(columns="Target", axis=1).copy(), df["Target"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=76)

    X_train_processed, X_test_processed = preprocessing(X_train, X_test)
    cross_validation(X_train_processed, y_train)
    return X_test_processed, X_train_processed, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
    # Model Evaluation 

    As shown from the above graphs, the scores seem equal across all folds. But there is a discrepenacy in performance across different models. Chosing random forest being the highest performant.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Improved Model 

    To improve the performance of random forest, I will do the following: 

    1. Feature selection.
    2. Hyperparameter tuning (Grid Search).
    """
    )
    return


@app.cell
def _(RandomForestClassifier, X_train_processed, y_train):
    # Train a random forest model 
    rf = RandomForestClassifier(max_depth=7, 
                                min_samples_leaf=5,
                                random_state=76, n_jobs=-1)
    rf.fit(X_train_processed, y_train)
    return (rf,)


@app.cell
def _(X_train_processed, pd, plt, rf, sns):
    # Sort most important features, and plot top 20 
    importances = rf.feature_importances_
    feature_names = X_train_processed.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    sns.barplot(feat_imp[:20])
    plt.xticks(rotation=45, ha="right")
    plt.show()
    return (feat_imp,)


@app.cell
def _(X_test_processed, X_train_processed, get_scores, rf, y_test, y_train):
    f1_train, acc_train = get_scores(y_actual=y_train, y_pred = rf.predict(X_train_processed))
    f1_test, acc_test = get_scores(y_actual=y_test, y_pred = rf.predict(X_test_processed))
    print(f"Training: F1: {f1_train:.3f}, Accuracy: {acc_train:.3f}.")
    print(f"Test: F1: {f1_test:.3f}, Accuracy: {acc_test:.3f}.")
    return


@app.cell
def _(
    RandomForestClassifier,
    X_test_processed,
    X_train_processed,
    feat_imp,
    get_scores,
    y_test,
    y_train,
):
    # We notice slight improvement from feature selection
    top_features = feat_imp[:20].index.to_list()
    rf_feature = RandomForestClassifier(max_depth=7, 
                                min_samples_leaf=5,
                                random_state=76, n_jobs=-1)
    rf_feature.fit(X_train_processed[top_features], y_train)
    f1_train_feat, acc_train_feat = get_scores(y_actual=y_train, y_pred = rf_feature.predict(X_train_processed[top_features]))
    f1_test_feat, acc_test_feat = get_scores(y_actual=y_test, y_pred = rf_feature.predict(X_test_processed[top_features]))
    print(f"Training: F1: {f1_train_feat:.3f}, Accuracy: {acc_train_feat:.3f}.")
    print(f"Test: F1: {f1_test_feat:.3f}, Accuracy: {acc_test_feat:.3f}.")
    return (top_features,)


@app.cell
def _(
    GridSearchCV,
    RandomForestClassifier,
    X_train_processed,
    top_features,
    y_train,
):
    # Grid search
    # Best parameters: {'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 400}
    param_grid = {
        'n_estimators': [400, 600],         # Number of trees
        'max_depth': [10, 15],         # Tree depth
        'min_samples_split': [4,6],    # Min samples to split a node
        'min_samples_leaf': [1],      # Min samples at a leaf node
        'max_features': ['sqrt'],   # Number of features to consider at each split
        'bootstrap': [False]         # Whether bootstrap samples are used
    }
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }
    rf_grid = RandomForestClassifier(random_state=76, n_jobs=-1)

    grid = GridSearchCV(
        estimator=rf_grid,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        refit="f1_macro", 
        verbose=2, 
        return_train_score=True
    )
    grid.fit(X_train_processed[top_features], y_train)

    print("Best parameters:", grid.best_params_)
    print("Best f1_macro score:", grid.best_score_)

    results = grid.cv_results_
    print("All metrics for best model:")
    for metric in scoring:
        print(f"{metric}: {results[f'mean_test_{metric}'][grid.best_index_]:.3f}")

    return (grid,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Output of the above grid search 
    ```
    Best parameters: {'bootstrap': False, 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 600}
    Best f1_macro score: 0.9037603761618254
    All metrics for best model:
    accuracy: 0.905
    f1_macro: 0.904
    ```

    # Final Model

    1. After thorough analysis, I chose random forest with the above parameters and selected features.
    2. Finally, after cross validatiion, feature selection, and hyperparameter tunning. I will train the final model with 80% ~ 20% split.
    """
    )
    return


@app.cell
def _(
    RandomForestClassifier,
    df,
    grid,
    preprocessing,
    top_features,
    train_test_split,
):
    best_params = grid.best_params_
    X_final, y_final = df.drop("Target",axis=1).copy(), df["Target"].copy()
    X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
        X_final, y_final, test_size=0.2, random_state=76
    )
    X_train_sel_processed, X_test_sel_processed = preprocessing(X_train_selected,
                                                                X_test_selected)
    X_train_sel_processed, X_test_sel_processed = X_train_sel_processed[top_features] , X_test_sel_processed[top_features]

    rf_best = RandomForestClassifier(**best_params)
    return (
        X_test_sel_processed,
        X_train_sel_processed,
        rf_best,
        y_test_selected,
        y_train_selected,
    )


@app.cell
def _(
    X_test_sel_processed,
    X_train_sel_processed,
    get_scores,
    rf_best,
    y_test_selected,
    y_train_selected,
):
    rf_best.fit(X_train_sel_processed, y_train_selected)
    y_pred = rf_best.predict(X_test_sel_processed)
    y_pred_train = rf_best.predict(X_train_sel_processed)
    f1_train_final, acc_train_final = get_scores(y_actual=y_train_selected, 
                                                 y_pred=y_pred_train)
    f1_test_final, acc_test_final = get_scores(y_actual=y_test_selected,
                                               y_pred=y_pred)

    print(f"Final Model, training score ~ F1: {f1_train_final:.4f} Accuracy: {acc_train_final:.4f}")
    print(f"Final Model, testing score ~ F1: {f1_test_final:.4f} Accuracy: {acc_test_final:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
