import datetime
import os
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def split_data(df):
    # drop Attack_type column because it is not needed for binary classification
    bin_df = df.drop(columns=['Attack_type'])
    X = bin_df.drop(columns=['Attack_label'])
    y = bin_df['Attack_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
    return X_train, X_test, y_train, y_test


def create_classifier_decision_tree(train, test):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    print(f"{datetime.datetime.now()} - Decision Tree: Grid search in progress...")
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, verbose=1, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - Decision Tree: Grid search completed.")
    return clf.best_estimator_


def create_classifier_logistic_regression(train, test):
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    print(f"{datetime.datetime.now()} - Logistic Regression: Grid search in progress...")
    clf = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, verbose=1, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - Logistic Regression: Grid search completed.")
    return clf.best_estimator_


def create_classifier_random_forest(train, test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    print(f"{datetime.datetime.now()} - Random Forest: Grid search in progress...")
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - Random Forest: Grid search completed.")
    return clf.best_estimator_


def save_best_model(clf, model_name):
    model_path = os.path.join('results/models', model_name + '.joblib')
    joblib.dump(clf, model_path)
    print(f"Best {model_name} model saved at {model_path}")


def test_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for ' + clf.__class__.__name__ + ' Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/binary/confusion_matrix_' + clf.__class__.__name__ + '.png')
    plt.show()
    report = classification_report(y_test, y_pred)
    with open('results/binary/classification_report_' + clf.__class__.__name__ + '.txt', 'w') as file:
        file.write(report)


def prepare_binary_classification(df):
    X_train, X_test, y_train, y_test = split_data(df)

    clf_decision_tree = create_classifier_decision_tree(X_train, y_train)
    save_best_model(clf_decision_tree, 'decision_tree_binary')
    clf_logistic_regression = create_classifier_logistic_regression(X_train, y_train)
    save_best_model(clf_logistic_regression, 'logistic_regression_binary')
    clf_random_forest = create_classifier_random_forest(X_train, y_train)
    save_best_model(clf_random_forest, 'random_forest_binary')

    return clf_decision_tree, clf_logistic_regression, clf_random_forest, X_test, y_test


def run_binary_classification(clf, test):
    clf_decision_tree, clf_logistic_regression, clf_random_forest = clf
    X_test, y_test = test

    print("Decision Tree Classifier...")
    test_classifier(clf_decision_tree, X_test, y_test)

    print("Logistic Regression Classifier...")
    test_classifier(clf_logistic_regression, X_test, y_test)

    print("Random Forest Classifier...")
    test_classifier(clf_random_forest, X_test, y_test)
