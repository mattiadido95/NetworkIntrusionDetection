import datetime
import os

import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def split_data(df):
    # drop Attack_label column because it is not needed for multiclass classification
    mC_df = df.drop(columns=['Attack_label'])
    X = mC_df.drop(columns=['Attack_type'])
    y = mC_df['Attack_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
    return X_train, X_test, y_train, y_test


def create_classifier_svm(train, test):
    param_grid = {
        "estimator__C": [1, 2, 4, 8],
        "estimator__kernel": ["poly", "rbf", "linear", "sigmoid"],
        "estimator__degree": [1, 2, 3, 4],
    }
    print(f"{datetime.datetime.now()} - SVM: Grid search in progress...")
    clf = GridSearchCV(OneVsRestClassifier(SVC()), param_grid, cv=5, verbose=2, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - SVM: Grid search completed.")
    return clf.best_estimator_


def create_classifier_random_forest(train, test):
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    print(f"{datetime.datetime.now()} - Random Forest: Grid search in progress...")
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=2, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - Random Forest: Grid search completed.")
    return clf.best_estimator_


def create_classifier_knn(train, test):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    print(f"{datetime.datetime.now()} - K-Nearest Neighbors: Grid search in progress...")
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=2, n_jobs=6)
    clf.fit(train, test)
    print(f"{datetime.datetime.now()} - K-Nearest Neighbors: Grid search completed.")
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
    plt.savefig('results/multiClass/confusion_matrix_' + clf.__class__.__name__ + '.png')
    plt.show()
    report = classification_report(y_test, y_pred)
    with open('results/multiClass/classification_report_' + clf.__class__.__name__ + '.txt', 'w') as file:
        file.write(report)


def prepare_multi_classification(df):
    X_train, X_test, y_train, y_test = split_data(df)

    # clf_svm = create_classifier_svm(X_train, y_train)
    # save_best_model(clf_svm, 'svm_multiClass')
    clf_random_forest = create_classifier_random_forest(X_train, y_train)
    save_best_model(clf_random_forest, 'random_forest_multiClass')
    clf_knn = create_classifier_knn(X_train, y_train)
    save_best_model(clf_knn, 'knn_multiClass')

    return 0, clf_random_forest, clf_knn, X_test, y_test


def run_multi_classification(clf, test):
    clf_svm, clf_random_forest, clf_knn = clf
    X_test, y_test = test

    # print("SVM Classifier...")
    # test_classifier(clf_svm, X_test, y_test)

    print("Random Forest Classifier...")
    test_classifier(clf_random_forest, X_test, y_test)

    print("K-Nearest Neighbors Classifier...")
    test_classifier(clf_knn, X_test, y_test)
