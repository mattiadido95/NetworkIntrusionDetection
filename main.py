from binary_classification import prepare_binary_classification, run_binary_classification
from multiClass_classification import prepare_multi_classification, run_multi_classification
from preprocessing import get_df

if __name__ == '__main__':
    # load data
    df = get_df()

    # run binary classification
    clf_decision_tree, clf_logistic_regression, clf_random_forest, X_test, y_test = prepare_binary_classification(df)
    run_binary_classification((clf_decision_tree, clf_logistic_regression, clf_random_forest), (X_test, y_test))

    # run multi classification
    clf_svm, clf_random_forest, clf_knn, X_test, y_test = prepare_multi_classification(df)
    run_multi_classification((clf_svm, clf_random_forest, clf_knn), (X_test, y_test))
