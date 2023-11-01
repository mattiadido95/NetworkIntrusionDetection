from binary_classification import prepare_binary_classification, run_binary_classification
from multiClass_classification import prepare_multi_classification, run_multi_classification
from preprocessing import get_df


def training():
    # load data
    df = get_df()
    print("\nSelect which type of classifier you want train:")
    print("1. Binary")
    print("2. Multi")
    choice = input("Select an option: ")
    if choice == '1':
        # run binary classification
        clf_decision_tree, clf_logistic_regression, clf_random_forest, clf_svm, X_test, y_test = prepare_binary_classification(
            df)
        run_binary_classification((clf_decision_tree, clf_logistic_regression, clf_random_forest),
                                  (X_test, y_test))
    elif choice == '2':
        # run multi classification
        clf_svm, clf_random_forest, clf_knn, X_test, y_test = prepare_multi_classification(df)
        run_multi_classification((clf_svm, clf_random_forest, clf_knn), (X_test, y_test))


def prediction():
    print("\nLoad models for prediction:")
    print("1. Binary Classifiers")
    print("2. Multi Classifiers")
    choice = input("Select an option: ")

    if choice == '1':
        # Load binary models
        print("work in progress...")

    elif choice == '2':
        # Load multi-class models
        print("work in progress...")


if __name__ == '__main__':

    while True:
        print("\nMenu:")
        print("1. Execute training of classifiers")
        print("2. Execute prediction of classifiers")
        print("3. Exit")

        choice = input("Select an option: ")

        if choice == '1':
            training()
        elif choice == '2':
            prediction()
        elif choice == '3':
            print("Bye!")
            break
        else:
            print("Option not valid.")
