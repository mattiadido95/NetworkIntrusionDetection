import pandas as pd
import pickle

from joblib import load

from binary_classification import prepare_binary_classification, run_binary_classification
from multiClass_classification import prepare_multi_classification, run_multi_classification
from preprocessing import get_df, create_6_class_dataframe, run
from six_class_classification import prepare_multi_6_class_classification, run_multi_6_class_classification


def training():
    # load data
    print("\nSelect which type of classifier you want train:")
    print("1. Binary")
    print("2. Multi")
    print("3. 6 Class")
    choice = input("Select an option: ")
    if choice == '1':
        # run binary classification
        df = get_df()
        clf_decision_tree, clf_logistic_regression, clf_random_forest, X_test, y_test = prepare_binary_classification(
            df)
        run_binary_classification((clf_decision_tree, clf_logistic_regression, clf_random_forest),
                                  (X_test, y_test))
    elif choice == '2':
        df = get_df()
        # run multi classification
        clf_svm, clf_random_forest, clf_knn, X_test, y_test = prepare_multi_classification(df)
        run_multi_classification((clf_svm, clf_random_forest, clf_knn), (X_test, y_test))
    elif choice == '3':
        df = get_df(True)
        # run 6 class classification
        clf_random_forest, clf_knn, X_test, y_test = prepare_multi_6_class_classification(df)
        run_multi_6_class_classification((clf_random_forest, clf_knn), (X_test, y_test))


def prediction():
    # load test dataset
    test_df = pd.read_csv('data\ML-EdgeIIoT-dataset-final-test.csv', low_memory=False)
    labels = {
        'Normal': 'Normal',
        'DDoS_UDP': "DDoS",
        'DDoS_ICMP': "DDoS",
        'DDoS_HTTP': "DDoS",
        'DDoS_TCP': "DDoS",
        'Vulnerability_scanner': "Scanning",
        'Password': "Malware",
        'SQL_injection': "Injection",
        'Uploading': "Injection",
        'Backdoor': "Malware",
        'Port_Scanning': "Scanning",
        'XSS': "Injection",
        'Ransomware': "Malware",
        'Fingerprinting': "Scanning",
        'MITM': "MITM",
    }
    test_df['6_Class'] = test_df['Attack_type'].map(labels)

    binary_classifier = load('results/models/random_forest_binary.joblib')
    multiclass_classifier = load('results/models/random_forest_multiClass.joblib')
    six_class_classifier = load('results/models/random_forest_multi6Class.joblib')

    columns_to_drop = ['Attack_label', 'Attack_type', '6_Class']

    y = test_df[['Attack_type', 'Attack_label', '6_Class']]
    X = test_df.drop(columns=columns_to_drop)

    predictions_binary = binary_classifier.predict(X)
    predictions_six_class = six_class_classifier.predict(X)
    predictions_multiclass = multiclass_classifier.predict(X)

    with open('results/prediction_results.txt', 'w') as file:
        for i in range(len(predictions_binary)):
            if predictions_binary[i] != y['Attack_label'][i]:
                file.write(f"Binary classifier prediction: {str(predictions_binary[i])} - Real label: {str(y['Attack_label'][i])} *** \n")
            else:
                file.write(f"Binary classifier prediction: {str(predictions_binary[i])} - Real label: {str(y['Attack_label'][i])}\n")

            if predictions_six_class[i] != y['6_Class'][i]:
                file.write(f"6 Class classifier prediction: {str(predictions_six_class[i])} - Real label: {str(y['6_Class'][i])} *** \n")
            else:
                file.write(f"6 Class classifier prediction: {str(predictions_six_class[i])} - Real label: {str(y['6_Class'][i])}\n")

            if predictions_multiclass[i] != y['Attack_type'][i]:
                file.write(f"Multi classifier prediction: {str(predictions_multiclass[i])} - Real label: {str(y['Attack_type'][i])} *** \n")
            else:
                file.write(f"Multi classifier prediction: {str(predictions_multiclass[i])} - Real label: {str(y['Attack_type'][i])}\n")

            file.write("\n")


def preprocessing():
    run()


if __name__ == '__main__':
    while True:
        print("\nMenu:")
        print("1. Execute preprocessing")
        print("2. Execute training of classifiers")
        print("3. Execute prediction of classifiers")
        print("4. Exit")

        choice = input("Select an option: ")

        if choice == '1':
            preprocessing()
        elif choice == '2':
            training()
        elif choice == '3':
            prediction()
        elif choice == '4':
            print("Bye!")
            break
        else:
            print("Option not valid.")



# TODO rifare train perche mancano i report
# TODO probabile fare il train dei multi class con dataset specifici