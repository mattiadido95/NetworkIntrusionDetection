import pandas as pd

if __name__ == '__main__':
    DDN_df = pd.read_csv('data\DNN-EdgeIIoT-dataset.csv', low_memory=False) # 2219201

    ML_df = pd.read_csv('data\ML-EdgeIIoT-dataset.csv', low_memory=False) # 157800

    # for each dataframe check if classes are balanced
    print("DNN dataset")
    print(DDN_df['Attack_type'].value_counts())
    print("\nML dataset")
    print(ML_df['Attack_type'].value_counts())
