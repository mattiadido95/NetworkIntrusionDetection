'''
This is the main file for the project.
Here starts the execution of the program in which we load from data folder the dataset in a pandas dataframe.
'''
import pandas as pd


if __name__ == '__main__':
    # import dataset from data folder and load it in a pandas dataframe
    df = pd.read_csv('data/ML-EdgeIIoT-dataset.csv', low_memory=False)
    #
