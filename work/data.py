import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
import utils.custom_logger as cl
import glob

logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']

def load_ica_results(_mask):
    
    #_mask = "ica_result*"
    path_pattern = 'results/' + _mask

    csv_files = glob.glob(path_pattern)
    df = pd.concat((pd.read_csv(file, sep=';') for file in csv_files), ignore_index=True)
    #df.to_csv('results/tmp.csv', sep=';')
    #df.scenario.unique()   
    
    return df

def get_data():
    #https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom

    df = load_data_file(TELCO_FILE)
    df = df.loc[df.MonthlyMinutes>0,:]
    
    df.dropna(inplace=True)
    
    #df.replace({'Yes': 1, 'No': 0}, inplace=True)
    
    X = df.drop(["CustomerID", "Churn", "MonthlyRevenue", "MonthlyMinutes"], axis=1)
    y = df['MonthlyMinutes']
    
    for c in X.columns:
        t = X[c].dtype
        if not (t in ['float64', 'int64']):
            X.drop(c, axis=1, inplace=True)
              

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    #same data for test and train!!!
    res_X_train = X_test
    res_y_train = y_test
    res_X_test = X_test
    res_y_test = y_test    
    
    logger.info("total dataset size: " + str(len(df)) + " dataset for training: " + str(len(res_X_train)))
    
    return res_X_train, res_y_train, res_X_test, res_y_test




def load_data_file(_data_file):
    df = pd.read_csv(_data_file, sep=',')
    #df = df.reset_index()
    
    return df

    
                    




