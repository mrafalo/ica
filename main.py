import os
if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")
    
import numpy as np
import pandas as pd
import work
import work.models as m
import work.data as d
import work.visualize as v
import yaml    
import importlib
from datetime import datetime
import tensorflow as tf
import random
import utils
import utils.custom_logger as cl
from sklearn.decomposition import FastICA
import glob
import time
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='svg'


logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']



def ica_results(_mask, _mse_reduction_threshold=0):
    #_mask = "tmp/ica_mse_result*"
    #_mse_reduction_threshold = -0.1
    path_pattern = 'results/' + _mask
    ica_files = glob.glob(path_pattern)
    
    for f in ica_files:
        df = pd.read_csv(f, sep=';')
        model_columns = [x for x in list(df.columns) if x not in ('scenario', 'predictions_file')]
        number_of_components = len(model_columns)
        
        base = df.loc[df.scenario == 'mse_base',model_columns].values
        components = df.loc[df.scenario != 'mse_base',model_columns].values
        
        for i in range(number_of_components):
            mse_reduction_prc = (components[i,:] - base) / base
            if np.mean(mse_reduction_prc) < _mse_reduction_threshold:
                print(f, "c_"+str(i) , round(np.mean(mse_reduction_prc),3))
 
 

    
def divergence_results(_mask, _number_of_components):
    _mask = "test2_divs4/div_result*"
    _number_of_components = 20
    
    path_pattern = 'results/' + _mask
    list_files = glob.glob(path_pattern)
    df_combined = pd.DataFrame();
    
    div_details = []
    k = 0
    for b in [0, 1, 2, 3]:
        for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            div_details.append((k, b, r))
            k = k + 1
            
    for f in list_files:
        df = pd.read_csv(f, sep=';')
        if 'source_filename' in df.columns:    
            df = df.drop('source_filename', axis=1)

        base = df.loc[df.component =='model_base',:].values[:,1:21]

        df = df.loc[df.component !='model_base',:]
        
        
        df_base = df
        result_values = df.drop('component', axis=1).values
        
        model_improvements = np.zeros((_number_of_components, _number_of_components))
        divs_count = result_values.shape[1] - _number_of_components - 1
        
        for k in range(_number_of_components):
            tmp = (result_values[k,0:20] - base) / base
            for j in range(_number_of_components):
                model_improvements[k,j] = tmp[0,j]
        
        df_model_improvements = pd.DataFrame(model_improvements, columns=["imp_" + str(k) for k in range(0, _number_of_components)]) 
        df.reset_index(inplace=True)
        df = pd.concat([df, df_model_improvements], ignore_index=True, axis=1)
        df.columns = list(df_base.columns) + list(df_model_improvements.columns)
       
        
        for k in range(_number_of_components):
            df['imp_split_'+str(k)] = 'no'
            df.loc[df['imp_'+str(k)]<0,'imp_split_'+str(k)] = 'yes'

        df_combined = pd.concat([df_combined,df])     



    for k in range(_number_of_components):
        for i in range(divs_count):
            tmp = df_combined.groupby('imp_split_'+str(k)).agg({'d_'+str(i): 'mean'}).values
            if tmp[0] != 0:
                ratio = abs(tmp[1]/tmp[0])
            else:
                ratio = 0
                
            if (ratio < 0.7) | (ratio > 1.3) :
                print(f, '_'+str(i), df_combined.groupby('imp_split_'+str(k)).agg({'d_'+str(i): 'mean'}))
    
    for k in div_details:
        var = 'd_'+str(k[0])
        #print(var, k)
        for j in range(_number_of_components):
            
            tmp = df_combined.groupby('imp_split_'+str(j)).agg({var: 'mean'}).values
            tmp_yes = tmp[1]
            tmp_no = tmp[0]
            if tmp[0] != 0: 
                ratio = abs(tmp[1]/tmp[0]) 
            else: 
                ratio = 0
                
            if (ratio>0) & ((ratio < 0.7) | (ratio > 1.3)):
                print(k)
                print(var, "imp_"+str(j), 'improved: ' + str(tmp_yes), 'not improved: ' + str(tmp_no), 
                      'diff: ' + str(np.round((tmp_yes-tmp_no)/tmp_no,2)))

    
    
# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(work.visualize)

    v.plot_scatter_ica(df_combined, ['imp_2','imp_2','imp_2'], ['d_10', 'd_21', 'd_32'], 
                       ['beta=0','beta=1', 'beta=2'], 'm2', 'plots/scatter1.png')

    v.plot_scatter_ica(df_combined, ['imp_17','imp_17','imp_17'], ['d_9', 'd_20', 'd_31'], 
                   ['beta=0','beta=1', 'beta=2'], 'm17', 'plots/scatter2.png')
  
    
   #_df, _x, _y, _labels, _caption
    
            
def compute_ica_iterator(_mask, _dest_folder):
    #_mask = "models_predictions*"
    path_pattern = 'results/' + _mask
    prediction_files = glob.glob(path_pattern)
    i = 0
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    for f in prediction_files:
        i = i + 1
        res_mse, res_ica = m.compute_ica(f)
        res_mse['source_filename'] = f
        res_ica['source_filename'] = f
        res_mse.to_csv(_dest_folder+"/ica_mse_result_" + str(i+1) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        res_ica.to_csv(_dest_folder+ "/ica_detail_result_" + str(i+1) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")

def compute_divergence_iterator(_mask, _dest_folder, _number_of_components):
    # _mask = "test2/ica_detail*"
    logger.info("divergence compute start for mask " + _mask)
    
    path_pattern = 'results/' + _mask
    list_files = glob.glob(path_pattern)
    i = 0
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    for f in list_files:
        i = i + 1
        logger.info("processing file: " + str(i) + "/" + str(len(list_files)))
        res = compute_divergence(f, _number_of_components)
        res['source_filename'] = f
        res.to_csv(_dest_folder+ "/div_result_" + str(i+1) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")

    logger.info("divergence compute finished; all OK!")
    
def compute_divergence(_filename, _number_of_components):
    
    # _filename = "results/test/ica_detail_result_28_20240304_1143.csv"
    # _number_of_components = 20
    df = pd.read_csv(_filename, sep=';')
    
    number_of_measures = 44
    divs = np.zeros((_number_of_components+1, number_of_measures))
    
    ica_model_results = df.iloc[:, 1 + 2*_number_of_components: len(df.columns)].values
    model_results = df.iloc[:,1:_number_of_components+1].values
    y_actual = df['y_actual'].values
    components_results = df.iloc[:, 1 + _number_of_components:1 + 2*_number_of_components].values
    

    mses = m.compute_mse(_number_of_components, y_actual, model_results, ica_model_results)

    k = 0
 
    xx = list(components_results.T)
    yy = ["c_" + str(k) for k in range(_number_of_components)]
    v.plot_dist_ica(xx, yy)
    
    
    for b in [0, 1, 2, 3]:
        for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            for i in range(_number_of_components):
                J = m.compute_J(components_results[:, i], b, r, 1)
                divs[i+1,k] = J

            k = k + 1
    
    res_array = np.concatenate((mses, divs), axis=1)
    
    component_names = ['model_base'] + ["c_" + str(k) for k in range(_number_of_components)]
    model_columns = ["m_" + str(k) for k in range(_number_of_components)] + ['avg_mse_reduction']
    div_names = ['d_' + str(k) for k in range(number_of_measures)]
    
    res = pd.DataFrame(res_array)
    tmp = pd.DataFrame(component_names)   
    res = pd.concat([tmp, res], ignore_index=True, axis=1)
    res.columns =  ['component'] + model_columns + div_names
    
    #res.to_csv("results/test/div.csv", index=False, sep=";")
    
    return res
    

def training_loop():
    
    logger.info("starting... epochs: " + str(EPOCHS))
     
    np.random.seed(m.SEED)
    tf.keras.utils.set_random_seed(m.SEED)
    random.seed(SEED)
    start = time.time()

    for i in range(ITERATIONS):
        logger.info("starting... iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
        X_train, y_train, X_test, y_test = d.get_data();
    
        res_preds, res_summary = m.train_models(X_train, y_train)
    
        curr_date = datetime.now().strftime("%Y%m%d_%H%M")
        predictions_file = "models_predictions_" + str(i+1) + "_" + curr_date + ".csv"
        res_preds.to_csv("results/" + predictions_file, mode='w', header=True, index=False, sep=";")
        res_summary.to_csv("results/models_summary_" + str(i+1) + "_" + curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        
        res_mse, res_ica = m.compute_ica('results/' + predictions_file)
        res_mse.to_csv("results/ica_mse_result_" + str(i+1) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        res_ica.to_csv("results/ica_detail_result_" + str(i+1) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")
    
    
        logger.info("done... iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
    stop = time.time()
    
    elapsed_sec = stop-start
    logger.info("training finished!, elapsed: " + str(elapsed_sec//60) + " minutes")

# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(work.visualize)

#training_loop()

#compute_ica_iterator(model_predictions/models_predictions*", "results/test2/")
compute_divergence_iterator("test2/ica_detail*", "results/test2_divs5/", 20)

#ica_results("test2/ica_mse_result*", -0.1)
#v.plot_heat_map_ica('results/test2\ica_mse_result_40_20240304_2210.csv', 'plots/heat6.png')
    

 
 
 