import os
import numpy as np
import pandas as pd
import work
import work.models as m
import work.data as d
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
import seaborn as sns
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
logger = cl.get_logger()
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='svg'

import plotly.figure_factory as ff

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']

  
def plot_ts_ica(_filename):
    
    _filename = "results/test2\ica_detail_result_11_20240304_2210.csv"
    _number_of_components = 20
    
    df = pd.read_csv(_filename, sep=';')
    l = 50
    
    df.loc[0:l,['y_actual', 'c_0_xp_0']].plot()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(4, 1, 1)
    plt.title("Original Signals")
    plt.plot(df.y_actual)
    
    plt.subplot(4, 1, 2)
    plt.title("m1 Signals")
    plt.plot(df.c_0_xp_0)
    
    # Generate random mixed signals
    np.random.seed(0)
    n_samples = 200
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)  # Signal 1
    s2 = np.sign(np.sin(3 * time))  # Signal 2
    s3 = np.random.randn(n_samples)  # Signal 3
    S = np.c_[s1, s2, s3]
    
    # Mixing matrix
    A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
    X = np.dot(S, A.T)  # Mixed signals
    
    # Apply ICA
    ica = FastICA(n_components=3)
    independent_components = ica.fit_transform(X)
    
    # Visualize the independent components
    plt.figure(figsize=(12, 6))
    
    plt.subplot(4, 1, 1)
    plt.title("Original Signals")
    plt.plot(S)
    
    plt.subplot(4, 1, 2)
    plt.title("Mixed Signals")
    plt.plot(X)
    
    plt.subplot(4, 1, 3)
    plt.title("ICA Components")
    plt.plot(independent_components)
    
    plt.subplot(4, 1, 4)
    plt.title("Original Signals (after ICA)")
    reconstructed_signals = np.dot(independent_components, A)
    plt.plot(reconstructed_signals)
    
    plt.tight_layout()
    plt.show()
   

def plot_scatter_ica(_df, _x, _y, _labels, _caption, _dest):
   fig = go.Figure()

   for i in range(len(_labels)): 
       fig.add_trace(go.Scatter(
            mode='markers',
            x=_df[_x[i]],
            y=_df[_y[i]],
            name=_labels[i] ,
            opacity=0.4
        ))
    

   fig.update_layout(
        margin = dict(t=10,r=10,b=10,l=10),
        xaxis_title="MSE improvement (model "+_caption+")",
        yaxis_title="noise distance",
        #legend_title="ratio (gausian)",
        legend=dict(
        yanchor="top",
        y=max(_df[_y[i]])+0.1,
        xanchor="left",
        x=0.01
        ),
        
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"
        )
    )
   
   fig.write_image(_dest)
   fig.show()
   
def plot_dist_ica(_data, _labels):
    fig = ff.create_distplot(_data, _labels ,show_rug=False, show_hist=False)
    fig.show()

def plot_heat_map_ica(_source, _dest):
    #_source = "results/test2\ica_mse_result_28_20240304_2210.csv"
    _number_of_components = 20
    
    df = pd.read_csv(_source, sep=';')
    base = df.loc[df.scenario =='mse_base',:].values[:,1:21]
    df = df.loc[df.scenario !='mse_base',:]
    df_base = df
    result_values = df.drop('scenario', axis=1).values
    
    model_improvements = np.zeros((_number_of_components, _number_of_components))
        
    for k in range(_number_of_components):
        tmp = (result_values[k,0:20] - base) / base
        for j in range(_number_of_components):
            model_improvements[k,j] = tmp[0,j]
    
    df_model_improvements = pd.DataFrame(model_improvements, columns=["imp_" + str(k) for k in range(0, _number_of_components)]) 
    df.reset_index(inplace=True)
    df = pd.concat([df, df_model_improvements], ignore_index=True, axis=1)
    df.columns = list(df_base.columns) + list(df_model_improvements.columns)

    df.index = ["c_" + str(k) for k in range(_number_of_components)]
    
    df_plot = df[["imp_" + str(k) for k in range(_number_of_components)]]

    fig = px.imshow(df_plot, aspect="auto", color_continuous_scale='OrRd')
    fig.update_coloraxes(showscale=False)

    fig.update_layout(
        margin = dict(t=10,r=10,b=10,l=10),
        showlegend = False,
        xaxis_title="models",
        yaxis_title="components",
        width = 700, height = 700,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"
        )
    )
       
    fig.write_image(_dest)
    fig.show()
