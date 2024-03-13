
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.decomposition import FastICA
from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import utils
import utils.custom_logger as cl

logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']


def gausian(_n):
    mu = 0      # Mean of the distribution
    sigma = 1   # Standard deviation of the distribution
    
    return np.random.normal(mu, sigma, _n)

def cauchy(_n):
    mu = 0    # Location parameter
    gamma = 1  # Scale parameter
    standard_samples = np.random.standard_cauchy(_n)

    return mu + gamma * standard_samples

def uniform(_n):
    
    low = 0    # Lower bound of the distribution
    high = 1   # Upper bound of the distribution

    return np.random.uniform(low, high, _n)

def print_summary(_x):
    print(" min:", str(round(np.min(_x),2)), 
          " max:", str(round(np.max(_x),2)), 
          " mean:", str(round(np.mean(_x),2)))

def standarize(_x):
    res = (_x - np.mean(_x)) / np.std(_x)
    res = res - np.min(res) + 1
    
    return res
    
def compute_D(_y, _z, _beta):
   
    _y = standarize(_y)
    _z = standarize(_z)

    
    if _beta > 0:
        res = sum((_y * (pow(_y, _beta) - pow(_z, _beta)) / _beta) - ((pow(_y, _beta+1) - pow(_z, _beta+1)) / (_beta+1)))
        return res
    
    if _beta == 0:
        res = sum(_y * np.log(_y/_z) - _y + _z)
        return res
 
    if _beta == -1:
        res = sum(np.log(_z/_y) - _y/_z - 1)
        return res
    
def compute_J(_y, _beta, _gausian_ratio, _cauchy_ratio=0):

    n = len(_y)
    b_gausian = _gausian_ratio;
    b_uniform = 1 - _gausian_ratio - _cauchy_ratio
    b_cauchy = _cauchy_ratio; 


    gausian_part =  b_gausian * np.log(compute_D(_y, gausian(n), _beta) / compute_D(gausian(n), _y, _beta))
    uniform_part = b_uniform * np.log(compute_D(_y, uniform(n), _beta) / compute_D(uniform(n), _y, _beta))
    cauchy_part = b_cauchy * np.log(compute_D(_y, cauchy(n), _beta) / compute_D(cauchy(n), _y, _beta))

    res = pow(pow(gausian_part,2) + pow(uniform_part,2) + pow(cauchy_part,2), 0.5)
    #res =  min(gausian_part,uniform_part, cauchy_part)
    return res
    
    
def ica_model(_X):
    ica2 = FastICA()
    y = ica2.fit_transform(_X) 
    #w = ica.mixing_
    #w = ica.components_
    return y

def mape_score(_y_test, _y_pred):
    
    return sum(abs((_y_test - _y_pred) / _y_test)) / len(_y_test);

def mse_score(_y_test, _y_pred):
    
    return sum(pow(_y_test - _y_pred,2)) / len(_y_test);

def model_random_forest(_X_train, _y_train, _X_test, _y_test):
    m1 = RandomForestRegressor(n_estimators=10)
    m1 = m1.fit(_X_train, _y_train)
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds


def model_nn_custom(_layers, _sizes, _activations, _input_size):
    
    model = Sequential()
    model.add(Input(shape=(_input_size,)))
    
    for i in range(_layers):
        model.add(Dense(_sizes[i], activation=_activations[i]))
    
    model.add(Dense(1, activation='linear'))
    
    return model

def model_fiter(_model,_X_train, _y_train, _X_test, _y_test, _scale=False):
    
    if _scale:
        sc = StandardScaler()
        _X_train = sc.fit_transform(_X_train)
        _X_test = sc.transform(_X_test)
        scale = 'scaled'
    else:
        scale = 'not_scaled'
            
        
    _model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    _model.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    mse, mape, r2, preds = model_preditor(_model, _X_test, _y_test)
    return mse, mape, r2, preds


def model_preditor(_model, _X_test, _y_test):
    m1_predict = _model.predict(_X_test, verbose=False)
    
    if len(m1_predict.shape) > 1:
        m1_predict = m1_predict[:,0]
    
    m1_mse = round(mean_squared_error(_y_test, m1_predict),3)
    m1_mape = round(mape_score(_y_test, m1_predict),3)
    m1_r2 = round(r2_score(_y_test, m1_predict), 3)

    return m1_mse, m1_mape, m1_r2, m1_predict

  
def train_models(_X, _y):
    
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])
    
    df_preds = pd.DataFrame({'y_actual': _y})
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])

    model_cfg = pd.read_csv(MODEL_CONFIG_FILE, sep=";")
    for _,c in model_cfg.iterrows():
        logger.info("model " + c['model_name'] + " training start dataset size: " + str(len(_X)))
        
        sizes = [int(num) for num in c['sizes'].split(',')]
        activations = [ss for ss in c['activations'].split(',')]
        m1 = model_nn_custom(c['layers'], sizes, activations, len(_X.columns))
        
        mse, mape, r2, preds = model_fiter(m1, _X, _y, _X, _y, False)
    
        new_row = {'model':c['model_name'], 'mse':mse, 'mape':mape, 'r2': r2}
        res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)    
        df_preds[c['model_name']] = preds
        
        logger.info("model " + c['model_name'] + " training finished...")
                
    return df_preds, res


def compute_mse(_number_of_components, _y_actual, _model_results, _components_results):
    
    mses = np.zeros((_number_of_components + 1,_number_of_components + 1))
    
    for j in range(_number_of_components):
        mse = mse_score(_y_actual, _model_results.T[j])
        mses[0,j] = round(mse)

    for i in range(_number_of_components):
        for j in range(_number_of_components):
    
            mse = mse_score(_y_actual, _components_results[:,i*_number_of_components + j])
         
            mses[i+1,j] = round(mse)
            
    base = mses[0,0:_number_of_components]

    for i in range(_number_of_components):
       
        mse_reduction_prc = (mses[i+1,0:_number_of_components] - base) / base
        
        
        # print(mses[i+1,0:_number_of_components])
        # print("-----")
        # print(np.round(mse_reduction_prc,2))
        # #print(np.mean(mse_reduction_prc))
        mses[i+1,_number_of_components] = np.mean(mse_reduction_prc)
    
    return mses 
    
def compute_ica(_predictions_file):
    
    #_predictions_file = "results\models_predictions_10_20240303_1427.csv"
    df = pd.read_csv(_predictions_file, sep=';')
    df = df.drop('source_filename', axis=1)
    
    y_actual = df['y_actual'].values
    x = df.drop('y_actual', axis=1).values
    x_full = df.values
    
    number_of_components = x.shape[1]
    number_of_cases = x.shape[0]
    
    ica = FastICA()
    y = ica.fit_transform(x) 
    mses = np.zeros((number_of_components + 1,number_of_components))
    xps = np.zeros((number_of_cases ,number_of_components))
   
    scenarios = [] 
    scenarios.append('mse_base')
    
    ica_components = np.concatenate((x_full, y), axis=1)
    ica_columns = ["c_" + str(k) for k in range(0, number_of_components)]
    xp_colnames = []
    
    for j in range(number_of_components):
        mse = mse_score(y_actual, x.T[j])
        mses[0,j] = round(mse)

    for i in range(number_of_components):
        z = y.copy()
        z[:,i] = 0
        scenarios.append('mse_c_' + str(i))
        
        xp = ica.inverse_transform(z)
        xp_colnames = xp_colnames + ['c_' + str(i) + "_xp_" + str(k) for k in range(0, number_of_components)]
        ica_components = np.concatenate((ica_components, xp), axis=1)

        for j in range(number_of_components):
            mse = mse_score(y_actual, xp[:,j])
            mses[i+1,j] = round(mse)
    
    res_ica = pd.DataFrame(ica_components, columns = list(df.columns)+ica_columns+xp_colnames)
        
    res = pd.DataFrame(mses, columns=df.drop('y_actual', axis=1).columns)
    tmp = pd.DataFrame(scenarios, columns=['scenario'])   
    res_mse = pd.concat([tmp, res], ignore_index=True, axis=1)
    res_mse.columns =  list(tmp.columns)+list(res.columns)
    res_mse['predictions_file'] = _predictions_file

    return res_mse, res_ica
