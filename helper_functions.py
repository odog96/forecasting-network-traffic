import pandas as pd
import numpy as np
import os
import cdsw
import joblib

from joblib import dump, load

import tensorflow as tf
from tensorflow import keras

import json

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense ,Input, LSTM, Concatenate, Dropout, Reshape, Lambda
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the saved OneHotEncoder fit object
ohefit = joblib.load("./model/ohefit.save")
print('got onefit')
# Load the saved StandardScaler object
scaler = joblib.load("./model/scaler.save")


@tf.keras.utils.register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def create_model(n_past, num_features, n_future, num_semi_temporal_features,num_non_temporal_features, lstm_units=64, non_temporal_size=50, semi_temporal_size=50, 
                 dense_layers=[50], activation='relu', dropout_rate=0.2):
    """ defines architecture for DNN. LSTM layers, layer size, input and ouput size"""
    # Temporal features input
    temporal_input = Input(shape=(n_past, num_features))

    # LSTM branch for temporal features
    lstm_out = LSTM(units=lstm_units, return_sequences=False)(temporal_input)
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # Non-temporal features input
    non_temporal_input = Input(shape=(num_non_temporal_features,))  
    non_temporal_branch = Dense(non_temporal_size, activation=activation)(non_temporal_input)
    non_temporal_branch = Dropout(dropout_rate)(non_temporal_branch)

    # Semi-temporal features input
    semi_temporal_input = Input(shape=(num_semi_temporal_features,))
    semi_temporal_branch = Dense(semi_temporal_size, activation=activation)(semi_temporal_input)
    semi_temporal_branch = Dropout(dropout_rate)(semi_temporal_branch)

    # Concatenate outputs
    combined = Concatenate()([lstm_out, non_temporal_branch, semi_temporal_branch])

    # Additional Dense layers as specified
    for layer_size in dense_layers:
        combined = Dense(layer_size, activation=activation)(combined)
        combined = Dropout(dropout_rate)(combined)

    # Final output layer
    final_output = Dense(n_future*2)(combined)  # 2 features
    final_output = Reshape((n_future, 2))(final_output)  # Reshape to [?, 12, 2]
    
    # Construct and compile the model
    model = Model(inputs=[temporal_input, non_temporal_input, semi_temporal_input], outputs=final_output)
    model.compile(optimizer='adam', loss=rmse)  # Adjust optimizer and loss as needed

    return model

def feature_create(df_in):
    df = df_in.copy()
    
    df['site_host'] = df['site'] + "_" + df['host']
    
    df['rx_bytes_delta'] = pd.to_numeric(df['rx_bytes_delta'], errors='coerce')
    df['tx_bytes_delta'] = pd.to_numeric(df['tx_bytes_delta'], errors='coerce')
    
    df['rx_bytes_delta_delta'] = df.groupby('site_host')['rx_bytes_delta'].diff()

    # Calculate tx_bytes_delta_delta
    df['tx_bytes_delta_delta'] = df.groupby('site_host')['tx_bytes_delta'].diff()

    # The first delta_delta value for each site_host group will be NaN because there's no previous value to subtract from.
    # You might want to fill these NaN values depending on your requirements, for example, with 0s:
    df['rx_bytes_delta_delta'] = df['rx_bytes_delta_delta'].fillna(0)
    df['tx_bytes_delta_delta'] = df['tx_bytes_delta_delta'].fillna(0)
    
    # ensure 'ts' is a datetime64 type
    df['ts'] = pd.to_datetime(df['ts'])

    # Extract hour of day
    df['hour_of_day'] = df['ts'].dt.hour

    # Extract day of the week (Monday=0, Sunday=6)
    df['day_of_week'] = df['ts'].dt.dayofweek
    
        # Encode 'hour_of_day' cyclically
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)

    # Encode 'day_of_week' cyclically
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
    df.drop(columns=['hour_of_day','day_of_week'],inplace=True)

    # Transform the new data using the loaded OneHotEncoder fit object
    new_data_encoded = ohefit.transform(df[['host', 'site']])
    df_encoded = pd.concat([df,new_data_encoded],axis = 1).drop(columns= ['host', 'site'])
    
    # scale
    features_to_scale = [col for col in df_encoded.columns if 'delta' in col]
    # Transform the selected columns using the loaded StandardScaler
    df_encoded[features_to_scale] = scaler.transform(df_encoded[features_to_scale])
    del(df)
    return df_encoded
    
def feature_set_count(df):
    """generating the size of the inputs based on number of fields of each type"""
    temporal_count = 0
    num_non_temporal_features = 0
    num_semi_temporal_features = 0
    for col in list(df.drop(columns=['site_host']).columns):
        if 'hour' in col or 'day' in col:
            num_semi_temporal_features +=1
        if 'host' in col or 'site' in col:
            num_non_temporal_features +=1
        if 'delta' in col:
            temporal_count +=1    
    return num_semi_temporal_features,num_non_temporal_features,temporal_count

# Generate training set with features and labels (from time series data)
def window_gen(x_t,x_nt,x_st,window,horizon):
    xt,yt, xnt,xst = [],[],[],[]    
    for i in range(window,len(x_t)-horizon):
        xt.append(x_t[i-window:i,])
        xnt.append(x_nt[i,:])
        xst.append(x_st[i,:])
        yt.append(x_t[i:i+horizon,:2])
    return xt,xnt,xst,yt

def split_feature_set(df_in):
    """ given a dataframe with a single site host combination will return 3 features sets"""
    # Temporal features
    temporal_features = df_in[['rx_bytes_delta', 'rx_bytes_delta_delta', 'tx_bytes_delta', 'tx_bytes_delta_delta']].values
    # Non-temporal features (one-hot encoded site and host)
    non_temporal_features_list = [col for col in df_in.columns if 'host' in col or 'site' in col]
    non_temporal_features = df_in[non_temporal_features_list].values
    # Semi-temporal features
    semi_temporal_features = df_in[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].values
    return temporal_features, non_temporal_features, semi_temporal_features

def create_time_series_datasets(input_df,window, horizon):
    """
    Construct feature sets and labels suitable for time-series forecasting from the input dataframe.

    This function iterates over each unique 'site_host' within the input dataframe,
    segregates the data for that 'site_host', and uses a sliding window approach to generate
    sequences of observations (features) and corresponding labels for use in training a time-series model.

    Parameters:
    - input_df (pd.DataFrame): The input dataframe containing time-series data and 'site_host' column.
    - window (int): The number of time steps to include in the input feature set (i.e., the look-back period).
    - horizon (int): The number of time steps to predict into the future (i.e., the forecast horizon).

    The function assumes that the 'site_host' column is present in input_df and that
    the split_feature_set and window_gen functions are defined elsewhere to handle
    the separation of features and the creation of time-series windows, respectively.

    Returns:
    - xt (np.array): Numpy array of temporal features for LSTM input.
    - xnt (np.array): Numpy array of non-temporal features.
    - xst (np.array): Numpy array of semi-temporal features.
    - yt (np.array): Numpy array of target values to predict, corresponding to each sequence.
    """
    xt,xst,xnt, yt = [],[],[],[]
    for series in input_df.site_host.unique():
        temp_df = input_df[input_df['site_host'] == series].copy()
        temp_df.drop(columns=['site_host'],inplace=True)
        x_t,x_nt,x_st = split_feature_set(temp_df)

        if len(xt) > 0: # check only scaled, assume it same for both
            xt_temp,xnt_temp,xst_temp, y_temp = window_gen(x_t,x_nt,x_st,window,horizon)
            for xt_array in xt_temp:
                xt.append(xt_array)

            for xnt_array in xnt_temp:
                xnt.append(xnt_array)

            for xst_array in xst_temp:
                xst.append(xst_array)

            for y_array in y_temp:
                yt.append(y_array) 
        else:
            xt,xnt,xst, yt = window_gen(x_t,x_nt,x_st,window,horizon)
    
    xt = np.array(xt)
    xnt = np.array(xnt)
    xst = np.array(xst)
    yt = np.array(yt)
    return xt,xnt,xst, yt

def preprocess_input(tx_bytes, rx_bytes, timestamp, site, host):
    # Convert inputs to NumPy arrays
    tx_bytes_tnsr = tf.reshape(tx_bytes, [-1])
    rx_bytes_tnsr = tf.reshape(rx_bytes, [-1])
    timestamp_tnsr = tf.reshape(timestamp, [-1])
    site_tnsr = tf.reshape(site, [-1])
    host_tnsr = tf.reshape(host, [-1])
    
    return tx_bytes_tnsr, rx_bytes_tnsr, timestamp_tnsr, site_tnsr, host_tnsr

def data_split(xt, xnt, xst, yt, batch_size):
    split_size = int(len(xt) * 0.8)
    
    xt_train, xt_valid = xt[:split_size], xt[split_size:]
    xnt_train, xnt_valid = xnt[:split_size], xnt[split_size:]
    xst_train, xst_valid = xst[:split_size], xst[split_size:]
    yt_train, yt_valid = yt[:split_size], yt[split_size:]
    
    train_ds = tf.data.Dataset.from_tensor_slices(((xt_train, xnt_train, xst_train), yt_train)).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices(((xt_valid, xnt_valid,xst_valid),yt_valid)).batch(batch_size)
    return train_ds, valid_ds, xt_valid, xnt_valid, xst_valid, yt_valid
