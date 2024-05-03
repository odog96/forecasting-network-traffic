import pandas as pd
import numpy as np
import os
import cdsw
import joblib

from joblib import dump, load

import tensorflow as tf
from tensorflow import keras


import json

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense ,Input, LSTM, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# remove if not required. Check first
#from sklearn.pipeline import make_pipeline 

# Transform Function

window = 24
horizon = 6

# Load the saved OneHotEncoder fit object
ohefit = joblib.load("/home/cdsw/model/ohefit.save")
print('got onefit')

# skipping scaling for now
# Load the saved StandardScaler object
#scaler = joblib.load("/home/cdsw/model/scaler.save")

# using root mean square error to punish model more heavily for missing spikes
@tf.keras.utils.register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def feature_create(df_in):
    df = df_in.copy()
    
    df.loc[:,'link_loc'] = df['link'] + "_" + df['location']
    
    df['rx_gbs'] = pd.to_numeric(df['rx_gbs'], errors='coerce')
    df['tx_gbs'] = pd.to_numeric(df['tx_gbs'], errors='coerce')
    
    df['rx_gbs_delta'] = df.groupby('link_loc')['rx_gbs'].diff()

    # Calculate tx_bytes_delta_delta
    df['tx_gbs_delta'] = df.groupby('link_loc')['tx_gbs'].diff()

    # The first delta_delta value for each site_host group will be NaN because there's no previous value to subtract from.
    # You might want to fill these NaN values depending on your requirements, for example, with 0s:
    df['rx_gbs_delta'] = df['rx_gbs_delta'].fillna(0)
    df['tx_gbs_delta'] = df['tx_gbs_delta'].fillna(0)
    
    # ensure 'time' is a datetime64 type
    df['time'] = pd.to_datetime(df['time'])

    # Extract hour of day
    df['hour_of_day'] = df['time'].dt.hour

    # Extract day of the week (Monday=0, Sunday=6)
    df['day_of_week'] = df['time'].dt.dayofweek
    
        # Encode 'hour_of_day' cyclically
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)

    # Encode 'day_of_week' cyclically
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
    df.drop(columns=['hour_of_day','day_of_week'],inplace=True)

    # Transform the new data using the loaded OneHotEncoder fit object
    new_data_encoded = ohefit.transform(df[['link', 'location']])
    df_encoded = pd.concat([df,new_data_encoded],axis = 1).drop(columns= ['link', 'location','link_loc'])
    
    #ensure df is sorted by time 
    df_encoded.sort_values(by=['time'],ascending=True,inplace=True)
  
    # scale
    #features_to_scale = [col for col in df_encoded.columns if 'delta' in col]
    # Transform the selected columns using the loaded StandardScaler
    #df_encoded[features_to_scale] = scaler.transform(df_encoded[features_to_scale])
    
    # Temporal features, ensure you pass just number of observations equal to window
    temporal_features = df_encoded[['rx_gbs', 'rx_gbs_delta', 'tx_gbs', 'tx_gbs_delta']].tail(window).values
    temporal_features = temporal_features.reshape(1, window, 4)  # Reshaping to match the input shape expected by the LSTM

    # Non-temporal features (one-hot encoded site and host)
    non_temporal_features_list = [col for col in df_encoded.columns if 'link' in col or 'location' in col]

    # Select the last row for non-temporal features and reshape
    non_temporal_features = df_encoded[non_temporal_features_list].iloc[-1].values
    non_temporal_features = non_temporal_features.reshape(1, -1)  # Reshaping to 1 row, with columns inferred

    # Semi-temporal features
    semi_temporal_features = df_encoded[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].iloc[-1].values
    semi_temporal_features = semi_temporal_features.reshape(1, 4)  # Reshaping to match the expected input shape
    
    del(df)
    del(df_encoded)
    
    return temporal_features, non_temporal_features, semi_temporal_features
    

@cdsw.model_metrics
def forecast_traffic(args):
    print(args)  

    
    input_ts_size = np.array(args['rx_gbs']).shape[0]
    rx_bytes = np.array(args['rx_gbs']).reshape((input_ts_size,1))
    tx_bytes = np.array(args['tx_gbs']).reshape((input_ts_size,1))
    
    # Check if the input length is less than the required window + 1
    if rx_bytes.shape[0] < window + 1 or tx_bytes.shape[0] < window + 1:
        print("Error: Insufficient input length for rx_bytes or tx_bytes. Required:", window + 1)
        sys.exit(1)  # Exit with a non-zero exit code to indicate error
    # Check if the inputs are of the same size
    if rx_bytes.shape[0] != tx_bytes.shape[0]:
        print("Error: rx_bytes and tx_bytes must be of the same size. rx_bytes of size:", rx_bytes.shape[0])
        print("tx_bytes of size:", tx_bytes.shape[0])
        sys.exit(1)  # Exit with a non-zero exit code to indicate error    
          
    link = args['link']
    location = args['location']
    time = args['time']
    
    # Repeat single values input_ts_size times to match the length of the series
    link_array = np.full((input_ts_size, 1), link)
    location_array = np.full((input_ts_size, 1), location)
    time_array = np.full((input_ts_size, 1), time)
    
    # Combine all arrays into a single numpy array
    combined_array = np.hstack((time_array, rx_bytes, tx_bytes, link_array, location_array))
    
    # Create a DataFrame from the combined array
    df = pd.DataFrame(combined_array, columns=['time', 'rx_gbs', 'tx_gbs', 'link', 'location'])
    
    # data prep
    temporal_features, non_temporal_features, semi_temporal_features = feature_create(df)
    
    print('returned features')
    
    # Model loading
    model_path = "/home/cdsw/model/lstm_model.keras"  # Consider making this configurable
    if os.path.exists(model_path):
        print('path exists')
        recon_model = keras.models.load_model(model_path)
    else:
        return {"error": "Model not found at specified path."}
    
    # Combine into a list for prediction as the model expects
    model_input = [temporal_features, non_temporal_features, semi_temporal_features]
    
    print('model input built')
    
    prediction = recon_model.predict(model_input)
    
    print('prediction made. See here:',prediction)
    
    start_ts = pd.to_datetime(df.time[:1].values[0]) + pd.Timedelta('1H')  # Start from the next timestamp, 1 hour later
    future_ts = pd.date_range(start=start_ts, periods=horizon, freq='1H')  # Generate timestamps in 1-hour increments
        
    output = {'rx_bytes': list(prediction[:, :,0]),'tx_bytes': list(prediction[:,:, 1]),\
    'time': [time.isoformat() for time in future_ts]}  # Convert each timestamp to ISO 8601 string format

    
        # Track inputs
        
    # cdsw.track_metric("input_rx_data", list(rx_bytes))
    # cdsw.track_metric("input_tx_data", list(tx_bytes))
    cdsw.track_metric("input_rx_data", args['rx_gbs'])
    #cdsw.track_metric("input_tx_data", args['tx_gbs'])

    print('done with input metrics recorded')
    
    # Track our prediction
#    cdsw.track_metric("rx bytes prediction", list(prediction[:, :,0]))
#    cdsw.track_metric("tx bytes prediction", list(prediction[:, :,1]))

    return output

#sample_input = {'rx_gbs': [7.914413843,   6.786963412,  5.236912162,  5.705485475,  6.112945307,  11.990017718,
#   8.899829778,  9.057491851,  10.068572354,  13.049322433,  11.579918355,  16.804975855000002,  18.285142227,
#   18.292477326,  16.187642118,  15.622562139,  16.046497403,  18.531759137,  15.580055183,  11.249281512,
#   11.438814856,  8.05654827,  11.968518351,  9.214088829,  7.432552623],
#  'tx_gbs': [3.052795016,  3.155528904,  2.669568002,  4.243171256,  4.652467422,  3.030165471,  3.128042709,
#   2.544871279,  2.485775424,  7.341796584,  6.062790366,  7.671077121,  8.435752711,  8.243963607,
#   7.52510098,  7.065984825999999,  7.758155361,  7.310517563999999,  7.685779896,  7.393583334,  4.166739866,
#   2.975712865,  2.916288219,  3.361139273,  3.105882991],
# 'time': ['2024-04-26 02:00:00'],
#  'link': ['Verizon'],
#  'location': ['US-DC4']}


#forecast_traffic(sample_input)


# # sample_output = {'rx_gbs': [7.914413843,   6.786963412,  5.236912162,  5.705485475,  6.112945307,  11.990017718],
#  'tx_gbs': [3.052795016,  3.155528904,  2.669568002,  4.243171256,  4.652467422,  3.030165471,  3.128042709],'time':[Timestamp('2024-03-15 16:55:00+0000', tz='UTC'),
#  Timestamp('2024-03-15 17:55:00+0000', tz='UTC'),
#  Timestamp('2024-03-15 18:55:00+0000', tz='UTC'),
#  Timestamp('2024-03-15 19:55:00+0000', tz='UTC'),
#  Timestamp('2024-03-15 20:55:00+0000', tz='UTC'),
#  Timestamp('2024-03-15 21:55:00+0000', tz='UTC')]