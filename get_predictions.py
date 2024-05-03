import numpy as np
import json, requests
import pandas as pd
import random
import cmlapi
import os
import time

window = 24

# function(s)
def dataframe_to_json(df):
    """Create a dictionary that will later be converted to a JSON object
    ensure that 
    """
    data = {
        'rx_gbs': df['rx_gbs'].iloc[-(window+1):].tolist(),
        'tx_gbs': df['tx_gbs'].iloc[-(window+1):].tolist(),
        'time': [df['time'].iloc[-1]],  # Only the last time entry
        'link': [df['link'].iloc[-1]],  # Only the last link entry
        'location': [df['location'].iloc[-1]]  # Only the last location entry
    }
    
    # build embedded dictionary step 1
    request_dict = {"request":data}

    # access key will be end point specific
    BackDict = {"accessKey":model_key}
    BackDict.update(request_dict)
    request_dict=BackDict
    
    return request_dict


# read data frame (or table data)
df = pd.read_csv('data/simple_synthetic_data.csv')

# sort data by time 
df.sort_values(by=['time'],inplace=True)
#select first 25

# For starters inference can be made twice a day

# model parameter configuration
model_name = "LSTM-no-metrics"
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
target_model = client.list_all_models(search_filter=json.dumps({"name": model_name}))
model_key = target_model.models[0].access_key
model_url = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model")



stride = 12 # determines frequecy of model inference
load_frequency = 2 # ratio of number of strides per load ground truth
                   # if job runs each hour, then 24 would kick of load 
                   # ground truth job every 24 hours. 

for j,i in enumerate(range(0,df.shape[0]-(window),stride)):
    temp_df = df.iloc[i:window+1+i,:] # 
    # convert input to json
    request_dict = dataframe_to_json(temp_df)
    
    r = requests.post(model_url, data=json.dumps(request_dict), headers={'Content-Type': 'application/json'})
    time.sleep(1)
    
    if j%load_frequency == 0:
        print('kicking of load actual values')
        # pass last date as parameter 
        #temp_df['time'].iloc[-1]
