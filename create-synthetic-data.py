import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import random

# this file will create a new synthetic data set.. 
# only used to test ML ops. Not to be used for training


# Load historical data
historical_data = pd.read_csv('data/netstats_hourly_4_3.csv')

# Group the data by link and location
grouped_data = historical_data.groupby(['link', 'location'])

# Create a dictionary to store the distributions
distributions = defaultdict(dict)

# Fit distributions for rx_gb and tx_gb
for (link, location), group in grouped_data:
    distributions[(link, location)]['rx_gbs'] = group['rx_gbs'].values
    distributions[(link, location)]['tx_gbs'] = group['tx_gbs'].values

# Function to generate new data
def generate_new_data(link, location, start_date, end_date):
    new_data = []
    current_date = start_date
    while current_date <= end_date:
        rx_gb = np.random.choice(distributions[(link, location)]['rx_gbs'])
        tx_gb = np.random.choice(distributions[(link, location)]['tx_gbs'])
        new_data.append([link, location, current_date, rx_gb, tx_gb])
        current_date += timedelta(hours=1)
    return new_data

# Set the simulation period
start_date = datetime(2024, 5, 1)  # May 1, 2024
end_date = datetime(2024, 5, 31)   # May 31, 2024

# Generate synthetic data
synthetic_data = []
for link, location in distributions.keys():
    synthetic_data.extend(generate_new_data(link, location, start_date, end_date))

# Create a DataFrame from the synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=['link', 'location', 'time', 'rx_gbs', 'tx_gbs'])

# Save the synthetic data to a CSV file
synthetic_df.to_csv('data/synthetic_data.csv', index=False)



# create a subset with single combination of link/location
sample_df = synthetic_df[['link','location']].drop_duplicates().sample()
link_pick = sample_df['link'].values[0]
location_pick = sample_df['location'].values[0]

simple_synt_df = synthetic_df[(synthetic_df.link == link_pick) & (synthetic_df.location == location_pick)]

print('simple_synth_df.shape',simple_synt_df.shape)


simple_synt_df.to_csv('data/simple_synthetic_data.csv', index=False)