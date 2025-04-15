import numpy as np
import pandas as pd

from preprocessing import preprocess_dataset

# Read the CSV files
acquired_df = pd.read_csv('Acquired Tech Companies.csv')
acquiring_df = pd.read_csv('Acquiring Tech Companies.csv')
acquisitions_df = pd.read_csv('Acquisitions.csv')
founders_df = pd.read_csv('Founders and Board Members.csv')

#  Preprocess each dataset

# --- Acquisitions remove $ from price and edit date format
acquisitions_df['Price'] = acquisitions_df['Price'].replace('[\$,]', '', regex=True).astype(float)
# print(acquisitions_df['Price'])
acquisitions_df['Deal announced on'] = pd.to_datetime(acquisitions_df['Deal announced on'], errors='coerce', dayfirst=True)
acquisitions_df['acquisition_year'] = acquisitions_df['Deal announced on'].dt.year
acquisitions_df['acquisition_month'] = acquisitions_df['Deal announced on'].dt.month
# print(acquisitions_df['acquisition_month'])
acquisitions_df.drop(columns=['Deal announced on'], inplace=True)
preprocessed_acquisitions = preprocess_dataset(acquisitions_df)

# --- Acquiring companies
acquiring_df['Total Funding ($)'] = acquiring_df['Total Funding ($)'].replace('[\$,]', '', regex=True).astype(float)
acquiring_df['Year Founded'] = pd.to_numeric(acquiring_df['Year Founded'], errors='coerce')
preprocessed_acquiring = preprocess_dataset(acquiring_df)

# --- Acquired companies
acquired_df['Year Founded'] = pd.to_numeric(acquired_df['Year Founded'], errors='coerce')
preprocessed_acquired = preprocess_dataset(acquired_df)

# --- Founders
preprocessed_founders = preprocess_dataset(founders_df)
