import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load Datasets
acquisitions_df = pd.read_csv("./TechAcquisitionPricePrediction/Acquisitions.csv")
acquiring_df = pd.read_csv("./TechAcquisitionPricePrediction/Acquiring Tech Companies.csv")
acquired_df = pd.read_csv("./TechAcquisitionPricePrediction/Acquired Tech Companies.csv")
founders_df = pd.read_csv("./TechAcquisitionPricePrediction/Founders and Board Members.csv")

# 2. Helper function to preprocess a dataset
def preprocess_dataset(df, numeric_columns=None, drop_columns=None):
    df = df.copy()
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Identify numerical and categorical columns
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    # -------------------------------
    # Handle missing values
    # -------------------------------

    # Impute numericals
    if numeric_columns:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(df[numeric_columns])
        scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=df.index)
    else:
        scaled_num_df = pd.DataFrame(index=df.index)

    # Impute categoricals
    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_cats = encoder.fit_transform(df[categorical_columns])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns),
                                      index=df.index)
    else:
        encoded_cat_df = pd.DataFrame(index=df.index)

    # Combine
    final_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)
    return final_df

# 3. Preprocess each dataset

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

