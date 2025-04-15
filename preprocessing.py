import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_dataset(df):
    df = df.copy()  # Work on a copy to avoid modifying the original dataframe
    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Handle missing values in numeric columns using the mean
    if numeric_columns:
        num_imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

        # Scale numeric features using StandardScaler (zero mean, unit variance)
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(df[numeric_columns])
        scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=df.index)
    else:
        scaled_num_df = pd.DataFrame(index=df.index)  # Empty if no numeric columns

    # Handle missing values in categorical columns by filling with 'Unknown'
    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

        # One-hot encode categorical columns
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_cats = encoder.fit_transform(df[categorical_columns])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns), index=df.index)
    else:
        encoded_cat_df = pd.DataFrame(index=df.index)  # Empty if no categorical columns

    # Combine the scaled numeric and encoded categorical features
    final_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

    return final_df
