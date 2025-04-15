import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Read the CSV files
acquisitions_df = pd.read_csv("Acquisitions.csv")
acquiring_df = pd.read_csv("Acquiring Tech Companies.csv")
acquired_df = pd.read_csv("Acquired Tech Companies.csv")
founders_df = pd.read_csv("Founders and Board Members.csv")

# Merge datasets on Acquisitions ID
acquiring_df_cleaned = acquiring_df.copy()
# Clean and split Companies
acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.split(",")


# Remove whitespace from each company name
acquiring_df_cleaned = acquiring_df_cleaned.explode("Acquisitions ID")
acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.strip()

merged_df = acquisitions_df.merge(
    acquiring_df_cleaned,
    how='left',
    left_on='Acquisitions ID',
    right_on= 'Acquisitions ID',
    suffixes=('', '_Acquirer')
)

merged_df = merged_df.merge(
    acquired_df,
    how='left',
    on='Acquisitions ID',
    suffixes=('', '_Acquired')
)
# Assume founders_df is your original founders file
founders_df_cleaned = founders_df.copy()

# Clean and split Companies
founders_df_cleaned["Companies"] = founders_df_cleaned["Companies"].str.split(",")


# Remove whitespace from each company name
founders_df_cleaned = founders_df_cleaned.explode("Companies")
founders_df_cleaned["Companies"] = founders_df_cleaned["Companies"].str.strip()

founders_agg = founders_df_cleaned.groupby("Companies").agg({
    "Name": "count"
}).rename(columns={"Name": "founders_count"}).reset_index()
# ---------------------------
# 5. Merge Founders Info with Main Dataset
# ---------------------------
# Merge for acquiring company
merged_df = merged_df.merge(founders_agg, how='left', left_on='Acquiring Company', right_on='Companies')

# merged_df.to_csv("accquir2ing3.csv", index=False)
print(merged_df.columns)
merged_df['Year of acquisition announcement'] = merged_df['Year of acquisition announcement'].astype(int)
merged_df['Deal announced on'] = pd.to_datetime(merged_df['Deal announced on'], errors='coerce')
merged_df['acquisition_year'] = merged_df['Deal announced on'].dt.year
merged_df['acquisition_month'] = merged_df['Deal announced on'].dt.month

# print(acquisitions_df['acquisition_month'])
merged_df.drop(columns=['Deal announced on'], inplace=True)

merged_df['Price'] = merged_df['Price'].replace('[\$,]', '', regex=True).astype(float)
merged_df['Total Funding ($)'] = merged_df['Total Funding ($)'].replace('[\$,]', '', regex=True).astype(float)
merged_df['Number of Employees'] = merged_df['Number of Employees'].fillna(0).replace('[\,]', '', regex=True).astype(int)

merged_df['Year Founded'] = pd.to_numeric(merged_df['Year Founded'], errors='coerce').fillna(0).astype(int)
merged_df['IPO'] = pd.to_numeric(merged_df['IPO'], errors='coerce').fillna(0).astype(int)
merged_df['Year of acquisition announcement'] = pd.to_numeric(merged_df['Year of acquisition announcement'], errors='coerce').fillna(0).astype(int)




# Convert column names to lowercase and replace spaces with underscores
merged_df.columns = [col.strip().lower().replace(' ', '_') for col in merged_df.columns]
# Identify numerical and categorical columns
numeric_columns = ["price", "number_of_employees_(year_of_last_update)", "total_funding_($)", "number_of_acquisitions", "year_founded_acquired", "founders_count", "acquisition_year", "acquisition_month", "year_of_acquisition_announcement", "ipo", "number_of_employees"]
print(f"Numbers: {numeric_columns}")
categorical_columns = ['status','terms']

string_columns = [col for col in merged_df.columns if col not in categorical_columns and col not in numeric_columns]

print(f"strings: {string_columns}")
# -------------------------------
# Handle missing values
# -------------------------------

# Impute numericals
num_imputer = SimpleImputer(strategy='median')
merged_df[numeric_columns] = num_imputer.fit_transform(merged_df[numeric_columns])
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(merged_df[numeric_columns])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=merged_df.index)

# Impute Strings

# Impute categoricals
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
merged_df[categorical_columns] = cat_imputer.fit_transform(merged_df[categorical_columns])
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_cats = encoder.fit_transform(merged_df[categorical_columns])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names(categorical_columns), index=merged_df.index)
# Combine
final_df = pd.concat([scaled_num_df, encoded_cat_df], axis=1)
print(final_df)
useless_columns = ['']

final_df.to_csv("accquir2222211ing2221.csv", index=False)
