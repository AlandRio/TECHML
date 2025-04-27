import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

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
    right_on='Acquisitions ID',
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
# print(merged_df.columns)
merged_df['Year of acquisition announcement'] = merged_df['Year of acquisition announcement'].astype(int)
merged_df['Deal announced on'] = pd.to_datetime(merged_df['Deal announced on'], errors='coerce')
merged_df['acquisition_year'] = merged_df['Deal announced on'].dt.year
merged_df['acquisition_month'] = merged_df['Deal announced on'].dt.month

# Dropping Duplicate columns
duplicate_columns = ['Acquiring Company_Acquirer', 'Acquired by', 'Acquiring Company_Acquirer', 'Deal announced on',
                     'Company', 'Companies']
merged_df.drop(columns=duplicate_columns, inplace=True)

merged_df['Price'] = merged_df['Price'].replace(r'[\$,]', '', regex=True).astype(float)
merged_df['Total Funding ($)'] = merged_df['Total Funding ($)'].replace(r'[\$,]', '', regex=True).astype(float)
merged_df['Number of Employees'] = merged_df['Number of Employees'].fillna(0).replace(r'[\,]', '', regex=True).astype(
    int)
mean_value = merged_df['Number of Employees'][merged_df['Number of Employees'] != 0].mean()
merged_df['Number of Employees'].replace(0, mean_value)
merged_df['Year Founded'] = pd.to_numeric(merged_df['Year Founded'], errors='coerce').fillna(0).astype(int)
merged_df['IPO'] = pd.to_numeric(merged_df['IPO'], errors='coerce').fillna(0).astype(int)
merged_df['Year of acquisition announcement'] = pd.to_numeric(merged_df['Year of acquisition announcement'],
                                                              errors='coerce').fillna(0).astype(int)

# Filling NaN Values
# Mode Columns
mode_columns = ['Number of Employees (year of last update)', 'acquisition_year', 'acquisition_month']
for column in mode_columns:
    mode_value = merged_df[column].mode()[0]
    merged_df[column] = merged_df[column].fillna(mode_value)
# Mean Columns
mean_columns = ['Year Founded', 'IPO', 'Number of Employees', 'Year Founded_Acquired', 'Year Founded_Acquired']
for column in mean_columns:
    mean_value = merged_df[column][merged_df[column] != 0].mean()
    merged_df[column] = merged_df[column].fillna(mean_value)
    merged_df[column] = merged_df[column].replace(0, mean_value).astype(int)
# Empty Columns
empty_columns = ['Founders', 'Board Members', 'Acquired Companies', ]
for column in empty_columns:
    merged_df[column] = merged_df[column].fillna("")
# Nothing Columns
nothing_columns = ['News', 'News Link', 'CrunchBase Profile', 'Image', 'Tagline', 'Market Categories', 'Address (HQ)',
                   'City (HQ)', 'State / Region (HQ)', 'Country (HQ)', 'Description', 'Homepage', 'Twitter', 'API',
                   'CrunchBase Profile_Acquired', 'Image_Acquired', 'Tagline_Acquired', 'Market Categories_Acquired',
                   'Address (HQ)_Acquired', 'City (HQ)_Acquired', 'State / Region (HQ)_Acquired',
                   'Country (HQ)_Acquired', 'Description_Acquired', 'Homepage_Acquired', 'Twitter_Acquired']
for column in nothing_columns:
    merged_df[column] = merged_df[column].fillna("Nothing")
# Zero Columns
zero_columns = ['Total Funding ($)', 'Number of Acquisitions', 'founders_count']
for column in zero_columns:
    merged_df[column] = merged_df[column].fillna(0)

# Convert column names to lowercase and replace spaces with underscores
merged_df.columns = [col.strip().lower().replace(' ', '_') for col in merged_df.columns]
dropped_columns = ['acquisitions_id', 'acquired_company', 'acquiring_company', 'acquisition_profile', 'news',
                   'news_link', 'crunchbase_profile', 'image', 'tagline', 'founders', 'board_members',
                   'address_(hq)', 'description', 'homepage', 'twitter', 'api',
                   'crunchbase_profile_acquired', 'image_acquired',
                   'tagline_acquired', 'address_(hq)_acquired', 'description_acquired', 'homepage_acquired',
                   'twitter_acquired', 'api_acquired']
merged_df.drop(dropped_columns, axis=1, inplace=True)

# print(f"strings: {string_columns}")
# -------------------------------
# Handle missing values
# -------------------------------


# feature engineering
merged_df['year_founded_acquired'] = pd.to_numeric(merged_df['year_founded_acquired'], errors='coerce').astype(int)
merged_df['year_founded'] = pd.to_numeric(merged_df['year_founded'], errors='coerce').astype(int)
merged_df['acquisition_year'] = pd.to_numeric(merged_df['acquisition_year'], errors='coerce').astype(int)

merged_df['acquirer_age'] = merged_df['acquisition_year'] - merged_df['year_founded']
merged_df['acquired_age'] = merged_df['acquisition_year'] - merged_df['year_founded_acquired']


def get_quarter(month):
    if pd.notnull(month):
        return ((month - 1) // 3) + 1
    else:
        return np.nan


merged_df["acquisition_quarter"] = merged_df["acquisition_month"].apply(get_quarter)

merged_df['funding_per_employee'] = merged_df['total_funding_($)'] / (merged_df['number_of_employees'] + 1e-6)

merged_df['acquisitions_per_year'] = merged_df['number_of_acquisitions'] / (merged_df['acquirer_age'] + 1e-6)

# Identify numerical and categorical columns
numeric_columns = ["price", "number_of_employees_(year_of_last_update)", "total_funding_($)", "number_of_acquisitions",
                   "year_founded_acquired", "founders_count", "acquisition_year", "acquisition_month",
                   "year_of_acquisition_announcement", "ipo", "number_of_employees", "year_founded", 'acquired_age',
                   "acquisition_quarter", 'funding_per_employee', 'acquisitions_per_year']
# print(f"Numbers: {numeric_columns}")
categorical_columns = ['status', 'terms']

string_columns = [col for col in merged_df.columns if col not in categorical_columns and col not in numeric_columns]
print(f"All columns= {numeric_columns, categorical_columns, string_columns}")
# Impute numericals
num_imputer = SimpleImputer(strategy='median')
merged_df[numeric_columns] = num_imputer.fit_transform(merged_df[numeric_columns])
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(merged_df[numeric_columns])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=merged_df.index)

# Impute Strings
vectorized_parts = []
# print(string_columns)
merged_df[string_columns] = merged_df[string_columns].fillna('')
merged_df[string_columns] = merged_df[string_columns].astype(str)
for col in string_columns:
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(merged_df[col])

    # Prefix the column names with the original column name
    tfidf_df = pd.DataFrame(X.toarray(), columns=[f'{col}_{word}' for word in vectorizer.get_feature_names_out()])

    vectorized_parts.append(tfidf_df)
df_vectorized = pd.concat(vectorized_parts, axis=1)

# Impute categoricals
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
merged_df[categorical_columns] = cat_imputer.fit_transform(merged_df[categorical_columns])
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(merged_df[categorical_columns])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns),
                              index=merged_df.index)
# Combine
final_df = pd.concat([scaled_num_df, encoded_cat_df, df_vectorized], axis=1)
# print(final_df)


# Apply variance thresholding to remove low variance features
selector = VarianceThreshold(threshold=0.01)  # You can adjust the threshold as needed
X_var = selector.fit_transform(final_df[numeric_columns])

# Get the columns selected based on variance
selected_features = final_df[numeric_columns].columns[selector.get_support()]
print(f"Selected features based on variance: {selected_features}")

# Covariance matrix to identify correlated features
cov_matrix = final_df[numeric_columns].cov()

# Check for high correlations (greater than 0.9 or less than -0.9)
highly_correlated_pairs = []

for i in range(len(cov_matrix.columns)):
    for j in range(i):
        if abs(cov_matrix.iloc[i, j]) > 0.9:  # You can adjust the threshold as needed
            highly_correlated_pairs.append((cov_matrix.columns[i], cov_matrix.columns[j]))

print(f"Highly correlated feature pairs: {highly_correlated_pairs}")

# Drop one of the correlated features (based on domain knowledge or a threshold)
drop_columns = [col for col1, col2 in highly_correlated_pairs for col in [col1, col2]]
drop_columns = list(set(drop_columns))  # To avoid dropping the same column multiple times

# Final feature selection after variance and covariance filtering
final_selected_features = [col for col in selected_features if col not in drop_columns]

X = final_df.copy()
Y = final_df['price']
X = X.drop('price', axis=1)
# print(X)
# print(Y)

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Remove price outliers
q_low = y_train.quantile(0.01)
q_high = y_train.quantile(0.99)
mask_price = (y_train >= q_low) & (y_train <= q_high)

# Apply to both
X_train = X_train[mask_price]
y_train = y_train[mask_price]

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Step 1: Feature Selection
selector = SelectKBest(score_func=f_regression, k=250)  # Try 250 top features (adjust k if you want)
X_train_selected = selector.fit_transform(X_train, y_train_log)
X_test_selected = selector.transform(X_test)

# Initialize and train the model

lr = LinearRegression()
lr.fit(X_train_selected, y_train_log)

# Evaluation
y_pred_lr_log = lr.predict(X_test_selected)
y_pred_lr = np.expm1(y_pred_lr_log)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

print(".." * 10)

# Create polynomial features (change degree as needed)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train_log)

# Predict
y_pred_log = model.predict(X_test_poly)
y_pred = np.expm1(y_pred_log)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Polynomial Regression:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

residuals_lr = y_test - y_pred_lr
residuals_poly = y_test - y_pred

plt.figure(figsize=(12, 7))

# Scatter with color based on absolute error for Linear Regression
scatter_lr = plt.scatter(
    y_test, y_pred_lr,
    c=np.abs(residuals_lr), cmap='coolwarm', alpha=0.7, edgecolors='k', label='Linear Regression'
)

# Scatter with color based on absolute error for Polynomial Regression
scatter_poly = plt.scatter(
    y_test, y_pred,
    c=np.abs(residuals_poly), cmap='viridis', alpha=0.7, edgecolors='k', label='Polynomial Regression'
)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'g--', lw=2, label='Perfect Prediction'
)

# Linear Regression Fit Line (Manual)
coeffs_lr = np.polyfit(y_test, y_pred_lr, 1)
regression_line_lr = np.poly1d(coeffs_lr)
x_vals = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x_vals, regression_line_lr(x_vals), color='black', linestyle='-', label='Linear Regression Fit')

# Polynomial Regression Fit Line (Manual)
coeffs_poly = np.polyfit(y_test, y_pred, 2)  # Degree 2 for Polynomial
regression_line_poly = np.poly1d(coeffs_poly)
plt.plot(x_vals, regression_line_poly(x_vals), color='orange', linestyle='--', label='Polynomial Regression Fit')

# Colorbars
cbar_lr = plt.colorbar(scatter_lr)
cbar_lr.set_label('Absolute Error (LR)')

cbar_poly = plt.colorbar(scatter_poly)
cbar_poly.set_label('Absolute Error (Poly)')

# Labels and Title
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression: Actual vs Predicted (Colored by Error)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
