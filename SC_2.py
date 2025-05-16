import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, f_regression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, RandomForestClassifier, StackingClassifier, \
    VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Create directory for saved models and preprocessors
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_preprocessors', exist_ok=True)

# Read the CSV files
acquisitions_df = pd.read_csv("Acquisitions.csv")
acquiring_df = pd.read_csv("Acquiring Tech Companies.csv")
acquired_df = pd.read_csv("Acquired Tech Companies.csv")
founders_df = pd.read_csv("Founders and Board Members.csv")
acquisition_class_df = pd.read_csv("Acquisitions class.csv")

# Merge datasets on Acquisitions ID
acquiring_df_cleaned = acquiring_df.copy()
# Clean and split Companies
acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.split(",")

# Remove whitespace from each company name
acquiring_df_cleaned = acquiring_df_cleaned.explode("Acquisitions ID")
acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.strip()
acquisitions_df["deal size"] = acquisition_class_df["Deal size class"]
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

# Merge for acquiring company
merged_df = merged_df.merge(founders_agg, how='left', left_on='Acquiring Company', right_on='Companies')

merged_df['Year of acquisition announcement'] = merged_df['Year of acquisition announcement'].astype(int)
merged_df['Deal announced on'] = pd.to_datetime(merged_df['Deal announced on'], errors='coerce')
merged_df['acquisition_year'] = merged_df['Deal announced on'].dt.year
merged_df['acquisition_month'] = merged_df['Deal announced on'].dt.month

# Dropping Duplicate columns
duplicate_columns = ['Acquiring Company_Acquirer', 'Acquired by', 'Acquiring Company_Acquirer', 'Deal announced on',
                     'Company', 'Companies']
merged_df.drop(columns=duplicate_columns, inplace=True)

# Clean up numeric columns
merged_df['Price'] = merged_df['Price'].replace(r'[\$,]', '', regex=True).astype(float)
merged_df['Total Funding ($)'] = merged_df['Total Funding ($)'].replace(r'[\$,]', '', regex=True).astype(float)
merged_df['Number of Employees'] = merged_df['Number of Employees'].fillna(0).replace(r'[\,]', '', regex=True).astype(
    int)
merged_df['Year Founded'] = pd.to_numeric(merged_df['Year Founded'], errors='coerce').fillna(0).astype(int)
merged_df['IPO'] = pd.to_numeric(merged_df['IPO'], errors='coerce').fillna(0).astype(int)
merged_df['Year of acquisition announcement'] = pd.to_numeric(merged_df['Year of acquisition announcement'],
                                                              errors='coerce').fillna(0).astype(int)

# Store original data for later reference
pickle.dump(merged_df, open('saved_preprocessors/original_data.pkl', 'wb'))

# Calculate values for imputation and store them
mode_columns = ['Number of Employees (year of last update)', 'acquisition_year', 'acquisition_month']
mode_fill_values = {col: merged_df[col].mode()[0] for col in mode_columns}

mean_columns = ['Year Founded', 'IPO', 'Number of Employees', 'Year Founded_Acquired']
mean_fill_values = {}
for col in mean_columns:
    mean_val = merged_df.loc[merged_df[col] != 0, col].mean()
    mean_fill_values[col] = mean_val

zero_columns = ['Total Funding ($)', 'Number of Acquisitions', 'founders_count']

fill_values = {
    'mode': mode_fill_values,
    'mean': mean_fill_values,
    'zero': zero_columns,
    'empty_string': ['Founders', 'Board Members', 'Acquired Companies'],
    'nothing_string': [
        'News', 'News Link', 'CrunchBase Profile', 'Image', 'Tagline', 'Market Categories', 'Address (HQ)',
        'City (HQ)', 'State / Region (HQ)', 'Country (HQ)', 'Description', 'Homepage', 'Twitter', 'API',
        'CrunchBase Profile_Acquired', 'Image_Acquired', 'Tagline_Acquired', 'Market Categories_Acquired',
        'Address (HQ)_Acquired', 'City (HQ)_Acquired', 'State / Region (HQ)_Acquired',
        'Country (HQ)_Acquired', 'Description_Acquired', 'Homepage_Acquired', 'Twitter_Acquired'
    ]
}

# Save to a pickle file
with open('saved_preprocessors/polynomial_transformer.pkl', 'wb') as f:
    pickle.dump(fill_values, f)

# Mode fill
for col, val in fill_values['mode'].items():
    merged_df[col].fillna(val, inplace=True)

# Mean fill + replace zeros with mean
for col, val in fill_values['mean'].items():
    merged_df[col] = merged_df[col].replace(0, val).fillna(val).astype(int)

# Zero fill
for col in fill_values['zero']:
    merged_df[col].fillna(0, inplace=True)

    # Empty string fill
for col in fill_values['empty_string']:
    merged_df[col].fillna("", inplace=True)

# "Nothing" fill
for col in fill_values['nothing_string']:
    merged_df[col].fillna("Nothing", inplace=True)


# Convert column names to lowercase and replace spaces with underscores
merged_df.columns = [col.strip().lower().replace(' ', '_') for col in merged_df.columns]
dropped_columns = ['acquisitions_id', 'acquisition_profile', 'news',
                   'news_link', 'crunchbase_profile', 'image', 'tagline', 'founders', 'board_members',
                   'address_(hq)', 'description', 'homepage', 'twitter', 'api',
                   'crunchbase_profile_acquired', 'image_acquired',
                   'tagline_acquired', 'address_(hq)_acquired', 'description_acquired', 'homepage_acquired',
                   'twitter_acquired', 'api_acquired']
merged_df.drop(dropped_columns, axis=1, inplace=True)

# Feature engineering
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

# Save the feature engineered dataframe
pickle.dump(merged_df, open('saved_preprocessors/feature_engineered_data.pkl', 'wb'))

# Identify numerical and categorical columns
numeric_columns = ["price", "number_of_employees_(year_of_last_update)", "total_funding_($)", "number_of_acquisitions",
                   "year_founded_acquired", "founders_count", "acquisition_year", "acquisition_month",
                   "year_of_acquisition_announcement", "ipo", "number_of_employees", "year_founded", 'acquired_age',
                   "acquisition_quarter", 'funding_per_employee', 'acquisitions_per_year']

categorical_columns = ['status', 'terms']

string_columns = [col for col in merged_df.columns if
                  col not in categorical_columns and col not in numeric_columns and col not in ["deal_size"]]
print(f"All columns= {numeric_columns, categorical_columns, string_columns}")

# Save column categories
column_categories = {
    'numeric_columns': numeric_columns,
    'categorical_columns': categorical_columns,
    'string_columns': string_columns
}
pickle.dump(column_categories, open('saved_preprocessors/column_categories.pkl', 'wb'))

# Impute numericals
num_imputer = SimpleImputer(strategy='median')
num_imputer.fit(merged_df[numeric_columns])
merged_df[numeric_columns] = num_imputer.transform(merged_df[numeric_columns])
pickle.dump(num_imputer, open('saved_preprocessors/num_imputer.pkl', 'wb'))

scaler = StandardScaler()
scaler.fit(merged_df[numeric_columns])
scaled_nums = scaler.transform(merged_df[numeric_columns])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=merged_df.index)
pickle.dump(scaler, open('saved_preprocessors/scaler.pkl', 'wb'))

# Impute Strings and store vectorizers
vectorizers = {}
merged_df[string_columns] = merged_df[string_columns].fillna('')
merged_df[string_columns] = merged_df[string_columns].astype(str)

vectorized_parts = []
for col in string_columns:
    vectorizer = TfidfVectorizer(max_features=500)
    vectorizer.fit(merged_df[col])
    X = vectorizer.transform(merged_df[col])

    # Save the vectorizer
    vectorizers[col] = vectorizer

    # Prefix the column names with the original column name
    tfidf_df = pd.DataFrame(X.toarray(), columns=[f'{col}_{word}' for word in vectorizer.get_feature_names_out()])
    vectorized_parts.append(tfidf_df)

# Save all vectorizers
pickle.dump(vectorizers, open('saved_preprocessors/text_vectorizers.pkl', 'wb'))

df_vectorized = pd.concat(vectorized_parts, axis=1)

# Impute categoricals
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
cat_imputer.fit(merged_df[categorical_columns])
merged_df[categorical_columns] = cat_imputer.transform(merged_df[categorical_columns])
pickle.dump(cat_imputer, open('saved_preprocessors/cat_imputer.pkl', 'wb'))

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(merged_df[categorical_columns])
encoded_cats = encoder.transform(merged_df[categorical_columns])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns),
                              index=merged_df.index)
pickle.dump(encoder, open('saved_preprocessors/cat_encoder.pkl', 'wb'))

# Combine
final_df = pd.concat([scaled_num_df, encoded_cat_df, df_vectorized, merged_df["deal_size"]], axis=1)

# Apply variance thresholding to remove low variance features
selector = VarianceThreshold(threshold=0.01)
selector.fit(final_df[numeric_columns])
pickle.dump(selector, open('saved_preprocessors/variance_selector.pkl', 'wb'))

# Get the columns selected based on variance
selected_features = final_df[numeric_columns].columns[selector.get_support()]
print(f"Selected features based on variance: {selected_features}")

# Save selected features
pickle.dump(selected_features, open('saved_preprocessors/variance_selected_features.pkl', 'wb'))

# Covariance matrix to identify correlated features
cov_matrix = final_df[numeric_columns].cov()

# Check for high correlations (greater than 0.9 or less than -0.9)
highly_correlated_pairs = []

for i in range(len(cov_matrix.columns)):
    for j in range(i):
        if abs(cov_matrix.iloc[i, j]) > 0.9:  # You can adjust the threshold as needed
            highly_correlated_pairs.append((cov_matrix.columns[i], cov_matrix.columns[j]))

print(f"Highly correlated feature pairs: {highly_correlated_pairs}")

# Save correlated pairs
pickle.dump(highly_correlated_pairs, open('saved_preprocessors/highly_correlated_pairs.pkl', 'wb'))

# Drop one of the correlated features (based on domain knowledge or a threshold)
drop_columns = [col for col1, col2 in highly_correlated_pairs for col in [col1, col2]]
drop_columns = list(set(drop_columns))  # To avoid dropping the same column multiple times

# Save drop columns
pickle.dump(drop_columns, open('saved_preprocessors/drop_columns.pkl', 'wb'))

# Final feature selection after variance and covariance filtering
final_selected_features = [col for col in selected_features if col not in drop_columns]
pickle.dump(final_selected_features, open('saved_preprocessors/final_selected_features.pkl', 'wb'))

X_regress = final_df.copy()
Y_regress = final_df['price']
X_regress = X_regress.drop(['price', 'deal_size'], axis=1)

X_class = final_df.copy()
Y_class = final_df['deal_size']
X_class = X_class.drop(['price', 'deal_size'], axis=1)

# Split the data for regression task
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regress, Y_regress, test_size=0.2,
                                                                    random_state=42)

# Remove price outliers
q_low = y_train_reg.quantile(0.01)
q_high = y_train_reg.quantile(0.99)
mask_price = (y_train_reg >= q_low) & (y_train_reg <= q_high)

# Save outlier thresholds
pickle.dump({'q_low': q_low, 'q_high': q_high}, open('saved_preprocessors/outlier_thresholds.pkl', 'wb'))

# Apply to both
X_train_reg = X_train_reg[mask_price]
y_train_reg = y_train_reg[mask_price]

y_train_log = np.log1p(y_train_reg)
y_test_log = np.log1p(y_test_reg)

# Feature Selection for Regression
selector_regression = SelectKBest(score_func=f_regression, k=43)
selector_regression.fit(X_train_reg, y_train_log)
X_train_selected = selector_regression.transform(X_train_reg)
X_test_selected = selector_regression.transform(X_test_reg)

# Save the regression feature selector
pickle.dump(selector_regression, open('saved_preprocessors/selector_regression.pkl', 'wb'))

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train_selected, y_train_log)

# Save the model
pickle.dump(lr, open('saved_models/linear_regression.pkl', 'wb'))

# Evaluation
y_pred_lr_log = lr.predict(X_test_selected)
y_pred_lr = np.expm1(y_pred_lr_log)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_lr))
print("R² Score:", r2_score(y_test_reg, y_pred_lr))

print(".." * 10)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
poly.fit(X_train_selected)
X_train_poly = poly.transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

# Save the polynomial transformer
pickle.dump(poly, open('saved_preprocessors/polynomial_transformer.pkl', 'wb'))

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_log)

# Save the polynomial regression model
pickle.dump(poly_model, open('saved_models/polynomial_regression.pkl', 'wb'))

# Predict
y_pred_log = poly_model.predict(X_test_poly)
y_pred = np.expm1(y_pred_log)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)
print("Polynomial Regression:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)


# General Feature Selection for ensemble models
selector_general = SelectKBest(score_func=f_regression, k=40)
selector_general.fit(X_train_reg, y_train_reg)
X_train_general = selector_general.transform(X_train_reg)
X_test_general = selector_general.transform(X_test_reg)

# Save the general feature selector
pickle.dump(selector_general, open('saved_preprocessors/selector_general.pkl', 'wb'))

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=44)
rf.fit(X_train_general, y_train_reg)
y_pred_rf = rf.predict(X_test_general)

# Save Random Forest model
pickle.dump(rf, open('saved_models/random_forest_regressor.pkl', 'wb'))

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_general, y_train_reg)
y_pred_xgb = xgb_model.predict(X_test_general)

# Save XGBoost model
pickle.dump(xgb_model, open('saved_models/xgboost_regressor.pkl', 'wb'))

# Feature selection for Ridge
selector_ridge = SelectKBest(score_func=f_regression, k=100)
selector_ridge.fit(X_train_reg, y_train_reg)
X_train_ridge = selector_ridge.transform(X_train_reg)
X_test_ridge = selector_ridge.transform(X_test_reg)

# Save Ridge selector
pickle.dump(selector_ridge, open('saved_preprocessors/selector_ridge.pkl', 'wb'))

# Train Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_ridge, y_train_reg)

# Save Ridge model
pickle.dump(ridge_model, open('saved_models/ridge_regression.pkl', 'wb'))

y_pred_ridge = ridge_model.predict(X_test_ridge)


# Best params for Random Forest via GridSearch
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train_general, y_train_reg)
print(f"Best params for RandomForest: {grid_search_rf.best_params_}")

# Save best Random Forest model
best_rf = RandomForestRegressor(**grid_search_rf.best_params_, random_state=44)
best_rf.fit(X_train_general, y_train_reg)
pickle.dump(best_rf, open('saved_models/best_random_forest_regressor.pkl', 'wb'))

# Feature Selection for Stacking
selector_stacking = SelectKBest(score_func=f_regression, k=200)
selector_stacking.fit(X_train_reg, y_train_reg)
X_train_stacking = selector_stacking.transform(X_train_reg)
X_test_stacking = selector_stacking.transform(X_test_reg)

# Save stacking selector
pickle.dump(selector_stacking, open('saved_preprocessors/selector_stacking.pkl', 'wb'))

# Create stacking model
poly_regressor = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge(alpha=1.0)),
    ('linear', LinearRegression())
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stacking_model.fit(X_train_stacking, y_train_reg)


# Save stacking model
pickle.dump(stacking_model, open('saved_models/stacking_regressor.pkl', 'wb'))

y_pred_stacking = stacking_model.predict(X_test_stacking)
print(f"RandomForest MSE: {mean_squared_error(y_test_reg, y_pred_rf)}, R2: {r2_score(y_test_reg, y_pred_rf)}")
print(f"XGBoost MSE: {mean_squared_error(y_test_reg, y_pred_xgb)}, R2: {r2_score(y_test_reg, y_pred_xgb)}")
print(f"Ridge MSE: {mean_squared_error(y_test_reg, y_pred_ridge)}, R2: {r2_score(y_test_reg, y_pred_ridge)}")
print(f"Stacking Model MSE: {mean_squared_error(y_test_reg, y_pred_stacking)}, R2: {r2_score(y_test_reg, y_pred_stacking)}")

# Classification Task
label_encoder = LabelEncoder()
label_encoder.fit(Y_class)
Y_class_encoded = label_encoder.transform(Y_class)

# Save label encoder
pickle.dump(label_encoder, open('saved_preprocessors/label_encoder.pkl', 'wb'))

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, Y_class_encoded, test_size=0.2,
                                                                    random_state=42, stratify=Y_class)

# Feature selection for classification
cls_selector = SelectKBest(score_func=f_classif, k=175)
cls_selector.fit(X_train_cls, y_train_cls)
X_train_cls_selected = cls_selector.transform(X_train_cls)
X_test_cls_selected = cls_selector.transform(X_test_cls)

# Save classification selector
pickle.dump(cls_selector, open('saved_preprocessors/classification_selector.pkl', 'wb'))

# Train Logistic Regression with different C values
best_lr = None
best_lr_accuracy = 0
print("\n--- Linear SVM ---")
for C in [0.01, 1, 100]:
    lr_model = LogisticRegression(C=C, max_iter=1000)
    lr_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, lr_model.predict(X_test_cls_selected))
    if acc > best_lr_accuracy:
        best_lr_accuracy = acc
        best_lr = lr_model
    print(f"C={C} -> Accuracy: {acc:.4f}")

# Save best logistic regression model
pickle.dump(best_lr, open('saved_models/best_logistic_regression.pkl', 'wb'))

# Train Linear SVM with different C values
best_lsvm = None
best_lsvm_accuracy = 0
print("\n--- Linear SVM ---")
for C in [0.01, 1, 100]:
    svm_model = SVC(kernel='linear', C=C, probability=True)
    svm_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, svm_model.predict(X_test_cls_selected))
    if acc > best_lsvm_accuracy:
        best_lsvm_accuracy = acc
        best_lsvm = svm_model
    print(f"C={C} -> Accuracy: {acc:.4f}")

# Save best linear SVM model
pickle.dump(best_lsvm, open('saved_models/best_linear_svm.pkl', 'wb'))

# Train Polynomial SVM with different degrees
best_psvm = None
best_psvm_accuracy = 0
print("\n--- Poly SVM ---")
for degree in [2, 3, 5]:
    svm_poly_model = SVC(kernel='poly', degree=degree, probability=True)
    svm_poly_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, svm_poly_model.predict(X_test_cls_selected))
    if acc > best_psvm_accuracy:
        best_psvm_accuracy = acc
        best_psvm = svm_poly_model
    print(f"Degree={degree} -> Accuracy: {acc:.4f}")

# Save best polynomial SVM model
pickle.dump(best_psvm, open('saved_models/best_poly_svm.pkl', 'wb'))

# Train KNN with different neighbor values
best_knn = None
best_knn_accuracy = 0
print("\n--- KNN ---")
for k in [3, 5, 11]:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, knn_model.predict(X_test_cls_selected))
    if acc > best_knn_accuracy:
        best_knn_accuracy = acc
        best_knn = knn_model
    print(f"n_neighbors={k} -> Accuracy: {acc:.4f}")

# Save best KNN model
pickle.dump(best_knn, open('saved_models/best_knn.pkl', 'wb'))

# Train Decision Tree with different depths
best_dt = None
best_dt_accuracy = 0
print("\n--- Decision Tree ---")
for depth in [3, 5, 10]:
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, dt_model.predict(X_test_cls_selected))
    if acc > best_dt_accuracy:
        best_dt_accuracy = acc
        best_dt = dt_model
    print(f"max_depth={depth} -> Accuracy: {acc:.4f}")

# Save best decision tree model
pickle.dump(best_dt, open('saved_models/best_decision_tree.pkl', 'wb'))

# Train Random Forest with different estimators
best_rf_cls = None
best_rf_cls_accuracy = 0
print("\n--- Random Forest ---")
for n in [50, 100, 200]:
    rf_cls_model = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_cls_model.fit(X_train_cls_selected, y_train_cls)
    acc = accuracy_score(y_test_cls, rf_cls_model.predict(X_test_cls_selected))
    if acc > best_rf_cls_accuracy:
        best_rf_cls_accuracy = acc
        best_rf_cls = rf_cls_model
    print(f"n_estimators={n} -> Accuracy: {acc:.4f}")

# Save best random forest classifier model
pickle.dump(best_rf_cls, open('saved_models/best_random_forest_classifier.pkl', 'wb'))

# Create dictionary of best models
models = {
    "Logistic Regression": best_lr,
    "Linear SVM": best_lsvm,
    "Poly SVM": best_psvm,
    "KNN": best_knn,
    "Decision Tree": best_dt,
    "Random Forest": best_rf_cls
}

# Create and train Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('Logistic Regression', models["Logistic Regression"]),
        ('Linear SVM', models["Linear SVM"]),
        ('KNN', models["KNN"]),
        ('Decision Tree', models["Decision Tree"]),
        ('Random Forest', models["Random Forest"])
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_clf.fit(X_train_cls_selected, y_train_cls)
stack_preds = stacking_clf.predict(X_test_cls_selected)
print("\nStacking Ensemble Results:")
print(f"Accuracy: {accuracy_score(y_test_cls, stack_preds):.4f}")
# Save stacking classifier
pickle.dump(stacking_clf, open('saved_models/stacking_classifier.pkl', 'wb'))

# Create and train Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('Logistic Regression', models["Logistic Regression"]),
        ('Linear SVM', models["Linear SVM"]),
        ('KNN', models["KNN"]),
        ('Decision Tree', models["Decision Tree"]),
        ('Random Forest', models["Random Forest"])
    ],
    voting='soft'
)

voting_clf.fit(X_train_cls_selected, y_train_cls)

# Save voting classifier
pickle.dump(voting_clf, open('saved_models/voting_classifier.pkl', 'wb'))
voting_preds = voting_clf.predict(X_test_cls_selected)
print("\nVoting Ensemble (Soft) Results:")
print(f"Accuracy: {accuracy_score(y_test_cls, voting_preds):.4f}")

print("All preprocessing steps and models have been saved successfully!")