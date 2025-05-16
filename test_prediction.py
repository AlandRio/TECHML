import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


class_flag = False

# Function to get acquisition quarter
def get_quarter(month):
    if pd.notnull(month):
        return ((month - 1) // 3) + 1
    else:
        return np.nan


# Load and merge input files
def load_and_merge_data(acquisitions_file, acquiring_file, acquired_file, founders_file, acquisition_class_file=None):
    try:
        acquisitions_df = pd.read_csv(acquisitions_file)
        acquiring_df = pd.read_csv(acquiring_file)
        acquired_df = pd.read_csv(acquired_file)
        founders_df = pd.read_csv(founders_file)
        if acquisition_class_file:
            acquisition_class_df = pd.read_csv(acquisition_class_file)
        else:
            acquisitions_df['deal_size'] = np.nan

        if class_flag:
            acquisitions_df['Price'] = np.nan

        acquiring_df_cleaned = acquiring_df.copy()
        acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.split(",")
        acquiring_df_cleaned = acquiring_df_cleaned.explode("Acquisitions ID")
        acquiring_df_cleaned["Acquisitions ID"] = acquiring_df_cleaned["Acquisitions ID"].str.strip()

        if acquisition_class_file:
            acquisitions_df["deal_size"] = acquisition_class_df["Deal size class"]

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

        founders_df_cleaned = founders_df.copy()
        founders_df_cleaned["Companies"] = founders_df_cleaned["Companies"].str.split(",")
        founders_df_cleaned = founders_df_cleaned.explode("Companies")
        founders_df_cleaned["Companies"] = founders_df_cleaned["Companies"].str.strip()

        founders_agg = founders_df_cleaned.groupby("Companies").agg({
            "Name": "count"
        }).rename(columns={"Name": "founders_count"}).reset_index()

        merged_df = merged_df.merge(founders_agg, how='left', left_on='Acquiring Company', right_on='Companies')

        merged_df['Year of acquisition announcement'] = pd.to_numeric(merged_df['Year of acquisition announcement'],
                                                                      errors='coerce').fillna(0).astype(int)
        merged_df['Deal announced on'] = pd.to_datetime(merged_df['Deal announced on'], errors='coerce')
        merged_df['acquisition_year'] = merged_df['Deal announced on'].dt.year
        merged_df['acquisition_month'] = merged_df['Deal announced on'].dt.month

        duplicate_columns = ['Acquiring Company_Acquirer', 'Acquired by', 'Deal announced on', 'Company', 'Companies']
        merged_df.drop(columns=[col for col in duplicate_columns if col in merged_df.columns], inplace=True)

        return merged_df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error in data merging: {e}")
        return None


# Preprocess merged data
def preprocess_data(merged_df):
    try:
        preprocessing_values = pickle.load(open('saved_preprocessors/preprocessing_values.pkl', 'rb'))
        column_categories = pickle.load(open('saved_preprocessors/column_categories.pkl', 'rb'))
        num_imputer = pickle.load(open('saved_preprocessors/num_imputer.pkl', 'rb'))
        scaler = pickle.load(open('saved_preprocessors/scaler.pkl', 'rb'))
        vectorizers = pickle.load(open('saved_preprocessors/text_vectorizers.pkl', 'rb'))
        cat_imputer = pickle.load(open('saved_preprocessors/cat_imputer.pkl', 'rb'))
        encoder = pickle.load(open('saved_preprocessors/cat_encoder.pkl', 'rb'))

        numeric_columns = column_categories['numeric_columns']
        categorical_columns = column_categories['categorical_columns']
        string_columns = column_categories['string_columns']

        merged_df['Price'] = merged_df['Price'].replace(r'[\$,]', '', regex=True).astype(float)
        merged_df['Total Funding ($)'] = merged_df['Total Funding ($)'].replace(r'[\$,]', '', regex=True).astype(float)
        merged_df['Number of Employees'] = merged_df['Number of Employees'].fillna(0).replace(r'[\,]', '',
                                                                                              regex=True).astype(int)
        merged_df['Year Founded'] = pd.to_numeric(merged_df['Year Founded'], errors='coerce').fillna(0).astype(int)
        merged_df['IPO'] = pd.to_numeric(merged_df['IPO'], errors='coerce').fillna(0).astype(int)
        merged_df['Year of acquisition announcement'] = pd.to_numeric(merged_df['Year of acquisition announcement'],
                                                                      errors='coerce').fillna(0).astype(int)

        mode_columns = ['Number of Employees (year of last update)', 'acquisition_year', 'acquisition_month']
        for column in mode_columns:
            merged_df[column] = merged_df[column].fillna(preprocessing_values[f'{column}_mode'])

        mean_columns = ['Year Founded', 'IPO', 'Number of Employees', 'Year Founded_Acquired']
        for column in mean_columns:
            mean_value = preprocessing_values[f'{column}_mean']
            merged_df[column] = merged_df[column].fillna(mean_value)
            merged_df[column] = merged_df[column].replace(0, mean_value).astype(int)

        empty_columns = ['Founders', 'Board Members', 'Acquired Companies']
        for column in empty_columns:
            merged_df[column] = merged_df[column].fillna("")

        nothing_columns = ['News', 'News Link', 'CrunchBase Profile', 'Image', 'Tagline', 'Market Categories',
                           'Address (HQ)',
                           'City (HQ)', 'State / Region (HQ)', 'Country (HQ)', 'Description', 'Homepage', 'Twitter',
                           'API',
                           'CrunchBase Profile_Acquired', 'Image_Acquired', 'Tagline_Acquired',
                           'Market Categories_Acquired',
                           'Address (HQ)_Acquired', 'City (HQ)_Acquired', 'State / Region (HQ)_Acquired',
                           'Country (HQ)_Acquired', 'Description_Acquired', 'Homepage_Acquired', 'Twitter_Acquired']
        for column in nothing_columns:
            merged_df[column] = merged_df[column].fillna("Nothing")

        zero_columns = ['Total Funding ($)', 'Number of Acquisitions', 'founders_count']
        for column in zero_columns:
            merged_df[column] = merged_df[column].fillna(0)

        merged_df.columns = [col.strip().lower().replace(' ', '_') for col in merged_df.columns]

        dropped_columns = ['acquisitions_id', 'acquisition_profile', 'news', 'news_link', 'crunchbase_profile', 'image',
                           'tagline', 'founders', 'board_members', 'address_(hq)', 'description', 'homepage', 'twitter',
                           'api',
                           'crunchbase_profile_acquired', 'image_acquired', 'tagline_acquired', 'address_(hq)_acquired',
                           'description_acquired', 'homepage_acquired', 'twitter_acquired', 'api_acquired']
        merged_df.drop(columns=[col for col in dropped_columns if col in merged_df.columns], inplace=True)

        merged_df['year_founded_acquired'] = pd.to_numeric(merged_df['year_founded_acquired'], errors='coerce').astype(
            int)
        merged_df['year_founded'] = pd.to_numeric(merged_df['year_founded'], errors='coerce').astype(int)
        merged_df['acquisition_year'] = pd.to_numeric(merged_df['acquisition_year'], errors='coerce').astype(int)
        merged_df['acquirer_age'] = merged_df['acquisition_year'] - merged_df['year_founded']
        merged_df['acquired_age'] = merged_df['acquisition_year'] - merged_df['year_founded_acquired']
        merged_df["acquisition_quarter"] = merged_df["acquisition_month"].apply(get_quarter)
        merged_df['funding_per_employee'] = merged_df['total_funding_($)'] / (merged_df['number_of_employees'] + 1e-6)
        merged_df['acquisitions_per_year'] = merged_df['number_of_acquisitions'] / (merged_df['acquirer_age'] + 1e-6)

        merged_df[numeric_columns] = num_imputer.transform(merged_df[numeric_columns])
        scaled_nums = scaler.transform(merged_df[numeric_columns])
        scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_columns, index=merged_df.index)

        merged_df[string_columns] = merged_df[string_columns].fillna('')
        merged_df[string_columns] = merged_df[string_columns].astype(str)
        vectorized_parts = []
        for col in string_columns:
            vectorizer = vectorizers[col]
            X = vectorizer.transform(merged_df[col])
            tfidf_df = pd.DataFrame(X.toarray(),
                                    columns=[f'{col}_{word}' for word in vectorizer.get_feature_names_out()])
            vectorized_parts.append(tfidf_df)
        df_vectorized = pd.concat(vectorized_parts, axis=1)

        merged_df[categorical_columns] = cat_imputer.transform(merged_df[categorical_columns])
        encoded_cats = encoder.transform(merged_df[categorical_columns])
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns),
                                      index=merged_df.index)

        final_df = pd.concat([scaled_num_df, encoded_cat_df, df_vectorized, merged_df['deal_size'], merged_df['price']],
                             axis=1)

        return final_df
    except FileNotFoundError as e:
        print(f"Error: Preprocessor file not found - {e}")
        return None
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# Run all saved models
def run_all_models(final_df):
    try:
        # Load preprocessors
        selector_regression = pickle.load(open('saved_preprocessors/selector_regression.pkl', 'rb'))
        selector_general = pickle.load(open('saved_preprocessors/selector_general.pkl', 'rb'))
        selector_ridge = pickle.load(open('saved_preprocessors/selector_ridge.pkl', 'rb'))
        selector_stacking = pickle.load(open('saved_preprocessors/selector_stacking.pkl', 'rb'))
        poly = pickle.load(open('saved_preprocessors/polynomial_transformer.pkl', 'rb'))
        cls_selector = pickle.load(open('saved_preprocessors/classification_selector.pkl', 'rb'))
        label_encoder = pickle.load(open('saved_preprocessors/label_encoder.pkl', 'rb'))

        # Prepare data
        X_regress = final_df.drop(['price', 'deal_size'], axis=1, errors='ignore')
        X_class = final_df.drop(['price', 'deal_size'], axis=1, errors='ignore')
        y_regress = final_df['price'] if 'price' in final_df.columns else None
        y_regress = y_regress.iloc[:, [0]]
        y_class = final_df['deal_size'] if 'deal_size' in final_df.columns else None

        # Initialize results
        predictions = {}
        metrics = {}

        # Regression models
        if not class_flag:
            regression_models = [
                ('linear_regression', selector_regression, lambda x: np.expm1(x)),
                ('polynomial_regression', selector_regression, lambda x: np.expm1(x), poly),
                ('random_forest_regressor', selector_general, lambda x: x),
                ('xgboost_regressor', selector_general, lambda x: x),
                ('ridge_regression', selector_ridge, lambda x: x),
                ('best_random_forest_regressor', selector_general, lambda x: x),
                ('stacking_regressor', selector_stacking, lambda x: x)
            ]
            print("\n=== REGRESSION MODELS ===")
            print(f"{'Model':<30} {'MSE':<10} {'R²':<10}")
            print("-" * 50)
            for model_name, selector, transform, *extra in regression_models:
                try:

                    model = pickle.load(open(f'saved_models/{model_name}.pkl', 'rb'))
                    X_selected = selector.transform(X_regress)
                    if extra:
                        X_selected = extra[0].transform(X_selected)
                    pred = transform(model.predict(X_selected))
                    predictions[f'{model_name}_Price'] = pred
                    mse = mean_squared_error(y_regress, pred)
                    r2 = r2_score(y_regress, pred)
                    metrics[model_name] = {'MSE': mse, 'R²': r2}
                    print(f"{model_name:<30} {mse:.4f}     {r2:.4f}")
                except FileNotFoundError:
                    print(f"Error: Model {model_name}.pkl not found")
                except Exception as e:
                    print(f"Error running {model_name}: {e}")

        # Classification models
        if class_flag:
            classification_models = [
                'best_logistic_regression',
                'best_linear_svm',
                'best_poly_svm',
                'best_knn',
                'best_decision_tree',
                'best_random_forest_classifier',
                'stacking_classifier',
                'voting_classifier'
            ]
            print("\n=== CLASSIFICATION MODELS ===")
            print(f"{'Model':<30} {'Accuracy':<10}")
            print("-" * 50)
            for model_name in classification_models:
                try:

                    model = pickle.load(open(f'saved_models/{model_name}.pkl', 'rb'))
                    X_selected = cls_selector.transform(X_class)
                    pred_encoded = model.predict(X_selected)
                    pred = label_encoder.inverse_transform(pred_encoded)
                    predictions[f'{model_name}_Deal_Size'] = pred
                    if y_class is not None and not y_class.isna().all():
                        valid_indices = y_class.notna()
                        y_class_valid = y_class[valid_indices]
                        pred_encoded_valid = pred_encoded[valid_indices]
                        if len(y_class_valid) > 0:
                            y_class_encoded = label_encoder.transform(y_class_valid)
                            accuracy = accuracy_score(y_class_encoded, pred_encoded_valid)
                            metrics[model_name] = {'Accuracy': accuracy}
                            print(f"{model_name:<30} {accuracy:.4f}")
                        else:
                            print(f"No non-null deal_size values for {model_name}, skipping metrics")
                    else:
                        print(f"No valid deal_size ground truth for {model_name}, skipping metrics")
                except FileNotFoundError:
                    print(f"Error: Model {model_name}.pkl not found")
                except Exception as e:
                    print(f"Error running {model_name}: {e}")

        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('all_models_predictions.csv', index=False)
        print("All predictions saved to 'all_models_predictions.csv'")

        # Save metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv('all_models_metrics.csv', index=False)
        print("All metrics saved to 'all_models_metrics.csv'")

        return predictions_df, metrics_df
    except Exception as e:
        print(f"Error in run_all_models: {e}")
        return None, None


# Main function
def make_predictions(acquisitions_file, acquiring_file, acquired_file, founders_file, acquisition_class_file=None):
    merged_df = load_and_merge_data(acquisitions_file, acquiring_file, acquired_file, founders_file,
                                    acquisition_class_file)
    if merged_df is None:
        print("Failed to load and merge data")
        return

    final_df = preprocess_data(merged_df)
    if final_df is None:
        print("Failed to preprocess data")
        return

    predictions_df, metrics_df = run_all_models(final_df)
    if predictions_df is not None:
        print("All models executed successfully!")


if __name__ == "__main__":
    acquisitions_file = "Acquisitions Class.csv"
    acquiring_file = "Acquiring Tech Companies.csv"
    acquired_file = "Acquired Tech Companies.csv"
    founders_file = "Founders and Board Members.csv"
    class_flag = True
    make_predictions(acquisitions_file, acquiring_file, acquired_file, founders_file,acquisitions_file)