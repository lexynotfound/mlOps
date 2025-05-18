import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


def preprocess_data(input_file, output_dir):
    """
    Automatically preprocess the career form dataset

    Args:
        input_file (str): Path to raw CSV file
        output_dir (str): Directory to save processed data and pipeline
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_file)

    # Define feature groups
    numerical_features = ['Expected salary (IDR)']
    categorical_features = ['Gender', 'Marital status', 'Highest formal of education',
                            'Current status', 'Experience']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ],
        remainder='drop'
    )

    # Fit and transform data
    processed_data = preprocessor.fit_transform(df)

    # Save preprocessor for later use
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    # Save processed data
    np.save(os.path.join(output_dir, 'processed_data.npy'), processed_data)

    # Save feature names
    feature_names = (
            numerical_features +
            preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
    )

    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save target extraction (Desired positions)
    desired_positions = df['Desired positions'].tolist()
    pd.DataFrame({'desired_positions': desired_positions}).to_csv(
        os.path.join(output_dir, 'target_data.csv'), index=False
    )

    print(f"Preprocessing completed. Files saved to {output_dir}")


if __name__ == "__main__":
    preprocess_data('../dataset/forminator-career-form-250124070425.csv', 'dataset/career_form_preprocessed')