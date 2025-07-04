# brain.py

import pickle
import pandas as pd

# Load trained components
def load_artifacts(model_path='churn_model.pkl', scaler_path='churn_scaler.pkl', feature_path='churn_features.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

# Predict churn probability
def predict_churn(user_input_dict, model, scaler, feature_names):
    input_filled = {feature: user_input_dict.get(feature, 0) for feature in feature_names}
    input_df = pd.DataFrame([input_filled])
    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)[0][1]
    label = model.predict(scaled_input)[0]
    return label, prob
