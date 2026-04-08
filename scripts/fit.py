# scripts/fit.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score
from catboost import CatBoostClassifier
import yaml
import os
import joblib

# обучение модели
def fit_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # загрузите результат предыдущего шага: initial_data.csv
    data = pd.read_csv('data/initial_data.csv')
    X = data.drop(columns=[params['target_col'], 'end_date'])
    y = data[params['target_col']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18, stratify=y)

    # реализуйте основную логику шага с использованием гиперпараметров

    # обучение модели
    cat_features = X.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    model = CatBoostClassifier(auto_class_weights=params['auto_class_weights'], verbose=False)

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(X_train, y_train)    

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:,0]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'f1: {f1:.3f}, roc_auc: {roc_auc:.3f}')

    # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs('models', exist_ok=True) 
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()