# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    
    # прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    
    data = pd.read_csv('data/initial_data.csv')

    X = data.drop(columns=[params['target_col']])
    y = data[params['target_col']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18, stratify=y)

    # загрузите результат прошлого шага: fitted_model.pkl
    with open('models/fitted_model.pkl', 'rb') as fd:
        pipeline = joblib.load(fd)
    
    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    # Проверка качества на кросс-валидации
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        pipeline,
        X_test,
        y_test,
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    print(f'cv_res: {cv_res}')

    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs('cv_results', exist_ok=True) 
    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_res, fd, indent=4)        
        

if __name__ == '__main__':
    evaluate_model()