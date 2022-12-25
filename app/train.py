from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import dill
from crud import create_linear_regression, create_random_forest


def choose_model(model_params):
    models = {
        'Линейная регрессия': Ridge,
        'Деревья решений': RandomForestRegressor
    }

    space = {
        'Линейная регрессия': {
            'alpha': model_params['alpha'],
            'fit_intercept': model_params['fit_intercept'],
            'random_state': model_params['random_state']
        },
        'Деревья решений': {
            'random_state': model_params['random_state'],
            'max_depth': model_params['max_depth'],
            'n_estimators': model_params['n_estimators'],
            'n_jobs': model_params['n_jobs']
        }
    }

    model_db = {
        'Линейная регрессия': create_linear_regression,
        'Деревья решений': create_random_forest
    }

    id_params = space[model_params['model_name']].copy()
    id_params['model_id'] = model_params['model_id']

    return models[model_params['model_name']](**space[model_params['model_name']]), id_params, model_db


def fit_model(model_params, data, db):
    '''Функция для обучения модели'''
    assert model_params['model_name'] in ['Линейная регрессия', 'Деревья решений'], ValueError('Такого класса моделей не существует')
    model, id_params, model_db = choose_model(model_params)
    data_table = pd.read_csv(data)
    model.fit(data_table[data_table.columns[:-1]], data_table.iloc[:, -1])

    if not os.path.isdir('models'):
        os.mkdir('./models')

    with open(os.path.join("models", f"{model_params['model_name']}{model_params['model_id']}.pkl"), 'wb') as f:
        dill.dump(model, f)

    model_db[model_params['model_name']](db, id_params)
