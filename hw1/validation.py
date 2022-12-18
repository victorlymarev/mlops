from pydantic import BaseModel, Field
from enum import Enum
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import os
import dill
from fastapi import HTTPException
import pandas as pd


class ModelName(str, Enum):
    lin_reg = "linreg"
    forest = "random_forest"


class HpsLinearReg(BaseModel):
    alpha: float = Field(default=0)
    fit_intercept: bool = Field(default=True)
    random_state: int | None = Field(default=None, ge=1)


class HpsRandomForest(BaseModel):
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=100000)
    random_state: int | None = Field(default=None, ge=0)
    n_jobs: int = Field(default=5)


def check_data_path(data_path: str) -> str:
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Файла с таким \
            именем не существует")
    return data_path


def fit_and_save_model(number, hps, data_path, typ):
    '''Функция для обучения модели'''
    model = Ridge(**hps) if typ == 'linreg' else RandomForestRegressor(**hps)
    data_table = pd.read_csv(check_data_path(data_path))
    model.fit(data_table[data_table.columns[:-1]], data_table.iloc[:, -1])

    if not os.path.isdir('models'):
        os.mkdir('./models')

    with open(os.path.join("models", f"{typ}_{number}.pkl"), 'wb') as f:
        dill.dump(model, f)
    return {'Model_name': f"{typ}_{number}.pkl", 'type': typ, 'number': number}