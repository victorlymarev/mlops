from fastapi import FastAPI, Query, Path, File, status
from pydantic import BaseModel, Field, root_validator
from enum import Enum
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import dill
import pandas as pd
import os


class ModelName(str, Enum):
    lin_reg = "linreg"
    forest = "random_forest"


app = FastAPI()

@app.get("/models", tags=['get_models'])
async def get_avalable_models():

    """Модели доступные для обучения"""

    return {"Модель 1": "liniar-reg",
            "Модель 2": "random-forest"}


class HpsLinearReg(BaseModel):
    alpha: float = Field(default=0)
    fit_intercept: bool = Field(default=True)
    random_state: int|None = Field(default=None, ge=1)


class HpsRandomForest(BaseModel):
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=100000)
    random_state: int|None = Field(default=None, ge=0)
    n_jobs: int = Field(default=5)


def fit_and_save_model(id, hps, data, type):
    model = Ridge(**hps) if type == 'linreg' else RandomForestRegressor(**hps)
    data_table = pd.read_csv(data)
    model.fit(data_table[data_table.columns[:-1]], data_table.iloc[:, -1])

    with open(f"models\\{type}_{id}.pkl", 'wb') as f:
        dill.dump(model, f)

        # f"{os.getcwd()}\\models\\{type}_{id}.pkl"

# выбор модели сделал через 2 функции, иначе не получается нормально валидировать гиперпараметры
# Возможно, есть красивый способ это сделать, но я пока не понял как
@app.post("/train/liniar-reg", tags=['train_model'], status_code=status.HTTP_201_CREATED)
async def train_model(*, id: int = Query(default=0, ge=0, example=0,
    description='Уникальный идентификатор модели'), hps: HpsLinearReg, data: str = Query(default='data.csv')):
    fit_and_save_model(id, hps.dict(), data, 'linreg')

@app.post("/train/random-forest", tags=['train_model'], status_code=status.HTTP_201_CREATED)
async def train_model(*, id: int = Query(default=0, ge=0, example=0,
    description='Уникальный идентификатор модели'), hps: HpsRandomForest, data: str = Query(default='data.csv')):
    fit_and_save_model(id, hps.dict(), data, 'random_forest')

@app.delete('/delmodel'tags=['delete_model'])
async def del_model(type: ModelName = Query(), id: int = Query(default=0, ge=0, example=0,
    description='Уникальный идентификатор модели')):
    os.remove(f"models\\{type.value}_{id}.pkl")


@app.put('/predict')
async def predict(type: ModelName = Query(),id: int = Query(default=0, ge=0, example=0,
    description='Уникальный идентификатор модели'), data: str = Query(default='data.csv')):
    with open(f"models\\{type.value}_{id}.pkl",
                  'rb') as file:
        model = dill.load(file)
        data_table = pd.read_csv(data)
        return model.predict(data_table[data_table.columns[:-1]])
