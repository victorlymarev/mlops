from fastapi import FastAPI, Query, HTTPException
import uvicorn
import dill
import pandas as pd
import os
from validation import (ModelName,
                        HpsLinearReg,
                        HpsRandomForest,
                        check_data_path,
                        fit_and_save_model
                        )

app = FastAPI()


@app.get("/models",
         tags=['Получение доступных для обучения моделей'],
         status_code=200)
async def get_avalable_models():

    """Классы моделей, доступные для обучения"""

    return {"Модель 1": "liniar-reg",
            "Модель 2": "random-forest"}


# выбор модели сделал через 2 функции,
# иначе не получается нормально валидировать гиперпараметры
# Возможно, есть красивый способ это сделать, но я пока не понял как
@app.post("/train/liniar-reg", tags=['Обучение моделей'], status_code=201)
async def train_reg(*, number: int = Query(default=0, ge=0, example=0,
                                           description='Уникальный \
                                                идентификатор модели'),
                    hps: HpsLinearReg,
                    data_path: str = Query(default='data.csv',
                    description='Путь к файлу с данными')
                    ):

    '''Обучение линейной регрессии'''

    return fit_and_save_model(number, hps.dict(), data_path, 'linreg')


@app.post("/train/random-forest", tags=['Обучение моделей'], status_code=201)
async def train_forest(*, number: int = Query(default=0, ge=0, example=0,
                                              description='Уникальный \
                                                идентификатор модели'),
                       hps: HpsRandomForest,
                       data_path: str = Query(default='data.csv',
                       description='Путь к файлу с данными')):

    '''Обучение случайного леса'''

    return fit_and_save_model(number, hps.dict(), data_path, 'random_forest')


@app.delete('/delmodel', tags=['Удаление моделей'], status_code=200)
async def del_model(typ: ModelName = Query(),
                    number: int = Query(default=0,
                                        ge=0,
                                        example=0,
                                        description='Уникальный \
                                             идентификатор модели')):

    '''Удаление модели'''

    if not os.path.exists(os.path.join("models", f"{typ.value}_{number}.pkl")):
        raise HTTPException(status_code=404, detail="Item not found")

    os.remove(os.path.join("models", f"{typ.value}_{number}.pkl"))


@app.put('/predict', status_code=200, tags=['Получение предсказаний'])
async def predict(typ: ModelName = Query(),
                  number: int = Query(
        default=0,
        ge=0,
        example=0,
        description='Уникальный идентификатор модели'),
        data_path: str = Query(default='data.csv',
                               description='Путь к файлу с данными')):

    '''Получение прогноза модели'''
    with open(os.path.join("models", f"{typ.value}_{number}.pkl"),
              'rb') as file:
        model = dill.load(file)
        data_table = pd.read_csv(check_data_path(data_path))
        return {'predictions':
                list(model.predict(data_table[data_table.columns[:-1]]))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
