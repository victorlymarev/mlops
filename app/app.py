from database import get_db
from train import fit_model
# import uvicorn
from fastapi import Depends, FastAPI, UploadFile, Query, File
from sqlalchemy.orm import Session

import models
from database import engine
from delete_model import delete_model
from predict import make_prediction

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


@app.get("/models",
         tags=['Получение доступных для обучения моделей'],
         status_code=200)
def get_avalable_models():
    """Классы моделей, доступные для обучения"""

    return {"Модель 1": 'Линейная регрессия',
            "Модель 2": 'Деревья решений'}


# здесь передача гиперпараметров сделана через параметры url запроса.
# конечно хорошо было бы сделать так, чтобы передовать их через json
# но это ограничение http и одновременно передавать json и файл сданными нельзя
# В принципе здесь можно изменить Query на Form и ничего не поменяется
# только запрос будет обрабатываться через форму, но я почему-то сделал так
# еще был способ сделать через Depends и использовать pydantic,
# но я не совсем понял как это работает, и переписал так
@app.post("/train", tags=['Обучение моделей'], status_code=201)
def create_user(data: UploadFile = File(description='Файл в формате csv,\
                 последняя колонка - таргет'),
                model_id: int = Query(ge=0),
                model_name: str = Query(example='Линейная регрессия'),
                alpha: float = Query(
                    default=None, description='Параметр линейной регрессии'),
                fit_intercept: bool = Query(
                    default=None, description='Параметр линейной регрессии'),
                random_state: int = Query(
                    default=None, description='Параметр линейной регрессии\
                         и деревьев решений'),
                max_depth: int = Query(
                    default=None, description='Параметр деревьев решений'),
                n_estimators: int = Query(
                    default=None, description='Параметр деревьев решений'),
                n_jobs: int = Query(
                    default=None, description='Параметр деревьев решений'),
                db: Session = Depends(get_db)):
    param_dict = {
        'model_id': model_id,
        'model_name': model_name,
        'alpha': alpha,
        'fit_intercept': fit_intercept,
        'random_state': random_state,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'n_jobs': n_jobs
    }
    fit_model(param_dict, data.file, db)
    return {'model_name': model_name, 'model_id': model_id}


@app.delete('/delmodel', tags=['Удаление моделей'], status_code=204)
async def del_model(model_name: str = Query(example='Линейная регрессия'),
                    model_id: int = Query(ge=0,
                                          example=0,
                                          description='Уникальный \
                                             идентификатор модели'),
                    db: Session = Depends(get_db)):

    '''Удаление модели'''
    delete_model(db, model_name, model_id)


# здесь файл берется тот же самый что и для обучения, 
# поэтому указал, что последняя колонка должна быть пустой
@app.put('/predict', status_code=200, tags=['Получение предсказаний'])
def predict(model_name: str = Query(example='Линейная регрессия'),
            model_id: int = Query(ge=0,
                                  example=0,
                                  description='Уникальный \
                                             идентификатор модели'),
            data: UploadFile = File(description='Файл в формате csv,\
             последняя колонка - таргет или пустой столбец')):
    return make_prediction(model_name, model_id, data.file)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
