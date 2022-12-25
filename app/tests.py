from fastapi.testclient import TestClient
from app import app
from database import get_db


client = TestClient(app)


def test_get():
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "Модель 1": 'Линейная регрессия',
        "Модель 2": 'Деревья решений'
    }


# Тут не указан файл с загрузкой
def test_train():
    response = client.post("/train",
                           model_id=1,
                           model_name='Линейная регрессия',
                           alpha=9,
                           fit_intercept=True,
                           random_state=1)
    assert response.status_code == 500


# перед этим надо создать модель с такими параметрами
def test_del_model():
    response = client.delete(
        '/delmodel',
        model_name='Линейная регрессия',
        model_id=1,
        db = get_db
    )
    assert response.status_code == 204

# уже несуществующая модель
def test_del_model():
    response = client.delete(
        '/delmodel',
        model_name='Линейная регрессия',
        model_id=1,
        db = get_db
    )
    assert response.status_code == 404
