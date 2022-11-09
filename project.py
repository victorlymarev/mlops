from flask import Flask
from flask_restx import Api, Resource
import json
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import dill

# Если что-то не работает, то лучше указывать абсолютные пути

app = Flask(__name__)
api = Api(app)

parser = api.parser()
parser.add_argument('Data path',
                    help='Путь к файлу с данными в формате csv, со столбцом y',
                    type=str,
                    default='data.csv')
parser.add_argument('Unique model name',
                    help='Название модели при сохранении',
                    type=str,
                    default="model_1")
parser.add_argument('model type',
                    help='Название типа модели',
                    choices=['Линейная регрессия', 'Решающий лес'],
                    default="Линейная регрессия",
                    required=True)
parser.add_argument('json path',
                    help='Путь к json файлу с гиперпараметрами',
                    type=str,
                    required=False,
                    default='parametrs/params.json')


@api.route('/train', methods=['PUT'], doc={'description': 'Обучите модель'})
@api.expect(parser)
class Train(Resource):
    '''Класс для обучения модели.
    Для того, чтобы обучить модель необходимо заполнить все поля,
    предложенные для заполнения
    '''

    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def put(self):
        args = parser.parse_args()
        model = self.get_model(args)
        data = pd.read_csv(args['Data path'])
        feats = [i for i in data.columns if i != 'y']
        model.fit(data[feats], data['y'])
        if not os.path.exists('models'):
            os.makedirs('models')
        if os.path.isfile(f'models/{args["Unique model name"]}.pkl'):
            with open(f'models/{args["Unique model name"]}.pkl', 'wb') as file:
                dill.dump(model, file)
            return 'Модель с таким именем уже существовала,' + \
                'модель была перезаписна', 200
        with open(f'models/{args["Unique model name"]}.pkl', 'wb') as file:
            dill.dump(model, file)
        return 'Успешно', 200

    def get_model(self, args):
        '''Метод возвращающий модель'''

        json_path = args['json path']
        with open(json_path, 'r') as file:
            params = json.load(file)
        if args['model type'] == 'Линейная регрессия':
            return Ridge(**params)
        return RandomForestRegressor(**params)


@api.route(
    '/get_avalable_models',
    methods=['GET'],
    doc={'description': 'Посмотрите доступные к обучению модели'})
class ShowAvalableModels(Resource):
    '''Класс для просмотра доступных к обучению моделей'''

    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        return """К обучению доступны 2 модели:
                Линейная регрессия
                Решающий лес""", 200
        # Кто-то говорил, что лучше возвращать json файл,
        # добавил эту опцию как комментарий, но не проверял
        # return json.dumps({'модель 1': 'Линейная регрессия',
        #       'модель 2': 'деревья решений'}), 200


@api.route('/delete',
           methods=['DELETE'],
           doc={'description': 'Удалите модель'})
@api.expect(parser)
class Delete(Resource):
    '''Класс для удаления существующих моделей'''

    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def delete(self):
        args = parser.parse_args()
        if not os.path.isfile(f'models/{args["Unique model name"]}.pkl'):
            return 'Такой модели не существует', 400
        os.remove(f'models/{args["Unique model name"]}.pkl')
        return 'Успешно', 200


@api.route('/predict',
           methods=['POST'],
           doc={'description':
                'Получите предсказания обученной модели'})
@api.expect(parser)
class Predict(Resource):
    '''Класс для прогнозов'''

    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def post(self):
        args = parser.parse_args()
        with open(f'models/{args["Unique model name"]}.pkl',
                  'rb') as file:
            model = dill.load(file)
        data = pd.read_csv(args['Data path'])
        feats = [i for i in data.columns if i != 'y']
        preds = model.predict(data[feats]).tolist()
        return preds, 200
        # Кто-то говорил, что лучше возвращать json файл,
        # добавил эту опцию как комментарий, но не проверял
        # return json.dumps({'predirctionns': preds}), 200


if __name__ == '__main__':
    app.run(debug=True)
