from flask import Flask
from flask_restx import Api, Resource
import json
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import dill
# я здесь много чего еще доделаю и переделаю, пока этот код работает,
# но остальные пункты не сделаны
# Создаем API
app = Flask(__name__)
api = Api(app)

parser = api.parser()
parser.add_argument('Путь к файлу с данными', type=str, default='data.csv')
parser.add_argument('Название модели при сохранении',
                    type=str,
                    default="model_1")
parser.add_argument('Название модели',
                    choices=['Линейная регрессия', 'Решающий лес'],
                    default="Линейная регрессия",
                    required=True)
parser.add_argument('Путь к json файлу с гиперпараметрами',
                    type=str,
                    required=False,
                    default='parametrs/params.json')


@api.route('/train', methods=['PUT'], doc={'description': 'Обучите модель'})
@api.expect(parser)
class Train(Resource):

    @api.doc(params={
        'Путь к файлу с данными': 'csv файл, с колонкой "y" для таргета'
    })
    def put(self):
        args = parser.parse_args()
        model = self.get_model(args)
        print('\n\n\n', model, '\n\n\n')
        data = pd.read_csv(args['Путь к файлу с данными'])
        feats = [i for i in data.columns if i != 'y']
        model.fit(data[feats], data['y'])
        if not os.path.exists('models'):
            os.makedirs('models')
        with open(f'models/{args["Название модели при сохранении"]}.pkl',
                  'wb') as file:
            dill.dump(model, file)
        return 'Успех'

    def get_model(self, args):
        json_path = args['Путь к json файлу с гиперпараметрами']
        with open(json_path, 'r') as file:
            params = json.load(file)
        if args['Название модели'] == 'Линейная регрессия':
            return Ridge(**params)
        return RandomForestRegressor(**params)


@api.route('/get_avalable_models', methods=['GET'])
class ShowAvalableModels(Resource):

    def get(self):
        return """К обучению доступны 2 модели:
                Линейная регрессия
                Решающий лес"""


@api.route('/delete',
           methods=['DELETE'],
           doc={'description': 'Удалите модель'})
@api.expect(parser)
class Delete(Resource):

    def delete(self):
        args = parser.parse_args()
        os.remove(f'models/{args["Название модели при сохранении"]}.pkl')
        return 'Успешно'


@api.route('/predict', methods=['POST'])
@api.expect(parser)
class Predict(Resource):

    def post(self):
        args = parser.parse_args()
        with open(f'models/{args["Название модели при сохранении"]}.pkl',
                  'rb') as file:
            model = dill.load(file)
        data = pd.read_csv(args['Путь к файлу с данными'])
        feats = [i for i in data.columns if i != 'y']
        preds = model.predict(data[feats]).tolist()
        return preds


if __name__ == '__main__':
    app.run(debug=True)
