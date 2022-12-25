import os
import dill
import pandas as pd


def make_prediction(model_name, model_id, data):
    with open(os.path.join("models", f"{model_name}{model_id}.pkl"),
              'rb') as file:
        model = dill.load(file)
        data_table = pd.read_csv(data)
        return {'predictions':
                list(model.predict(data_table.iloc[:, :-1]))}
