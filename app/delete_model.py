from fastapi import HTTPException
from crud import del_model
import os


def delete_model(db, model_name, model_id):
    if not os.path.exists(os.path.join("models", f"{model_name}{model_id}.pkl")):
        raise HTTPException(status_code=404, detail="Item not found")

    os.remove(os.path.join("models", f"{model_name}{model_id}.pkl"))

    del_model(db, model_name, model_id)
