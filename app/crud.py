from sqlalchemy.orm import Session

import models


def create_linear_regression(db: Session, model_params: dict):
    db_model = models.LinRegDb(
        model_id=model_params['model_id'],
        alpha=model_params['alpha'],
        fit_intercept=model_params['fit_intercept'],
        random_state=model_params['random_state']
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def create_random_forest(db: Session, model_params: dict):
    db_model = models.RandomForestDb(
        model_id=model_params['model_id'],
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        random_state=model_params['random_state'],
        n_jobs=model_params['n_jobs']
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def del_model(db: Session, model_name, model_id):
    db_names = {
        'Линейная регрессия': models.LinRegDb,
        'Деревья решений': models.RandomForestDb
    }
    db.query(db_names[model_name]).filter(
        db_names[model_name].model_id == model_id).delete()
    db.commit()
