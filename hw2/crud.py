from sqlalchemy.orm import Session

import modelsdb, validation


def create_lin_reg(db: Session, hps: validation.HpsLinearReg, number: int):
    db_item = validation.Item(**hps.dict(), number=number)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
