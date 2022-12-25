from sqlalchemy import Boolean, Column, Integer, Float
from sqlalchemy.orm import relationship

from database import Base


class LinRegDb(Base):
    __tablename__ = "linreg"

    model_id = Column('model_id', Integer, primary_key=True)
    alpha = Column('alpha', Float)
    fit_intercept = Column('fit_intercept', Boolean)
    random_state = Column('random_state', Integer)


class RandomForestDb(Base):
    __tablename__ = "random_forest"

    imodel_id = Column('model_id', Integer, primary_key=True)
    n_estimators = Column("n_estimators", Integer)
    max_depth = Column("max_depth", Integer)
    random_state = Column('random_state', Integer)
    n_jobs = Column('n_jobs', Integer)
