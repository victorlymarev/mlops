from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float
from database import Base


class LinReg(Base):
    __tablename__ = "linreg"

    number = Column(Integer, primary_key=True, index=True)
    alpha = Column(Float)
    fit_intercept = Column(Boolean)
    random_state = Column(Integer)


class RanFor(Base):
    __tablename__ = "random_forest"

    number = Column(Integer, primary_key=True, index=True)
    n_estimators = Column(Integer)
    max_depth = Column(Integer)
    random_state = Column(Integer)
    n_jobs = Column(Integer)