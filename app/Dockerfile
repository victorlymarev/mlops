FROM python:3.10

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN pip install poetry

WORKDIR /app
COPY . /app

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction

CMD [ "uvicorn",  'app:app', "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000