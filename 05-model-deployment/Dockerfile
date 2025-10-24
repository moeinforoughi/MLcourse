FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ./Pipfile ./Pipfile.lock /app/

RUN pipenv install --system --deploy

COPY ./predict.py ./model_c=1.0.bin /app/

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]