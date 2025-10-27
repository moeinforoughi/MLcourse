FROM python:3.13.5-slim-bookworm

RUN pip install uv

WORKDIR /code

COPY .python-version pyproject.toml uv.lock /code/

COPY pipeline_v1.bin predict.py /code/

RUN uv sync 

CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
