FROM python:3.11-slim

WORKDIR /app

COPY ./src ./src
COPY ./models ./models
COPY ./requirements.txt .

ENV APP_MODE="prod"

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
