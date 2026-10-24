FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scoring ./scoring
COPY data ./data
# Модель обучается при сборке образа (около 20 секунд), поэтому в git её нет.
RUN python -m scoring.train

RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
