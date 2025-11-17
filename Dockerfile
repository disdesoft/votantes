# ===========================================================
#  DOCKERFILE PARA API DE PREDICCIÓN KNN
# ===========================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY voter_intentions_3000.csv .

# Railway detecta este puerto automáticamente
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
