# ===========================================================
#  DOCKERFILE OPTIMIZADO - API DE PREDICCIÓN KNN
# ===========================================================

FROM python:3.11-slim

# Crear directorio de trabajo
WORKDIR /app

# Actualizar pip y dependencias del sistema (solo lo necesario)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Mejor práctica: actualizar pip antes de instalar requirements
RUN pip install --upgrade pip

# Copiar requirements primero (mejor cache Docker)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app.py .
COPY voter_intentions_3000.csv .

# Railway detecta este puerto automáticamente
EXPOSE 8000

# Evita que Python use buffer (mejor logging)
ENV PYTHONUNBUFFERED=1

# Comando de ejecución
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
