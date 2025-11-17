# ===========================================================
#  DOCKERFILE PARA API DE PREDICCIÓN KNN
# ===========================================================

FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app.py .
COPY voter_intentions_3000.csv .

# Exponer puerto
EXPOSE 80

# Variables de entorno
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
