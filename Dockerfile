# Usamos una imagen base oficial de Python
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el requirements.txt para aprovechar la caché de capas de Docker
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido del proyecto al directorio de trabajo
COPY . .

# Variable de entorno para asegurarse que Python no genere .pyc y flush output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Por defecto, al ejecutar el contenedor correrá el pipeline principal
CMD ["python", "pipeline/main_pipeline.py"]
