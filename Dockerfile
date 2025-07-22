# ---- Etapa 1: Builder (Sin cambios) ----
# Usamos una imagen completa para instalar dependencias con Poetry
# FROM python:3.12-slim as builder
FROM python:3.12-slim as base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

FROM base

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el entorno virtual con las dependencias ya instaladas desde la etapa 'builder'
# COPY --from=builder /app/.venv ./.venv
COPY --from=base /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copiamos los artefactos y el código fuente necesarios
COPY src/ ./src/

# Exponemos el puerto de la API
EXPOSE 8000

# Comando para iniciar el servidor. Uvicorn se encuentra gracias a la variable PATH.
CMD ["uvicorn", "src.geocodificador_ine_mvp.main:app", "--host", "0.0.0.0", "--port", "8000"]