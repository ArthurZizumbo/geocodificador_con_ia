# Usa una imagen base de Python slim para mantener el tamaño bajo
FROM python:3.12-slim AS final

# Crea un usuario no-root por seguridad y establece el directorio de trabajo
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

# Previene que Python escriba archivos .pyc y asegura que la salida no se almacene en búfer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copia el archivo de requerimientos (que ya no tiene torch) y asígnale el propietario correcto
COPY --chown=appuser:appuser requirements.txt .

# Crea y activa el entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. Instala las dependencias desde nuestro archivo de requerimientos limpio.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente de la aplicación y asigna el propietario
COPY --chown=appuser:appuser ./src ./src

# Cambia al usuario no-root para la ejecución del proceso
USER appuser

# Expone el puerto que la aplicación usará
EXPOSE 8000

# Define el comando para correr la aplicación
CMD ["uvicorn", "src.geocodificador_ine_mvp.main:app", "--host", "0.0.0.0", "--port", "8000"]