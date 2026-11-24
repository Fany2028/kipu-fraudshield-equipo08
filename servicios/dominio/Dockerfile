FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/modelo

WORKDIR /app

COPY servicios/dominio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Archivos necesarios para la inferencia local
COPY modelo/__init__.py ./modelo/__init__.py
COPY modelo/predecir.py ./modelo/predecir.py
COPY modelo/kipu_fraudshield_ep3_conservador.joblib ./modelo/kipu_fraudshield_ep3_conservador.joblib
COPY modelo/kipu_fraudshield_ep3_metadata.json ./modelo/kipu_fraudshield_ep3_metadata.json

# Cliente para la futura integración con SageMaker
COPY modelo/cliente_sagemaker.py ./modelo/cliente_sagemaker.py

COPY servicios/dominio/app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
