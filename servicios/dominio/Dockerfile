FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/modelo

WORKDIR /app

COPY servicios/dominio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modelo/cliente_sagemaker.py ./modelo/cliente_sagemaker.py
COPY modelo/__init__.py ./modelo/__init__.py
COPY servicios/dominio/app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]