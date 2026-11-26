FROM python:3.12-slim

WORKDIR /app

COPY servicios/worker_llm/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY servicios/worker_llm/ .

CMD ["python", "worker.py"]