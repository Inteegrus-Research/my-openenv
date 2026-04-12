FROM python:3.12-slim

WORKDIR /app

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY env/ ./env/
COPY tasks/ ./tasks/
COPY graders/ ./graders/
COPY server/ ./server/
COPY fixtures/ ./fixtures/
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]