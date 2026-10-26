# The API (FastAPI + the LangGraph pipeline). MongoDB Atlas and Qdrant Cloud
# are external services reached over the internet; Ollama runs on the host by
# default (see docker-compose.yml). Secrets come from .env at run time and
# are never baked into the image (.dockerignore excludes .env).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
