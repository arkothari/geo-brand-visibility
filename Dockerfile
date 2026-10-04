FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Cloud platforms inject $PORT; default to 8001 for local docker run
EXPOSE 8001
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"
