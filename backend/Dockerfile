FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (psycopg binary wheel should be enough; no build tools needed)
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir uv pip && \
    pip install --no-cache-dir -e .

# Copy app
COPY app /app/app

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
