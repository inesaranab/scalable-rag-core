# The sandbox's own image: minimal surface, non-root, one job.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root: code escaping exec still cannot touch the system.
RUN useradd -m -u 1000 sandbox_user

WORKDIR /app
# Dependencies before code: they change rarely, code changes constantly,
# and a changed layer invalidates every layer after it.
RUN pip install --no-cache-dir fastapi uvicorn
COPY runner.py .

USER sandbox_user
EXPOSE 8080
CMD ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "8080"]
