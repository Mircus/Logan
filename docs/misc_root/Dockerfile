# Minimal, reproducible build for Logical-GANs
FROM python:3.11-slim

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \        build-essential \    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire repo (keep .dockerignore in repo to slim the context)
COPY . .

# Upgrade pip and install package if a standard build file exists
RUN python -m pip install --no-cache-dir --upgrade pip && \    ( [ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "setup.cfg" ] ) && \    python -m pip install --no-cache-dir -e . || true

# Default: simple check message; override with `docker run ... <cmd>`
CMD ["python", "-c", "print('Logical-GANs container ready. Override CMD to run your entrypoint.')"]
