FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip && python -m pip install -e .[dev]
CMD ["logical-gans-repro", "--property", "bipartite", "--quick"]
