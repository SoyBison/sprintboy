FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3-dev

# Copy dependency files first (cached layer)
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY src/ ./src/

# Install dependencies and project
RUN uv sync --frozen --no-dev

# Run with uv
CMD ["uv", "run", "python", "-m", "bot.main"]
