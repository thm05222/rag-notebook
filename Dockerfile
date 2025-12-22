# Build stage
FROM python:3.12-slim-bookworm AS builder

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies required for building certain Python packages
# Add Node.js 20.x LTS for building frontend
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gcc g++ git make \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set build optimization environment variables
ENV MAKEFLAGS="-j$(nproc)"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set the working directory in the container to /app
WORKDIR /app

# Copy dependency files and minimal package structure first for better layer caching
COPY pyproject.toml uv.lock ./
COPY open_notebook/__init__.py ./open_notebook/__init__.py

# Install dependencies with optimizations (this layer will be cached unless dependencies change)
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . /app

# Install frontend dependencies and build
WORKDIR /app/frontend

# Build argument for Turnstile (passed at build time)
ARG NEXT_PUBLIC_TURNSTILE_SITE_KEY=""
ENV NEXT_PUBLIC_TURNSTILE_SITE_KEY=${NEXT_PUBLIC_TURNSTILE_SITE_KEY}

RUN npm ci
RUN npm run build

# Return to app root
WORKDIR /app

# Runtime stage
FROM python:3.12-slim-bookworm AS runtime

# Install only runtime system dependencies (no build tools)
# Add Node.js 20.x LTS for running frontend
# Add Tesseract OCR for content-core OCR support
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ffmpeg \
    supervisor \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official method
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container to /app
WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY --from=builder /app /app

# Copy cc_config.yaml for content-core OCR configuration
# Note: File is also mounted via docker-compose.yml, but included in image as fallback
COPY cc_config.yaml /app/cc_config.yaml

# Copy built frontend from builder stage
# Note: Next.js standalone mode requires static files to be accessible
# from both the standalone directory (for server.js) and the root .next directory
COPY --from=builder /app/frontend/.next/standalone /app/frontend/
COPY --from=builder /app/frontend/.next/static /app/frontend/.next/static
# Also copy static files to standalone/.next/static since server.js runs from standalone/ directory
COPY --from=builder /app/frontend/.next/static /app/frontend/.next/standalone/.next/static
COPY --from=builder /app/frontend/public /app/frontend/public

# Expose ports for Frontend and API
EXPOSE 8188 5055

RUN mkdir -p /app/data

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create log directories
RUN mkdir -p /var/log/supervisor

# Runtime API URL Configuration
# The API_URL environment variable can be set at container runtime to configure
# where the frontend should connect to the API. This allows the same Docker image
# to work in different deployment scenarios without rebuilding.
#
# If not set, the system will auto-detect based on incoming requests.
# Set API_URL when using reverse proxies or custom domains.
#
# Example: docker run -e API_URL=https://your-domain.com/api ...

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
