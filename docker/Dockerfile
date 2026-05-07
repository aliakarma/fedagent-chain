# FedAgent-Chain Research Container
# Base: PyTorch official image with CUDA 11.8 support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="Toqeer Ali Syed <t.syed@institution.edu>"
LABEL description="FedAgent-Chain: Secure Federated and Agentic AI Framework"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/fedagent-chain/fedagent-chain"

# Set working directory
WORKDIR /fedagent-chain

# Install system dependencies (minimised for layer caching efficiency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (maximise Docker layer cache reuse)
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY nodes/ ./nodes/
COPY data/synthetic/ ./data/synthetic/

# Create output directories (actual data mounted as volumes at runtime)
RUN mkdir -p data/processed experiments/runs experiments/results experiments/figures \
    blockchain_storage logs

# Install the package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Set environment variables
ENV PYTHONPATH="/fedagent-chain:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1
ENV PYTHONHASHSEED=42

# Expose ports for API and dashboard
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command: verify installation
CMD ["python", "-c", "import src; print('FedAgent-Chain v1.0.0 — container ready')"]
