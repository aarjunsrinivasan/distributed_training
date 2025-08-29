# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Create a non-root user for running tests
RUN useradd -m -s /bin/bash testuser && \
    chown -R testuser:testuser /workspace

# Switch to non-root user
USER testuser

# Set environment variables for distributed training
ENV MASTER_ADDR=localhost
ENV MASTER_PORT=12357
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1

# Default command
CMD ["python", "test_all_ddp.py", "--mode", "test"]
