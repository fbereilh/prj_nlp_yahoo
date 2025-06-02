# Use conda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n app_env python=3.12 -y
SHELL ["/bin/bash", "-c"]

# Activate conda environment and install dependencies
RUN conda init bash && \
    echo "conda activate app_env" >> ~/.bashrc && \
    . ~/.bashrc && \
    pip install --no-cache-dir faiss-cpu==1.11.0 && \
    pip install --no-cache-dir sentence-transformers==4.1.0

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Install remaining Python dependencies
RUN . ~/.bashrc && \
    pip install --no-cache-dir -r requirements.txt

# Create directory for models and data
RUN mkdir -p models data && \
    chmod 777 data  # Ensure data directory is writable

# Copy model files and setup script
COPY models/ ./models/
COPY download_models.py .

# Run model setup script
RUN . ~/.bashrc && python download_models.py

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Set environment variable to use CPU by default
ENV USE_CUDA=0

# Set the default command to run the application
CMD ["conda", "run", "-n", "app_env", "python", "app.py"] 