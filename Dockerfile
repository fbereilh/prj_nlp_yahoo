# Use conda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

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
COPY requirements_working.txt ./requirements.txt

# Install remaining Python dependencies
RUN . ~/.bashrc && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p models

# Create directory for data if it doesn't exist
RUN mkdir -p data

# Expose the port the app runs on
EXPOSE 5001

# Set the default command to run the application
CMD ["conda", "run", "-n", "app_env", "python", "app.py"] 