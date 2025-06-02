FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN conda create -n app_env python=3.12 -y
SHELL ["/bin/bash", "-c"]

RUN conda init bash && \
    echo "conda activate app_env" >> ~/.bashrc && \
    . ~/.bashrc && \
    pip install --no-cache-dir faiss-cpu==1.11.0 && \
    pip install --no-cache-dir sentence-transformers==4.1.0

COPY requirements.txt ./requirements.txt
RUN . ~/.bashrc && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models data && chmod 777 data

COPY models/ ./models/
COPY download_models.py .
RUN . ~/.bashrc && python download_models.py

COPY . .

EXPOSE 5001

ENV USE_CUDA=0
ENV PORT=5001

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

CMD ["conda", "run", "-n", "app_env", "python", "app.py"] 