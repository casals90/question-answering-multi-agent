FROM jupyter/base-notebook:latest

# expose default Jupyter port
EXPOSE 8888

# Switch to root to install packages and set permissions
USER root

# Install system dependencies (ffmpeg) first
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY config/requirements.txt /app/config/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools && \
    pip install backports.tarfile && \
    pip install -r /app/config/requirements.txt

# Create Hugging Face cache directory and set permissions
RUN mkdir -p /home/jovyan/.cache/huggingface && \
    chown -R jovyan:users /home/jovyan/.cache && \
    chmod -R 775 /home/jovyan/.cache


COPY config/settings.yaml /config/settings.yaml

# Switch back to jovyan user
USER jovyan

WORKDIR /home/jovyan/work
