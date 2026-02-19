FROM python:3.8-bullseye

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk-headless \
    build-essential \
    libgtk-3-dev \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Definir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

env NUMBA_DISABLE_JIT=0
env NUMBA_CACHE_DIR=/tmp/numba_cache

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "main.py"]
